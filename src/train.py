
from sympy import sec
import os, time, argparse, random, yaml
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch import amp

from data_loader.loader_general import make_dataloaders_financial
from network.chomo_transformer import ChomoTransformer_forBTC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(name)

def _nan_guard(x, name):
    if torch.isnan(x).any() or torch.isinf(x).any():
        n_nan = torch.isnan(x).sum().item()
        n_inf = torch.isinf(x).sum().item()
        print(f"[NaNGuard] {name}: nan={n_nan} inf={n_inf}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    return x

def main(cfg):
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    print(f"[Device] {device}")

    # ==== 0) wandb  ====
    wb = cfg.get("wandb", {})
    use_wandb = bool(wb.get("enabled", False))
    run = None
    if use_wandb:
        run = wandb.init(
            project=wb.get("project", "chomo-trading"),
            dir=wb.get("dir", None),
            # entity=wb.get("entity", None),
            mode=wb.get("mode", "online"),
            config=cfg,                         # record the config so that easy to reproduce
            tags=wb.get("tags", None),
            notes=wb.get("notes", None),
            group=wb.get("group", None),
            name=None
        )

    # === 1) data ===
    d = cfg["data"]
    train_loader, val_loader, meta = make_dataloaders_financial(
        csv_path=d["csv_path"],
        x_cols=d["x_cols"],
        x_z_cols=d.get("x_z_cols", None),
        seq_len=d["seq_len"],
        horizon=d["horizon"],
        y_col=d.get("y_col", None),
        price_col=d.get("price_col", "close"),
        target_kind=d.get("target_kind", "logret"),
        batch_size=d["batch_size"],
        train_ratio=d["train_ratio"],
        gap=d.get("gap", 0),
        num_workers=d["num_workers"],
        pin_memory=True,
        norm=d["norm"],
        rolling_win=d["rolling_win"],
        time_col=d.get("time_col", None),
        shuffle_train=d["shuffle_train"],
    )
    print(f"[Data] d_in={meta['d_in']} L={meta['seq_len']} train={meta['n_train']} val={meta['n_val']}")

    # === 2) model ===
    m = cfg["model"]
    model = ChomoTransformer_forBTC(
        d_in=meta["d_in"], d_model=m["d_model"], nhead=m["nhead"],
        num_layers=m["num_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
        out_dim=m["out_dim"], use_last_token=m["use_last_token"]
    ).to(device)

    # === 3) optim ===
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    loss_fn = nn.SmoothL1Loss()
    use_bf16 = (device.type == "cuda")

    # === 4) I/O ===
    out_dir = cfg["io"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    run_name = time.strftime("%Y%m%d-%H%M%S")
    best_path = os.path.join(out_dir, f"{run_name}_best_forBTC.pt")
    last_path = os.path.join(out_dir, f"{run_name}_last_forBTC.pt")
    best_metric = float("inf")
    if use_wandb and wandb.run is not None:
        wandb.run.name = run_name

    # === 5) training loop ===
    for epoch in range(1, o["epochs"]+1):
        model.train()
        running = 0.0; n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            xb = _nan_guard(xb, "xb_train")
            yb = _nan_guard(yb, "yb_train")

            opt.zero_grad(set_to_none=True)

            # --- train：BF16 autocast GradScaler (no GradScaler) ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(xb).squeeze(-1)   # [B, 1] to [B]
                # see if pred or yb has nan
                _nan_guard(pred, "pred_train")
                _nan_guard(yb, "yb_train")

                loss = loss_fn(pred, yb)
                _nan_guard(loss, "loss_train")

            loss.backward()

            # gradient clipping
            if o.get("grad_clip", 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])

            opt.step()

            # print & accumulate
            # print(f"epoch {epoch} train_loss={loss.item():.6f}")
            running += loss.item() * xb.size(0); n += xb.size(0)

        train_loss = running / max(n, 1)

        # === eval ===
        model.eval(); vloss = 0.0; nv = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                # --- eval：BF16 autocast ---
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    v = loss_fn(model(xb).squeeze(-1), yb).item()
                vloss += v * xb.size(0); nv += xb.size(0)

        val_loss = vloss / max(nv, 1)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # wandb record
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": opt.param_groups[0]["lr"],
                "time/epoch_sec": sec,
                "meta/seq_len": meta["seq_len"],
                "meta/horizon": meta["horizon"],
            }, step=epoch)
            
        # save
        torch.save({"model_state": model.state_dict(), "meta": meta, "cfg": cfg}, last_path)

        # save the best
        if val_loss < best_metric:
            best_metric = val_loss
            torch.save({"model_state": model.state_dict(), "meta": meta, "cfg": cfg}, best_path)
            
            # # save the weight pt file in wandb
            # if use_wandb:
            #     art = wandb.Artifact(name=f"btc_transformer_best_{run_name}", type="model")
            #     art.add_file(best_path)
            #     wandb.log_artifact(art)  # Log the artifact

    print(f"[Done] best {best_metric:.6f} | best_path={best_path} | last_path={last_path}")
    
    if use_wandb:
        wandb.summary["best/val_loss"] = best_metric
        wandb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
