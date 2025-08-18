
import argparse, torch, yaml
from data_loader.loader_vrb_st import make_dataloaders_financial_forBTC
from network.chomo_transformer import ChomoTransformer_forBTC

def load_checkpoint_forBTC(path, device):
    ckpt = torch.load(path, map_location=device)
    return ckpt["model_state"], ckpt["meta"], ckpt["cfg"]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, meta, cfg = load_checkpoint_forBTC(args.checkpoint, device)

    d = cfg["data"]
    _, val_loader, meta2 = make_dataloaders_financial_forBTC(
        csv_path=d["csv_path"],
        x_cols=d["x_cols"],
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
        shuffle_train=False,
    )
    assert meta2["d_in"] == meta["d_in"], "Data dimension mismatch"

    m = cfg["model"]
    model = ChomoTransformer_forBTC(
        d_in=meta["d_in"], d_model=m["d_model"], nhead=m["nhead"],
        num_layers=m["num_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
        out_dim=m["out_dim"], use_last_token=m["use_last_token"]
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loss_fn = torch.nn.SmoothL1Loss()
    vloss = 0.0; nv = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            vloss += loss_fn(model(xb).squeeze(-1), yb).item() * xb.size(0); nv += xb.size(0)
    print(f"[Eval] val_loss={vloss/max(nv,1):.6f} on {nv} samples")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    main(args)
