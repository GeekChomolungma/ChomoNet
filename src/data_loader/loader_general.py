
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from .helper import *

__all__ = [
    "FinancialCSVWindowDataset",
    "make_dataloaders_financial",
]

class FinancialCSVWindowDataset(Dataset):
    """
    Financial time-series dataset for windowed modeling.
    - Features: x_cols
    - Target:
        * if y_col provided -> use that column directly.
        * else -> build from a price column (price_col) with make_future_target(kind, horizon).
    - Normalization:
        * norm='rolling' -> rolling z-score on features (history-only, no leakage).
        * norm=None -> no normalization.
    """
    def __init__(self, csv_path: str,
                 x_cols,
                 x_z_cols: list[str] | None = None,
                 seq_len: int = 96,
                 horizon: int = 1,
                 y_col: str | None = None,
                 price_col: str | None = "close",
                 target_kind: str = "logret",
                 norm: str | None = "rolling",
                 rolling_win: int = 512,
                 time_col: str | None = None):
        df = pd.read_csv(csv_path)

        # drop the original df rows who has NaN in any of its columns
        first_ok = df.dropna().index[0] if df.isna().any().any() else 0
        if first_ok > 0:
            df = df.iloc[first_ok:].reset_index(drop=True)

        # convert time column to datetime
        if time_col is not None and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.sort_values(time_col).reset_index(drop=True)

        X_df = df[x_cols].copy()
        X_df = X_df.astype(np.float32)

        self.rwin = 0
        if norm == "rolling" and x_z_cols:
            self.rwin = rolling_win
            zcols = [c for c in x_z_cols if c in X_df.columns]
            if len(zcols) > 0:
                arr = X_df[zcols].to_numpy(dtype=np.float32)
                arr = rolling_zscore(arr, win=rolling_win)    # 只对 zcols 做 z
                X_df.loc[:, zcols] = arr
        X = X_df.to_numpy(dtype=np.float32)

        if y_col is not None:
            y = df[y_col].to_numpy(dtype=np.float32)

            # --- strictly check, ensuring user defined y_col has a NaN tail, which means alignment future Y to current X ---
            if horizon >= len(y):
                raise ValueError(f"horizon ({horizon}) must be < len(y) ({len(y)}).")

            tail = y[-horizon:]
            if not np.all(np.isnan(tail)):
                bad = np.where(~np.isnan(tail))[0].tolist()
                raise ValueError(
                    f"y_col '{y_col}' must be NaN for the last {horizon} rows "
                    f"(future targets). Non-NaN at tail offsets: {bad}"
                )

            if np.isnan(y[:-horizon]).any():
                raise ValueError(
                    f"y_col '{y_col}' contains NaN before the last {horizon} rows; "
                    "please clean or impute earlier NaNs."
                )

        else:
            assert price_col in df.columns, f"price_col '{price_col}' not in CSV"
            y = make_future_target(
                df[price_col].to_numpy(dtype=np.float32),
                horizon=horizon,
                kind=target_kind,
                fill='nan'
            )

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.L = int(seq_len); self.h = int(horizon)

    def __len__(self):
        return len(self.X) - self.rwin - self.L - self.h + 1

    def __getitem__(self, i):
        xs = self.X[i + self.rwin : i + self.rwin + self.L]                  # [L, d_in], last step is t = i+L-1
        target = self.y[i + self.rwin + self.L - 1]                          # y[t] is aligned to t, no need to add h
        return xs, target

# ---------- factory ----------
def make_dataloaders_financial(csv_path: str, x_cols: list[str], 
                               x_z_cols: list[str] | None = None,
                               seq_len=96, horizon=1,
                               y_col=None, price_col="close", target_kind="logret",
                               batch_size=128, train_ratio=0.8, gap=0,
                               num_workers=0, pin_memory=True,
                               norm="rolling", rolling_win=512,
                               time_col=None, shuffle_train=False):
    ds = FinancialCSVWindowDataset(csv_path, x_cols, x_z_cols, seq_len, horizon,
                                   y_col=y_col, price_col=price_col, target_kind=target_kind,
                                   norm=norm, rolling_win=rolling_win, time_col=time_col)
    train_idx, val_idx = time_split_indices(len(ds), train_ratio=train_ratio, gap=gap)
    train_ds = Subset(ds, list(train_idx))
    val_ds = Subset(ds, list(val_idx))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    meta = {"d_in": ds.X.shape[1], "seq_len": seq_len, "horizon": horizon,
            "n_train": len(train_ds), "n_val": len(val_ds)}
    return train_loader, val_loader, meta
