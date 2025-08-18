
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

__all__ = [
    "FinancialCSVWindowDataset",
    "make_dataloaders_financial",
    "time_split_indices",
]

# ---------- helpers ----------
def time_split_indices(n_samples: int, train_ratio: float = 0.8, gap: int = 0):
    """
    Contiguous chronological split.
    Optional 'gap' leaves a gap between train and val to avoid temporal bleed.
    """
    n_train = int(n_samples * train_ratio)
    n_train = max(1, min(n_train, n_samples-1-gap))
    train_idx = range(0, n_train)
    val_idx = range(n_train + gap, n_samples)
    return train_idx, val_idx

def rolling_zscore(x: np.ndarray, win: int = 512, eps: float = 1e-8):
    """
    Rolling z-score using ONLY past data.
    x: [T, d] -> out[t] uses stats of x[:t].
    """
    T, d = x.shape
    out = np.empty_like(x, dtype=np.float64)
    buf = np.zeros((win, d), dtype=np.float64)
    csum = np.zeros((d,), dtype=np.float64)       # size [1, d]
    csum2 = np.zeros((d,), dtype=np.float64)      # size [1, d]
    
    # head for rolling update
    head = 0
    # n for window size
    n = 0

    for t in range(T):
        if t == 0: 
            out[t] = 0.0
            continue

        prev = x[t-1].astype(np.float64) # previous vector in t-1, size [1, d]
        if n < win:
            buf[n] = prev; n += 1 # after this plus 1, the n is equal to t.
            # z score calculation depends on [t-w, t-1] mean and std, so just save the prev to buf
            csum += prev; csum2 += prev*prev
        else:
            old = buf[head]; csum -= old; csum2 -= old*old
            buf[head] = prev; csum += prev; csum2 += prev*prev
            head = (head + 1) % win  # a loop mechanism, update the buf one by one

        mean = csum / n
        var = np.maximum(csum2 / n - mean*mean, 0.0)
        std = np.sqrt(var + eps)
        out[t] = ((x[t] - mean) / std).astype(np.float32)
    return out

def make_future_target(series: np.ndarray, horizon: int = 1, kind: str = "logret", fill='nan'):
    if horizon < 1 or horizon >= len(series):
        raise ValueError("horizon must be >=1 and < len(series)")
    
    # return part
    s = series.astype(np.float64)
    if kind == "logret":
        y = np.log(s[horizon:] / s[:-horizon])
    elif kind == "pct":
        y = (s[horizon:] - s[:-horizon]) / s[:-horizon]
    else:
        raise ValueError(f"Unknown target kind: {kind}")
    
    # fill part
    if fill == 'nan':
        tail = np.full(horizon, np.nan, dtype=y.dtype)
    elif fill == 0:
        tail = np.zeros(horizon, dtype=y.dtype)
    else:
        tail = np.full(horizon, fill, dtype=y.dtype)

    return np.concatenate([y, tail]).astype(np.float32)

# ---------- Dataset ----------
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
        target = self.y[i + self.rwin + self.L - 1]          # y[t] is aligned to t, no need to add h
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
