import numpy as np

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
