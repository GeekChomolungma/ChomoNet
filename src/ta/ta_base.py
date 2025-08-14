
from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd

SeriesLike = Union[pd.Series, str]


@dataclass
class PriceColumnMap:
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    time: str = "starttime"


class BaseTA:
    """
    Minimal base class for TA indicators operating on a kline DataFrame.
    Contains Pine-style primitives to be reused by derived indicators.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columns: PriceColumnMap = PriceColumnMap(),
        sort_by_time: bool = True,
        coerce_numeric: bool = True,
    ) -> None:
        self.columns = columns
        self.df = df.copy()
        self._ensure_required_columns()

        if coerce_numeric:
            for col in [columns.open, columns.high, columns.low, columns.close, columns.volume]:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if sort_by_time and columns.time in self.df.columns:
            self.df = self.df.sort_values(columns.time).reset_index(drop=True)

    # -----------------------
    # Core Pine-style ops
    # -----------------------
    def series(self, series_or_col: SeriesLike) -> pd.Series:
        if isinstance(series_or_col, pd.Series):
            s = series_or_col.reindex(self.df.index)
        else:
            s = self.df[series_or_col]
        return pd.to_numeric(s, errors="coerce")

    def sma(self, source: SeriesLike, length: int) -> pd.Series:
        s = self.series(source)
        return s.rolling(window=length, min_periods=length).mean()

    def ema(self, source: SeriesLike, length: int) -> pd.Series:
        s = self.series(source)
        # Pine's EMA uses ewm with adjust=False
        return s.ewm(span=length, adjust=False, min_periods=length).mean()

    def stdev(self, source: SeriesLike, length: int, biased: bool = True) -> pd.Series:
        s = self.series(source)
        ddof = 0 if biased else 1
        return s.rolling(window=length, min_periods=length).std(ddof=ddof)

    def atr(self, length: int) -> pd.Series:
        c = self.columns
        high = self.df[c.high].astype(float)
        low = self.df[c.low].astype(float)
        close = self.df[c.close].astype(float)
        prev_close = close.shift(1)
        tr = np.maximum.reduce([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ])
        return self._wilder_rma(pd.Series(tr, index=self.df.index, dtype=float), length)

    @staticmethod
    def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
        a_prev = a.shift(1)
        b_prev = b.shift(1)
        out = (a_prev <= b_prev) & (a > b)
        return out.fillna(False)

    @staticmethod
    def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
        a_prev = a.shift(1)
        b_prev = b.shift(1)
        out = (a_prev >= b_prev) & (a < b)
        return out.fillna(False)

    # -----------------------
    # Helpers
    # -----------------------
    def _ensure_required_columns(self) -> None:
        c = self.columns
        required = [c.open, c.high, c.low, c.close]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required kline columns: {missing}. Present: {list(self.df.columns)}")

    @staticmethod
    def _wilder_rma(series: pd.Series, length: int) -> pd.Series:
        s = pd.Series(series, dtype=float).copy()
        r = pd.Series(np.nan, index=s.index, dtype=float)
        if length <= 0:
            raise ValueError("length must be positive")
        if len(s) == 0:
            return r

        sma = s.rolling(window=length, min_periods=length).mean()
        first_idx = sma.first_valid_index()
        if first_idx is None:
            return r

        start_pos = sma.index.get_loc(first_idx)
        r.iloc[start_pos] = sma.loc[first_idx]
        for i in range(start_pos + 1, len(s)):
            r.iat[i] = (r.iat[i - 1] * (length - 1) + s.iat[i]) / length
        return r
