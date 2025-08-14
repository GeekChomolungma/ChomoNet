# supertrend_from_pine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Iterable
import numpy as np
import pandas as pd

from .ta_base import BaseTA, PriceColumnMap


# -----------------------------
# Dataclasses (Equivalent to Pine's type)
# -----------------------------
@dataclass
class Alerts:
    # s: "Sell Signal" (trend changes from Bull to Bear); b: "Buy Signal" (trend changes from Bear to Bull)
    s: pd.Series  # bool
    b: pd.Series  # bool


@dataclass
class SuperTrendOut:
    # Aligned with Pine: type supertrend { float s; int d; }
    s: pd.Series        # supertrend line itself (use lower band for Bull, upper band for Bear)
    d: pd.Series        # direction: +1 means Bear, -1 means Bull
    up: pd.Series       # final upper band (final_up), bearish
    dn: pd.Series       # final lower band (final_dn), bullish


# -----------------------------
# Utility: Pine-style nz / avg
# -----------------------------
def nz(s: Union[pd.Series, float, int], fill: float = 0.0) -> pd.Series:
    if isinstance(s, pd.Series):
        return s.fillna(fill)
    return pd.Series([s], dtype=float).fillna(fill)


def avg(*xs: Iterable[pd.Series]) -> pd.Series:
    # Pine's math.avg with variable number of arguments
    xs = [pd.to_numeric(x, errors="coerce") for x in xs]
    out = xs[0].copy().astype(float)
    for x in xs[1:]:
        out = out.add(x, fill_value=np.nan)
    return out / float(len(xs))


# -----------------------------
# Bar: Derived from BaseTA (core)
# -----------------------------
class Bar(BaseTA):
    """
    Construct a synthetic OHLC "bar" from any price source (e.g., close),
    strictly aligned with Pine's:
        o = nz(src[1])
        h = max(nz(src[1]), src)
        l = min(nz(src[1]), src)
        c = src
    Then implement src()/bar_atr()/st() methods in the bar's own coordinate system.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        src: Union[str, pd.Series] = "close",
        *,
        columns: PriceColumnMap = PriceColumnMap(),
        nz_fill_value: float = 0.0,      # Pine's nz default is 0
    ) -> None:
        super().__init__(df=df, columns=columns)
        # This src is the "input source", different from member function b.src(...)
        s = self.series(src).astype(float)
        prev = s.shift(1)

        prev_nz = prev.fillna(nz_fill_value)
        self.o = prev_nz
        self.h = pd.Series(np.maximum(prev_nz.values, s.values), index=s.index, dtype=float)
        self.l = pd.Series(np.minimum(prev_nz.values, s.values), index=s.index, dtype=float)
        self.c = s

        # # print the nan or zero number of the self.h
        # print(f"Bar: {self.o.isna().sum()} NaN in o, {self.o.isnull().sum()} Null in o")
        # print(f"Bar: {self.h.isna().sum()} NaN in h, {self.h.isnull().sum()} Null in h")
        # print(f"Bar: {self.l.isna().sum()} NaN in l, {self.l.isnull().sum()} Null in l")
        # print(f"Bar: {self.c.isna().sum()} NaN in c, {self.c.isnull().sum()} Null in c")

        # Convenient index alignment
        self.index = self.df.index

    # ---- method src(bar b, simple string src) -> float ----
    def src_method(self, which: str) -> pd.Series:
        """
        Pine's bar.src(...):
            'oc2'   = avg(o, c)
            'hl2'   = avg(h, l)
            'hlc3'  = avg(h, l, c)
            'ohlc4' = avg(o, h, l, c)
            'hlcc4' = avg(h, l, c, c)
        """
        w = which.lower()
        if w == "oc2":
            return avg(self.o, self.c)
        if w == "hl2":
            return avg(self.h, self.l)
        if w == "hlc3":
            return avg(self.h, self.l, self.c)
        if w == "ohlc4":
            return avg(self.o, self.h, self.l, self.c)
        if w == "hlcc4":
            return avg(self.h, self.l, self.c, self.c)
        raise ValueError(f"Unsupported src code: {which}")

    # ---- method atr(bar b, simple int len) -> float ----
    def bar_atr(self, length: int) -> pd.Series:
        """
        Pine version of TR/ATR, calculated **on the bar's own OHLC**:
            tr = na(h[1]) ? (h - l)
                           : max( max(h - l, abs(h - c[1])), abs(l - c[1]) )
            atr = (length == 1) ? tr : ta.rma(tr, length)
        """
        h, l, c = self.h.astype(float), self.l.astype(float), self.c.astype(float)
        c_prev = c.shift(1)
        # First bar's h[1] is NA
        tr_core = pd.concat(
            [
                (h - l).abs(),
                (h - c_prev).abs(),
                (l - c_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        tr = tr_core.copy()
        
        # ta.rma: Wilder RMA
        print(f"bar_atr length: {length}")
        if length == 1:
            atr = tr
        else:
            atr = self._wilder_rma(tr, length)
        return atr

    # ---- method st(bar b, simple float factor, simple int len) -> supertrend ----
    def st(self, factor: float, length: int) -> SuperTrendOut:
        """
        Reproduce Pine's SuperTrend recursion (with final upper/lower bands).
        """
        atr = self.bar_atr(length)
        hl2 = self.src_method("hl2")

        up_basic = hl2 + factor * atr
        dn_basic = hl2 - factor * atr
        up_basic.fillna(0, inplace=True)
        dn_basic.fillna(0, inplace=True)
        
        print(f"Bar: {up_basic.isna().sum()} NaN in up, {up_basic.isnull().sum()} Null in up")
        print(f"Bar: {dn_basic.isna().sum()} NaN in dn, {dn_basic.isnull().sum()} Null in dn")

        # Generate "final upper/lower bands" with stickiness, according to Pine rules
        up = up_basic.copy()
        dn = dn_basic.copy()

        for i in range(1, len(self.index)):
            up_prev = up.iat[i - 1]
            if np.isnan(up_prev):
                up_prev = 0
                
            dn_prev = dn.iat[i - 1]
            if np.isnan(dn_prev):
                dn_prev = 0

            c_prev = self.c.iat[i - 1]

            cur_up_b = up_basic.iat[i]
            cur_dn_b = dn_basic.iat[i]

            up.iat[i] = cur_up_b if (cur_up_b < up_prev) or (c_prev > up_prev) else up_prev
            dn.iat[i] = cur_dn_b if (cur_dn_b > dn_prev) or (c_prev < dn_prev) else dn_prev

        # print up and dn's nan and null number
        print(f"Bar: {up.isna().sum()} NaN in up, {up.isnull().sum()} Null in up")
        print(f"Bar: {dn.isna().sum()} NaN in dn, {dn.isnull().sum()} Null in dn")

        # Direction d and supertrend line s (recursion depends on previous st / up / dn)
        d = pd.Series(np.nan, index=self.index, dtype=float)
        s = pd.Series(np.nan, index=self.index, dtype=float)

        for i in range(len(self.index)):
            if i == 0 or np.isnan(atr.iat[i - 1]):
                d.iat[i] = 1.0  # Initial value: align with Pine's na(atr[1]) => 1 (Bear)
            else:
                st_prev = s.iat[i - 1]
                up_prev = up.iat[i - 1]
                price = self.c.iat[i]

                # st[1] == nz(up[1]) ?
                if np.isfinite(st_prev) and np.isfinite(up_prev) and abs(st_prev - up_prev) < 1e-12:
                    d.iat[i] = -1.0 if price > up.iat[i] else +1.0
                else:
                    d.iat[i] = +1.0 if price < dn.iat[i] else -1.0

            # st := dir == -1 ? dn : up
            s.iat[i] = dn.iat[i] if d.iat[i] == -1.0 else up.iat[i]

        # Convert to integer ±1
        d = d.astype(int)

        return SuperTrendOut(s=s, d=d, up=up, dn=dn)

    # ---- Equivalent to Pine's alerts.new(...) (for feature/label generation for training) ----
    @staticmethod
    def alerts_from_st(st: SuperTrendOut) -> Alerts:
        # Pine: math.sign(ta.change(st.d)) ==  1 -> Sell
        #       math.sign(ta.change(st.d)) == -1 -> Buy
        chg = st.d.diff()
        sign = chg.apply(lambda x: 0 if pd.isna(x) else (1 if x > 0 else (-1 if x < 0 else 0)))
        sell = sign.eq(1)
        buy = sign.eq(-1)
        return Alerts(s=sell, b=buy)


# -----------------------------
# Convenience wrapper: Compute SuperTrend directly from raw df
# -----------------------------
class PineSuperTrend:
    """
    High-level interface: Pass in raw market df and price source column name (default 'close'),
    returns Pine-equivalent supertrend result and signals.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        price_source: Union[str, pd.Series] = "close",
        *,
        columns: PriceColumnMap = PriceColumnMap(),
        nz_fill_value: float = 0.0,
    ) -> None:
        self.bar = Bar(df=df, src=price_source, columns=columns, nz_fill_value=nz_fill_value)

    def compute(self, factor: float = 5.5, length: int = 20) -> tuple[SuperTrendOut, Alerts]:
        st = self.bar.st(factor=factor, length=length)
        alerts = Bar.alerts_from_st(st)
        return st, alerts

    def to_dataframe(
        self,
        st_out,
        alerts,
        prefix: str = "st"
    ) -> pd.DataFrame:
        """
        according the pine original definition:
        d = -1 means dn (bullish), +1 means up (bearish)
        """
        s, d, up, dn = st_out.s, st_out.d, st_out.up, st_out.dn

        out = pd.DataFrame(
            {
                f"{prefix}_line": s.astype(float),
                f"{prefix}_dir": d.astype(int), 
                f"{prefix}_up": up.astype(float),
                f"{prefix}_dn": dn.astype(float),
                f"{prefix}_buy": alerts.b.astype(bool),
                f"{prefix}_sell": alerts.s.astype(bool),
            },
            index=s.index,
        )
        return out
    