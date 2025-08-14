
from __future__ import annotations

import numpy as np
import pandas as pd

from .ta_base import BaseTA, PriceColumnMap


class VolatilityReversionBands(BaseTA):
    """
    Translation of your Pine script to compute VRB values.
    Adds columns with a chosen prefix; returns a DataFrame.
    """

    def compute(
        self,
        length: int = 20,
        mult: float = 2.0,
        atr_mult: float = 1.5,
        src_col: str = "close",
        weak_signals: bool = True,
        prefix: str = "vrb",
        inplace: bool = False,
        stdev_biased: bool = True,
    ) -> pd.DataFrame:
        df = self.df if inplace else self.df.copy()

        close = df[src_col].astype(float)

        basis = self.sma(src_col, length)
        st = self.stdev(src_col, length, biased=stdev_biased)
        atr_val = self.atr(length)

        bb_upper = basis + st * float(mult)
        bb_lower = basis - st * float(mult)
        atr_band = atr_val * float(atr_mult)

        rev_upper = bb_upper + atr_band
        rev_lower = bb_lower - atr_band

        long_sig = self.crossunder(close, rev_lower)
        short_sig = self.crossover(close, rev_upper)

        df[f"{prefix}_basis_{length}"] = basis
        df[f"{prefix}_stdev_{length}"] = st
        df[f"{prefix}_bb_upper"] = bb_upper
        df[f"{prefix}_bb_lower"] = bb_lower
        df[f"{prefix}_atr_{length}"] = atr_val
        df[f"{prefix}_atr_band"] = atr_band
        df[f"{prefix}_reversal_upper"] = rev_upper
        df[f"{prefix}_reversal_lower"] = rev_lower
        df[f"{prefix}_long_signal"] = long_sig.astype("bool")
        df[f"{prefix}_short_signal"] = short_sig.astype("bool")

        if weak_signals:
            denom = (rev_upper - rev_lower).clip(lower=1e-6)
            ratio = (close - rev_lower) / denom
            ratio_clamped = ratio.clip(lower=0.0, upper=1.0)

            df[f"{prefix}_ratio"] = ratio
            df[f"{prefix}_ratio_clamped"] = ratio_clamped
            df[f"{prefix}_near_upper"] = (ratio_clamped >= 0.9).astype("bool")
            df[f"{prefix}_near_lower"] = (ratio_clamped <= 0.1).astype("bool")

        if inplace:
            self.df = df
            return self.df
        return df
