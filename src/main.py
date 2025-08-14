# main.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dbio import MongoConfig, MongoWrapper

from ta.ta_vrb import VolatilityReversionBands
from ta.ta_supertrend import PineSuperTrend

# --- 1) Prepare data --------------------------------------------------------
cfg = MongoConfig.from_env()
mw = MongoWrapper(cfg)
df = mw.klines_to_df(mw.fetch_klines("BTCUSDT", "4h"))

# Minimal placeholder protection to avoid errors when no data is available
try:
    df
except NameError:
    raise RuntimeError("Please provide K-line data to variable df (must contain open/high/low/close/volume/starttime).")

# Convert time column to DatetimeIndex (skip if already present)
time_col = "starttime"
if time_col in df.columns:
    # Assume milliseconds timestamp; if in seconds, change to unit='s'
    df[time_col] = pd.to_datetime(df[time_col], unit="ms", errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    x = df[time_col]
else:
    # If no time column, use row index as x-axis
    x = pd.RangeIndex(start=0, stop=len(df), step=1)

# --- 2) Calculate indicators ------------------------------------------------
vrb_inst = VolatilityReversionBands(df)
df_vrb = vrb_inst.compute(
    length=24, mult=2.5, atr_mult=1, src_col="close",
    weak_signals=True, prefix="vrb", inplace=False
)

st_inst = PineSuperTrend(df_vrb, price_source="close")
st, alerts = st_inst.compute(factor=5.5, length=20)
df_st = st_inst.to_dataframe(st, alerts, prefix="st")

# --- 3) Select required columns ---------------------------------------------
df_out = df_vrb.join(df_st, how="left")

close = df_out["close"].astype(float)
# VRB upper/lower boundaries (in the screenshot: purple and light green)
vrb_up = df_out["vrb_reversal_upper"]
vrb_dn = df_out["vrb_reversal_lower"]

# Supertrend: final_up/final_dn + direction
st_up = df_out["st_up"]   # Bear → draw red
st_dn = df_out["st_dn"]   # Bull → draw blue
st_dir = df_out["st_dir"] # -1=Bull, 1=Bear
print(f"length of st_up is {len(st_up)}")
print(f"length of st_dn is {len(st_dn)}")
print(f"length of st_dir is {len(st_dir)}")

# To show lines only during their respective trend segments: mask with NaN
bull_dn = st_dn.where(st_dir == -1)  # Bull: show dn
bear_up = st_up.where(st_dir ==  1)  # Bear: show up
print(f"length of bull_dn is {len(bull_dn)}")
print(f"length of bear_up is {len(bear_up)}")

# Print last 3 rows of bull_dn and bear_up
print(f"bull_dn last 3 ones: \n {bull_dn.tail(3)} \n")
print(f"bear_up last 3 ones: \n {bear_up.tail(3)} \n")

# --- 4) Plotting ------------------------------------------------------------
plt.figure(figsize=(28, 12))

# Print last 3 rows of x index and time, and top 3 as well
print(f"X last 3 ones: \n {x.tail(3)} \n")
print(f"X top 3 ones: \n {x.head(3)} \n")

# Close price thin line (similar to TradingView's background price path)
plt.plot(x, close, lw=1.0, alpha=0.6, label="Close", zorder=2)

# VRB: upper purple, lower light green
plt.plot(x, vrb_up, color="#B000B5", lw=1.6, label="VRB Upper", zorder=3)   # Purple
plt.plot(x, vrb_dn, color="#33C8A3", lw=1.6, label="VRB Lower", zorder=3)   # Light green

# Supertrend: Bull=blue dn; Bear=red up
plt.plot(x, bull_dn, color="#1E88E5", lw=2.0, label="Supertrend Bull (dn)", zorder=4)
plt.plot(x, bear_up, color="#E53935", lw=2.0, label="Supertrend Bear (up)", zorder=4)

# Mark the "last valid point" to ensure the last candle of a new trend is visible
last_bull = bull_dn.last_valid_index()
if last_bull is not None:
    plt.scatter([x.loc[last_bull]], [bull_dn.loc[last_bull]], s=36, color="#1E88E5", zorder=5)

last_bear = bear_up.last_valid_index()
if last_bear is not None:
    plt.scatter([x.loc[last_bear]], [bear_up.loc[last_bear]], s=36, color="#E53935", zorder=5)

# Visual style adjustments to mimic TradingView
plt.grid(alpha=0.15)
plt.title("VRB & Supertrend (approx. TradingView look)")
plt.xlabel("Time" if time_col in df.columns else "Index")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
