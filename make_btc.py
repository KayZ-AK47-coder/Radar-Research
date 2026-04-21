"""Generate synthetic but realistic BTC hourly bars for the backtest."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
n = 24 * 365 * 2   # 2 years of hourly bars
ts = pd.date_range("2023-01-01", periods=n, freq="h")

# GARCH-ish vol clustering + mild OU drift -> creates tradeable mean reversion
vol = np.zeros(n); vol[0] = 0.010
for i in range(1, n):
    vol[i] = np.sqrt(1e-7 + 0.88 * vol[i-1]**2 + 0.08 * rng.normal()**2 * vol[i-1]**2)

shocks = rng.normal(size=n) * vol
# add a weak OU pull so log-price mean-reverts around a slow drift
log_p = np.zeros(n); log_p[0] = np.log(42_000.0)
drift = 0.00002
for i in range(1, n):
    mr = -0.015 * (log_p[i-1] - (np.log(42_000.0) + drift * i))
    log_p[i] = log_p[i-1] + mr + shocks[i]

close = np.exp(log_p)
df = pd.DataFrame({"ts": ts, "close": close.round(2)})
df.to_csv("btc_hourly.csv", index=False)
print(df.head(), "\n", df.tail())
print(f"rows={len(df)}  min={df.close.min():.0f}  max={df.close.max():.0f}")
