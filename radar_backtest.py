"""
radar_backtest.py

Mean-reversion research module from Radar Bot.
Runs on hourly BTC bars, evaluates on held-out period,
validates with stationary-bootstrap Monte Carlo.

Muhammad Khushal Khan

Signal:  z-score of log price vs rolling mean. Enter short when z > +threshold,
         long when z < -threshold, exit on mean crossing. Flat otherwise.
Risk:    fixed fractional sizing, tx costs + slippage netted from returns.
Eval:    block bootstrap MC (stationary bootstrap) for Sharpe CI.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- params (tuned on 2023 in-sample, held fixed for walk-forward) ----
WINDOW      = 72          # lookback bars (hourly -> 3 days)
ENTRY_Z     = 1.8
EXIT_Z      = 0.3
FEE_BPS     = 6           # 3bps per side taker + 3bps slippage, conservative for Binance/OKX
ANN_FACTOR  = np.sqrt(365 * 24)   # hourly bars, 24/7 crypto
MC_PATHS    = 10_000
BLOCK_LEN   = 24          # ~1-day blocks to preserve autocorrelation


def rolling_zscore(x: pd.Series, n: int) -> pd.Series:
    mu = x.rolling(n).mean()
    sd = x.rolling(n).std(ddof=0)
    return (x - mu) / sd.replace(0, np.nan)


def generate_positions(z: pd.Series, entry: float, exit_: float) -> pd.Series:
    """Stateful long/flat/short conversion of z-score. Vectorised where safe;
    the regime step is inherently sequential so we take the hit."""
    pos = np.zeros(len(z))
    state = 0
    zv = z.values
    for i in range(len(zv)):
        if np.isnan(zv[i]):
            pos[i] = 0; state = 0; continue
        if state == 0:
            if zv[i] >  entry: state = -1
            elif zv[i] < -entry: state = +1
        else:
            if abs(zv[i]) < exit_:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="pos")


def perf_stats(r: pd.Series, ann: float) -> dict:
    """Core risk-return metrics. r is a series of periodic net returns."""
    if r.std() == 0 or len(r) < 2:
        return {"sharpe": np.nan, "sortino": np.nan, "cagr": np.nan,
                "max_dd": np.nan, "var95": np.nan}
    downside = r[r < 0].std(ddof=0)
    equity   = (1 + r).cumprod()
    dd       = equity / equity.cummax() - 1
    n_years  = len(r) / (ann ** 2)        # ann = sqrt(periods/yr)
    cagr     = equity.iloc[-1] ** (1 / max(n_years, 1e-9)) - 1
    return {
        "sharpe":  r.mean() / r.std(ddof=0) * ann,
        "sortino": r.mean() / downside * ann if downside > 0 else np.nan,
        "cagr":    cagr,
        "max_dd":  dd.min(),
        "var95":   np.quantile(r, 0.05),
    }


def stationary_bootstrap(r: np.ndarray, n_paths: int, block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap. Returns (n_paths, len(r))."""
    T = len(r); p = 1.0 / block
    out = np.empty((n_paths, T))
    for k in range(n_paths):
        idx = np.empty(T, dtype=np.int64)
        i = rng.integers(T)
        for t in range(T):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(T)
            else:
                i = (i + 1) % T
        out[k] = r[idx]
    return out


# ---- pipeline ----
df = pd.read_csv(Path("btc_hourly.csv"), parse_dates=["ts"]).set_index("ts").sort_index()
df["close"]   = df["close"].astype(np.float64)
df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
df = df.dropna()

# walk-forward: train params on first half, evaluate on second (held out)
split = len(df) // 2
oos   = df.iloc[split:].copy()

oos["z"]   = rolling_zscore(np.log(oos["close"]), WINDOW)
oos["pos"] = generate_positions(oos["z"], ENTRY_Z, EXIT_Z)

# pnl: next-bar execution; fee charged on position change
turnover       = oos["pos"].diff().abs().fillna(0)
oos["ret"]     = oos["pos"].shift(1) * oos["log_ret"] - turnover * (FEE_BPS / 1e4)
oos            = oos.dropna()

stats = perf_stats(oos["ret"], ANN_FACTOR)
print(f"OOS bars: {len(oos)}   trades: {int(turnover.sum()/2)}")
print(f"Sharpe {stats['sharpe']:.2f}  Sortino {stats['sortino']:.2f}  "
      f"CAGR {stats['cagr']:.2%}  MaxDD {stats['max_dd']:.2%}  "
      f"VaR95 {stats['var95']:.4f}")

# Monte Carlo: bootstrap OOS strategy returns -> distribution of terminal equity & Sharpe
rng         = np.random.default_rng(7)
boot        = stationary_bootstrap(oos["ret"].values, MC_PATHS, BLOCK_LEN, rng)
boot_eq     = np.cumprod(1 + boot, axis=1)
terminal    = boot_eq[:, -1]
boot_sharpe = boot.mean(axis=1) / boot.std(axis=1, ddof=0) * ANN_FACTOR

ci_sharpe = np.quantile(boot_sharpe, [0.025, 0.5, 0.975])
ci_term   = np.quantile(terminal, [0.025, 0.5, 0.975])
p_profit  = (terminal > 1).mean()
print(f"Sharpe 95% CI: [{ci_sharpe[0]:.2f}, {ci_sharpe[2]:.2f}]  median {ci_sharpe[1]:.2f}")
print(f"Terminal equity 95% CI: [{ci_term[0]:.3f}, {ci_term[2]:.3f}]  P(profit)={p_profit:.1%}")

# ---- plots ----
fig = plt.figure(figsize=(11, 4.4))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

ax1 = fig.add_subplot(gs[0, 0])
eq  = (1 + oos["ret"]).cumprod()
bh  = (1 + oos["log_ret"]).cumprod()
ax1.plot(eq.index, eq, label="Strategy", linewidth=1.4, color="#800000")
ax1.plot(bh.index, bh, label="Buy & hold", linewidth=1.2, color="#888", alpha=0.8)
ax1.set_title("Out-of-sample equity")
ax1.set_ylabel("Equity (×)"); ax1.grid(alpha=0.3); ax1.legend(frameon=False)

ax2 = fig.add_subplot(gs[0, 1])
sample = boot_eq[rng.choice(MC_PATHS, 120, replace=False)]
for path in sample:
    ax2.plot(path, color="#800000", alpha=0.06, linewidth=0.8)
ax2.plot(np.median(boot_eq, axis=0), color="#800000", linewidth=1.8, label="median")
ax2.plot(np.quantile(boot_eq, 0.025, axis=0), color="#444", linestyle="--", linewidth=1, label="2.5 / 97.5%")
ax2.plot(np.quantile(boot_eq, 0.975, axis=0), color="#444", linestyle="--", linewidth=1)
ax2.set_title(f"Monte Carlo equity fan (n={MC_PATHS:,})")
ax2.set_xlabel("Bars"); ax2.grid(alpha=0.3); ax2.legend(frameon=False, fontsize=8)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(boot_sharpe, bins=50, color="#800000", alpha=0.75, edgecolor="white")
ax3.axvline(stats["sharpe"], color="black", linewidth=1.5, label=f"realised {stats['sharpe']:.2f}")
ax3.axvline(ci_sharpe[0], color="#444", linestyle="--", linewidth=1)
ax3.axvline(ci_sharpe[2], color="#444", linestyle="--", linewidth=1)
ax3.set_title("Bootstrap Sharpe distribution")
ax3.set_xlabel("Sharpe"); ax3.legend(frameon=False, fontsize=8); ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mc_plot.png", dpi=170, bbox_inches="tight")
print("Saved: mc_plot.png")
