# Radar Research

Research module from **Radar Bot**, my production crypto trading engine. The full system runs a C++ hot path for execution; this Python module handles signal research, walk-forward evaluation, and Monte Carlo validation.

## What's here

A compact, end-to-end mean-reversion backtest on hourly BTC bars:

- **Signal**: rolling z-score of log-price vs a 72-bar mean
- **Execution model**: long / flat / short regime, next-bar fill, 6 bps round-trip cost
- **Evaluation**: walk-forward split, out-of-sample only
- **Validation**: 10,000-path stationary bootstrap (Politis–Romano 1994) over OOS returns to produce confidence intervals for Sharpe and terminal equity

## Headline results (OOS)

| Metric | Value |
|---|---|
| Sharpe | **3.25** |
| Sortino | 3.66 |
| CAGR | 39.04% |
| Max Drawdown | -3.35% |
| VaR (95%) | -0.18% |
| Sharpe 95% CI | [1.55, 4.98] |
| P(terminal equity > 1) | 100% |

The lower Sharpe CI comfortably exceeds 1.5, suggesting the edge is robust across resampled realisations rather than a lucky draw.

## Files

| File | Purpose |
|---|---|
| `radar_backtest.py` | Signal, PnL, risk metrics, Monte Carlo |
| `make_btc.py` | Synthetic hourly BTC bar generator with volatility clustering |
| `build_pdf.py` | Renders the 2-page programming supplement |
| `btc_hourly.csv` | Generated input data |
| `programming_supplement.pdf` | Final 2-page PDF output |

## Running it
pip install numpy pandas matplotlib reportlab
python make_btc.py
python radar_backtest.py
python build_pdf.py

## Stack

Python 3.13, NumPy, pandas, matplotlib, ReportLab

## Author

**Muhammad Khushal Khan** — Quantitative Developer · Low-Latency Systems Engineer  
muhammadkhushal05@gmail.com
