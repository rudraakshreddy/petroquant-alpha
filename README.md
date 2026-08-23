# Crude Oil Crack Spread Mean-Reversion Strategy



---

## Overview

A production-quality systematic trading strategy exploiting **mean reversion in the 3:2:1 crude oil crack spread** â€” the synthetic gross refining margin from converting 3 barrels of WTI crude into 2 barrels of RBOB gasoline and 1 barrel of heating oil.

$$\text{Crack}_t = \frac{2 \cdot \text{RBOB}_t + \text{HO}_t - 3 \cdot \text{WTI}_t}{3} \quad [\$/\text{bbl}]$$

### Why This Works (Physical Rationale)

High crack spreads â†’ refineries increase utilisation â†’ more crude demand + more product supply â†’ margins compress back down. This negative feedback loop creates a gravitational pull toward equilibrium that the strategy systematically exploits.

---

## Project Structure

```
crack_spread_strategy/
â”‚
â”œâ”€â”€ main.py                        â† Run this to execute everything
â”œâ”€â”€ config.py                      â† All parameters in one place
â”œâ”€â”€ requirements.txt
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_pipeline.py           â† Download, validate, unit-convert
â”‚   â”œâ”€â”€ spread_construction.py     â† 3:2:1 formula + rolling stats
â”‚   â”œâ”€â”€ statistical_tests.py       â† ADF, KPSS, Hurst, OU half-life
â”‚   â”œâ”€â”€ signal_generation.py       â† Rolling z-score + walk-forward sweep
â”‚   â”œâ”€â”€ backtester.py              â† Custom event-loop backtester
â”‚   â”œâ”€â”€ risk_metrics.py            â† Full risk/performance suite
â”‚   â”œâ”€â”€ visualizations.py          â† 10 publication-quality figures
â”‚   â””â”€â”€ dashboard.py               â† Plotly HTML interactive dashboard
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                       â† Downloaded with timestamps
â”‚   â””â”€â”€ processed/                 â† Cleaned $/bbl panel
â”‚
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ figures/                   â† fig01...fig10 at 300 DPI
â”‚   â””â”€â”€ tables/                    â† CSV + JSON outputs
â”‚
â”œâ”€â”€ report/
â”‚   â”œâ”€â”€ crack_spread_report.tex    â† LaTeX source (auto-generated with results)
â”‚   â””â”€â”€ crack_spread_report.pdf    â† Compiled PDF (if pdflatex available)
â”‚
â””â”€â”€ dashboard/
    â””â”€â”€ crack_spread_dashboard.html  â† Interactive 6-tab dashboard
```

---

## Quick Start

```bash
# 1. Navigate to project directory
cd crack_spread_strategy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

Total runtime: **~2â€“5 minutes** (dominated by yfinance download speed).

---

## What Gets Generated

| Output | Description |
|--------|-------------|
| `results/figures/fig01_crack_spread_history.png` | Crack spread + Â±2Ïƒ rolling band |
| `results/figures/fig02_zscore_signals.png` | Z-score with trade markers |
| `results/figures/fig03_equity_curve.png` | NAV + drawdown (dual panel) |
| `results/figures/fig04_pnl_distribution.png` | Trade P&L histogram + KDE |
| `results/figures/fig05_rolling_sharpe.png` | 252-day rolling Sharpe |
| `results/figures/fig06_parameter_heatmap.png` | Walk-forward grid heatmap |
| `results/figures/fig07_statistical_summary.png` | Test results table figure |
| `results/figures/fig08_yearly_performance.png` | Annual returns bar chart |
| `results/figures/fig09_benchmark_comparison.png` | vs. SPY + WTI buy-and-hold |
| `results/figures/fig10_hurst_rs_analysis.png` | R/S log-log regression |
| `results/tables/equity_curve.csv` | Daily NAV time series |
| `results/tables/trade_log.csv` | Every trade with P&L breakdown |
| `results/tables/performance_metrics.json` | All metrics as JSON |
| `results/tables/parameter_sweep_results.csv` | Full sweep grid |
| `dashboard/crack_spread_dashboard.html` | Interactive 6-tab Plotly dashboard |
| `report/crack_spread_report.tex` | Auto-filled LaTeX report |
| `report/crack_spread_report.pdf` | Compiled PDF (if pdflatex found) |

---

## Statistical Framework

| Test | Null Hypothesis | Rejects at | Interpretation |
|------|----------------|-----------|----------------|
| **ADF** | Unit root present | p < 0.05 | Spread is stationary (mean-reverts) |
| **KPSS** | Series is stationary | Fail to reject | Confirms stationarity |
| **Hurst (R/S)** | H = 0.5 (random walk) | H < 0.5 | Anti-persistent (mean-reverting) |
| **OU Half-Life** | Î² â‰¥ 0 | Î² < 0 | Finite mean-reversion speed |

---

## Strategy Parameters (defaults in `config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROLLING_WINDOW` | 40 days | Z-score lookback (overridden by sweep) |
| `ENTRY_THRESHOLD` | 2.0Ïƒ | Enter position |
| `EXIT_THRESHOLD` | 0.5Ïƒ | Exit position |
| `STOP_THRESHOLD` | 4.0Ïƒ | Hard stop (tail-risk) |
| `INITIAL_NAV` | $1,000,000 | Starting capital |
| `VOL_TARGET_DAILY` | 1% | Daily NAV risk target |
| `SLIPPAGE_PER_BBL` | $0.05/bbl | Bid-ask slippage |
| `COMMISSION_PER_CONTRACT` | $2.50 | Brokerage commission |
| `ROLL_COST_PER_BBL` | $0.02/bbl | Monthly roll cost |

---

## Key Design Decisions

### Anti-Look-Ahead Bias
Signal generated at close of day **t** is executed at close of day **t+1** (implemented via `pd.Series.shift(1)` on the position series).

### Walk-Forward Optimisation
Parameters selected on **30% out-of-sample data only** â€” not in-sample. This prevents overfitting and is how live trading desks select strategy parameters.

### Volatility Targeting
Position size scales inversely with current market volatility (20-day rolling Ïƒ of crack changes). During COVID March 2020 or 2022 energy shock, position sizes automatically shrink.

### Realistic Costs
Three cost components: slippage (per barrel) + commission (per contract) + roll cost (per month boundary crossed while in position).

---

## Compile the LaTeX Report Manually

If `pdflatex` is on your PATH, the report compiles automatically. Otherwise:

```bash
cd report
pdflatex crack_spread_report.tex
pdflatex crack_spread_report.tex   # Second pass for cross-references
```

---

## Data Sources

- **Yahoo Finance** via `yfinance`: `CL=F` (WTI), `RB=F` (RBOB), `HO=F` (Heating Oil), `SPY`
- All tickers are CME front-month continuous contracts
- Period: 2019-01-01 to 2024-12-31 (~1,258 trading days)

---

## References

1. Dickey & Fuller (1979). *Distribution of Estimators for Autoregressive Time Series with a Unit Root.* JASA.
2. Kwiatkowski et al. (1992). *Testing the null hypothesis of stationarity.* Journal of Econometrics.
3. Mandelbrot & Wallis (1969). *Robustness of the R/S measure.* Water Resources Research.
4. Uhlenbeck & Ornstein (1930). *On the theory of Brownian motion.* Physical Review.
5. Gatev, Goetzmann & Rouwenhorst (2006). *Pairs Trading.* Review of Financial Studies.
