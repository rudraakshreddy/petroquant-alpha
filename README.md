# PetroQuant Alpha

**A seasonally adjusted mean-reversion system for the 3:2:1 crude oil crack spread.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-live-FF4B4B.svg)](https://petroquant-alpha.streamlit.app)

**[Live dashboard](https://petroquant-alpha.streamlit.app)** · **[Full technical paper (PDF)](report/petroquant_report.pdf)**

---

## What this is

The 3:2:1 crack spread is the synthetic gross refining margin from converting three barrels of crude oil
into two barrels of gasoline and one of distillate:

```
S_t = (2·RBOB_t + HO_t − 3·WTI_t) / 3        [$ per barrel]
```

Economic theory predicts it mean-reverts: high margins raise refinery utilisation, which simultaneously
increases crude demand and product supply, compressing the margin back down.

This repository contains a complete study of that proposition over **2 January 2019 – 30 December 2024**
(1,510 trading days), a window deliberately chosen to contain two structural dislocations — the 2020 demand
collapse and the 2022 refining shock — together with a trading system built from the statistical
characterisation and validated under rolling-origin walk-forward analysis.

## Headline results

| Metric | This system | S&P 500 ETF | WTI buy-and-hold |
|---|---:|---:|---:|
| Total return (6y) | **75.75 %** | — | — |
| CAGR | 9.87 % | 17.17 % | 7.30 % |
| Annualised volatility | 10.04 % | — | — |
| Sharpe ratio | 0.490 | 0.647 | −0.326 |
| Maximum drawdown | **−12.72 %** | −33.72 % | −156.76 % |
| **Calmar ratio** | **0.78** | 0.51 | 0.05 |
| Hit rate / profit factor | 82.1 % / 6.69 | — | — |

Under rolling-origin walk-forward validation (parameters re-selected on training data only):

| Configuration | Stitched OOS Sharpe | OOS return | Max DD | Profitable folds |
|---|---:|---:|---:|---:|
| z-score on raw spread | −0.019 | +5.0 % | −41.9 % | 3/5 |
| + stop semantics | 0.417 | +25.5 % | −17.8 % | 3/5 |
| **+ deseasonalised signal** | **0.661** | **+36.4 %** | **−11.4 %** | **5/5** |

All five folds independently selected the same configuration (`window=30`, `entry=2.0σ`).

> **Reported honestly:** a stationary-bootstrap 95 % interval for the out-of-sample Sharpe is
> `[−0.160, 1.560]` and does **not** exclude zero. The relative improvement and the drawdown reduction are
> robust; the absolute Sharpe level is not resolvable on 625 out-of-sample days.

## The three design decisions that matter

**1. The statistical diagnostics disagree — and that disagreement is the design input.**

| Test | Result | Verdict |
|---|---|---|
| Augmented Dickey–Fuller | stat −2.28, p = 0.178 | cannot reject unit root |
| KPSS (level / trend) | 2.04 / 0.63, p < 0.01 | rejects stationarity |
| Hurst exponent (R/S) | H ≈ 0.97 | persistent, not reverting |
| Ornstein–Uhlenbeck fit | β = −0.0246, t = −4.34 | **reverting**, half-life 28.2 d |

One of four supports the trading premise. The reconciliation: the spread reverts **locally** toward an
equilibrium that itself **migrates**. A rolling estimate gives a finite half-life (median 24.6 days) over
~80 % of the sample, undefined only across the regime transitions.

**2. The signal is deseasonalised; the P&L is not.**

Refining margins have a documented annual cycle (driving season, heating demand, refinery turnarounds).
A first-order harmonic component of amplitude **\$4.55/bbl** against a total standard deviation of
\$10.99/bbl is predictable — and a z-score on the raw level trades against it as though it were
disequilibrium. Harmonic coefficients are fitted on an **expanding window using only prior data**, refitted
annually. Crucially, the adjustment applies to the *signal only*; P&L is marked on the actual traded spread.

**3. The stop-loss suppresses re-entry.**

Because `θ_stop > θ_entry` by construction, any bar on which the stop fires also satisfies the entry
condition. A stop implemented as a bare flattening would close and immediately reopen the same position on
the same bar, leaving exposure unchanged. The state machine therefore latches: entries are suppressed until
`|z| < θ_entry`.

## Repository layout

```
├── main.py                     Run the full pipeline end to end
├── config.py                   Every tunable parameter, one place
├── streamlit_app.py            Interactive dashboard
│
├── src/
│   ├── data_pipeline.py        Download, validate, $/gal → $/bbl conversion
│   ├── spread_construction.py  3:2:1 formula and rolling statistics
│   ├── statistical_tests.py    ADF, KPSS, Hurst R/S, OU half-life
│   ├── signal_generation.py    Rolling z-score, state machine, walk-forward sweep
│   ├── backtester.py           Event-loop backtester with explicit cost accounting
│   ├── risk_metrics.py         Sharpe, Sortino, Calmar, VaR, CVaR, drawdown
│   ├── visualizations.py       Publication figures
│   └── report_generator.py     LaTeX report generation
│
├── data/processed/             Aligned price panel and benchmarks
├── results/
│   ├── figures/                300-DPI figures
│   └── tables/                 Metrics, trade log, equity curve, parameter sweep
└── report/                     Technical paper (LaTeX source + PDF)
```

## Quick start

```bash
git clone https://github.com/rudraakshreddy/petroquant-alpha.git
cd petroquant-alpha
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py            # full pipeline: data → tests → sweep → backtest → figures → report
streamlit run streamlit_app.py
```

Every parameter lives in `config.py`; changing a value there propagates through the whole pipeline without
touching module code.

## Methodology in brief

**Data.** Front-month CME futures: WTI (`CL=F`), RBOB gasoline (`RB=F`), NY Harbor ULSD (`HO=F`). Gasoline
and distillate are quoted in \$/gallon and are converted at 42 gal/bbl before the spread formula is applied
— omitting this produces a numerically plausible but economically meaningless series. Calendars are aligned
on their intersection (99.9 % of dates common).

**Signal.** Rolling z-score of the deseasonalised spread. The lookback grid `{20,30,40,50,60}` brackets the
22–42 day range implied a priori by the fitted 28.2-day OU half-life.

**Execution.** A signal from the close of day *t* is executed at the close of day *t+1* (one-period lag on
the position series), which removes the most common source of look-ahead bias.

**Sizing.** Volatility targeting at 1 % daily NAV risk against the trailing 20-day standard deviation of
spread changes, rounded down to whole 1,000-barrel contracts.

**Costs.** \$0.05/bbl slippage + \$2.50/contract commission on each of entry and exit, plus \$0.02/bbl roll
cost per month boundary crossed. Total \$44,520 over the full backtest — 4.5 % of initial capital.

**Validation.** Rolling-origin walk-forward from a 756-day minimum training history. At each origin the
grid is evaluated on the training window only, the best configuration is applied to the next unseen block,
and the origin advances. Out-of-sample returns from all folds are concatenated into one continuous series.
Confidence intervals use the stationary bootstrap (Politis & Romano, 1994).

## Component contribution

Elements tested and **rejected** are reported alongside those adopted, so the reader can judge both:

| Component | Full-sample Sharpe | Max DD | Adopted |
|---|---:|---:|:--:|
| z-score on raw spread | 0.188 | −24.5 % | baseline |
| + stop re-entry suppression | 0.386 | −19.3 % | ✔ |
| + time stop (2 × half-life) | 0.386 | −19.3 % | ✔ |
| + volatility floor | 0.358 | −19.5 % | ✘ |
| + gross leverage cap | 0.336 | −19.0 % | ✘ |
| + asymmetric short threshold | 0.069 | −21.8 % | ✘ |
| + half-life regime gate | −0.006 | −24.4 % | ✘ |
| **+ deseasonalisation (K=1)** | **0.490** | **−12.7 %** | ✔ |

Four of six additions do not help. The two that do are the two derived from the fitted process — the
half-life and the annual cycle.

## Limitations

- **Sample length.** Six years, 1,510 observations. Bootstrap intervals do not exclude zero.
- **Search intensity.** 15 parameter combinations per fold, 8 configurations compared. No deflated Sharpe
  ratio or reality-check adjustment applied, so reported statistics are upper bounds.
- **Continuous-contract approximation.** Front-month series with a flat per-crossing roll charge; real
  calendar-spread roll costs and basis effects are not modelled.
- **Cost model.** Constant \$0.05/bbl slippage. Liquidity deteriorates during exactly the dislocations that
  dominate this sample, so realised costs would likely be higher.
- **Execution realism.** Fills assumed at settlement with no market impact; positions reach 38,000 barrels.
- **Single instrument.** One spread, one exchange, one period. No claim of generality.

This is a research result, not a deployable trading product.

## Citation

```bibtex
@techreport{reddy2026petroquant,
  title  = {A Seasonally Adjusted Mean-Reversion System for the 3:2:1 Crude Oil Crack Spread},
  author = {Yeddula Rudraaksh Reddy and S. N. Chakri},
  year   = {2026},
  type   = {Technical Report},
  url    = {https://github.com/rudraakshreddy/petroquant-alpha}
}
```

## Authors

- **Yeddula Rudraaksh Reddy** — primary and corresponding author ·
  [yeddularudraaksh@gmail.com](mailto:yeddularudraaksh@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/rudraakshreddy) · [GitHub](https://github.com/rudraakshreddy)
- **S. N. Chakri** — [snchakrim@gmail.com](mailto:snchakrim@gmail.com) ·
  [LinkedIn](https://www.linkedin.com/in/snchakri) · [GitHub](https://github.com/snchakri) ·
  [snchakri.com](https://snchakri.com)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Market data are obtained from publicly available end-of-day futures settlement series and are not
redistributed by this repository.
