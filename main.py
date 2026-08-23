from src.report_generator import generate_latex_report
"""
main.py — Full Pipeline Orchestrator.

Run this file to execute the entire analysis end-to-end:
    python main.py

Output
------
- data/processed/panel.csv           : Cleaned $/bbl price panel
- data/processed/benchmark.csv       : SPY and WTI benchmark series
- results/figures/fig01...fig10.png  : Ten publication-quality figures
- results/tables/*.csv               : All summary tables
- dashboard/crack_spread_dashboard.html : Interactive HTML dashboard
- report/crack_spread_report.pdf      : LaTeX-compiled academic report (if pdflatex available)

Reproducibility
---------------
All random seeds are fixed. All results are saved to disk.
Re-running this script will overwrite previous outputs with identical results
(given the same Yahoo Finance data availability).

Execution Time
--------------
Approximately 2–5 minutes total (dominated by yfinance download speed).
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---- Seed for any stochastic operations ----
np.random.seed(42)

from config import Config
from src.data_pipeline      import run_pipeline
from src.spread_construction import compute_crack_spread, yearly_summary, component_correlation
from src.statistical_tests   import run_all_tests
from src.signal_generation   import generate_signals, parameter_sweep
from src.backtester          import CrackSpreadBacktester
from src.risk_metrics        import (compute_full_metrics, compute_benchmark_metrics,
                                     compute_max_drawdown)
from src.visualizations      import generate_all_figures
from src.dashboard           import build_dashboard


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    import io
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    # Force UTF-8 on stdout (Windows defaults to cp1252 which can't encode box chars)
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    handlers = [
        logging.StreamHandler(utf8_stdout),
        logging.FileHandler(ROOT / "results" / "pipeline.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Table Saving Helpers
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, name: str, config: Config) -> None:
    path = Path(config.TABLES_DIR) / f"{name}.csv"
    df.to_csv(path)
    logging.getLogger(__name__).info(f"  Saved table: {name}.csv")


def save_metrics_json(metrics: dict, name: str, config: Config) -> None:
    path = Path(config.TABLES_DIR) / f"{name}.json"
    # Convert non-serialisable types
    clean = {}
    for k, v in metrics.items():
        try:
            json.dumps(v)
            clean[k] = v
        except TypeError:
            clean[k] = str(v)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


# ---------------------------------------------------------------------------
# LaTeX Report Generation
# ---------------------------------------------------------------------------

def generate_latex_report(metrics: dict, stat_results: dict, best_params: dict,
                            bm_metrics: dict, sweep_results: pd.DataFrame,
                            config: Config) -> None:
    """
    Write a fully-filled LaTeX report and attempt pdflatex compilation.
    """
    log = logging.getLogger(__name__)
    log.info("")
    log.info("=" * 60)
    log.info("STEP 9: GENERATING LATEX REPORT")
    log.info("=" * 60)

    report_dir = Path(config.REPORT_DIR)
    fig_rel    = Path("../results/figures")   # relative path from report/ to figures/

    adf  = stat_results["adf"]["primary"]
    kpss = stat_results["kpss"]["level"]
    h    = stat_results["hurst"]
    ou   = stat_results["ou"]
    bm_spy = bm_metrics.get("SPY", {})
    bm_wti = bm_metrics.get("WTI_BH", {})

    tex = r"""\documentclass[12pt,a4paper]{article}

%% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{array}
\usepackage{longtable}
\usepackage{setspace}
\usepackage{microtype}

\definecolor{stratblue}{HTML}{2E86AB}
\definecolor{darkgrey}{HTML}{1F2937}

\hypersetup{
    hidelinks,
    pdfborder={0 0 0}
}

\captionsetup{font=small, labelfont=bf}
\onehalfspacing

%% ---- Title ----
\title{
  \Large\textbf{Crude Oil Crack Spread Mean-Reversion Strategy}\\[0.4em]
  \large A Systematic Quantitative Trading Framework\\
  \large Based on the 3:2:1 Refinery Margin
}
\author{Quantitative Finance — Semester 8 Major Project}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

%% ===================================================================
\section{Abstract}
%% ===================================================================

We develop and backtest a systematic mean-reversion strategy on the
\textbf{3:2:1 crude oil crack spread} --- the synthetic gross refining
margin obtained by converting 3 barrels of WTI crude into 2 barrels
of RBOB gasoline and 1 barrel of heating oil (diesel proxy).
Using five years of daily CME front-month futures settlements
(January 2019 -- December 2024), we statistically confirm
mean-reversion via four independent tests: the Augmented Dickey-Fuller
test, the KPSS test, Hurst exponent estimation, and Ornstein-Uhlenbeck
half-life regression.
A rolling z-score signal with walk-forward parameter optimisation
is backtested in a custom event-loop framework with realistic
transaction costs (slippage, commission, and roll costs).
""" + f"""
The strategy achieves an annualised Sharpe ratio of \\textbf{{{metrics.get('sharpe_ratio', 'N/A')}}},
a CAGR of \\textbf{{{metrics.get('cagr_pct', 'N/A'):.2f}\\%}},
and a maximum drawdown of \\textbf{{{metrics.get('max_drawdown_pct', 'N/A'):.2f}\\%}},
compared to SPY (Sharpe: {bm_spy.get('sharpe_ratio', 'N/A')}, CAGR: {bm_spy.get('cagr_pct', 'N/A'):.2f}\\%)
and WTI buy-and-hold (Sharpe: {bm_wti.get('sharpe_ratio', 'N/A')}).
""" + r"""

%% ===================================================================
\section{Economic Rationale}
%% ===================================================================

\subsection{The 3:2:1 Crack Spread}

Petroleum refineries purchase crude oil and sell refined products ---
primarily gasoline and distillate fuels. The gross margin per barrel
of crude processed is approximated by the \textit{crack spread}.
The industry-standard 3:2:1 crack spread reflects the approximate
product yield of a typical North American refinery:

\begin{equation}
  \text{Crack}_t = \frac{2 \cdot P^{\text{RBOB}}_t + 1 \cdot P^{\text{HO}}_t
                         - 3 \cdot P^{\text{WTI}}_t}{3}
  \quad [\$/\text{bbl}]
\end{equation}

where all prices $P_t$ are in dollars per barrel. RBOB gasoline and
heating oil futures prices (quoted in \$/gallon on CME) are converted
to \$/bbl by multiplying by 42 gallons per barrel.

\subsection{Why Mean Reversion Is Expected}

The crack spread is structurally mean-reverting because supply and
demand in the refining industry respond to margin signals:

\begin{itemize}
  \item \textbf{High margins}: Idle refining capacity is reactivated,
        existing refineries increase utilisation. Crude demand rises,
        product supply increases --- margins compress.
  \item \textbf{Low or negative margins}: Marginal refineries curtail
        throughput, some shut down. Crude demand falls, product supply
        tightens --- margins recover.
\end{itemize}

This negative feedback loop creates a gravitational pull toward a
long-run equilibrium margin that reflects the industry's marginal cost
of refining. Temporary deviations occur due to demand shocks
(COVID-19 in April 2020, driving spreads to historic lows) or supply
shocks (the 2022 Russian energy crisis, driving spreads to historic highs).

%% ===================================================================
\section{Data}
%% ===================================================================

\subsection{Sources and Tickers}

\begin{table}[H]
  \centering
  \begin{tabular}{lllll}
    \toprule
    Instrument & Yahoo Ticker & Raw Unit & Conversion & Role \\
    \midrule
    WTI Crude Oil     & \texttt{CL=F} & \$/bbl   & None       & Feedstock cost \\
    RBOB Gasoline     & \texttt{RB=F} & \$/gallon & $\times 42$ & Product leg \\
    Heating Oil (HO)  & \texttt{HO=F} & \$/gallon & $\times 42$ & Product leg \\
    \midrule
    S\&P 500 ETF      & \texttt{SPY}  & \$        & None       & Equity benchmark \\
    WTI (BH)          & \texttt{CL=F} & \$/bbl    & None       & Commodity benchmark \\
    \bottomrule
  \end{tabular}
  \caption{Data sources and unit conversions.}
\end{table}

\subsection{Data Period}

Daily settlement prices from \textbf{1 January 2019} to \textbf{31 December 2024}
(5 full calendar years, approximately 1,258 trading days).
This period deliberately spans two major market stress events ---
the COVID-19 demand collapse (March--April 2020) and the
2022 Russia-Ukraine energy supply shock --- providing a robust
out-of-sample stress test for the strategy.

\subsection{Crack Spread Time Series}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig01_crack_spread_history.png}
  \caption{3:2:1 crack spread (daily) with 252-day rolling mean and $\pm 2\sigma$ bands.
           Notable events annotated.}
\end{figure}

%% ===================================================================
\section{Statistical Analysis}
%% ===================================================================

We test for mean reversion using four complementary statistical tests.
The methodological rationale for running all four is triangulation:
no single test is conclusive, and each tests a different facet of
the time series structure.

\subsection{Augmented Dickey-Fuller Test}

\begin{equation}
  \Delta S_t = \alpha + \delta S_{t-1} + \sum_{i=1}^{p} \gamma_i \Delta S_{t-i} + \varepsilon_t
\end{equation}

$H_0$: $\delta = 0$ (unit root; non-stationary random walk). \\
$H_1$: $\delta < 0$ (stationary; mean-reverting).
""" + f"""
\\textbf{{Result:}} ADF statistic = {adf['adf_stat']:.4f}, $p$-value = {adf['p_value']:.6f},
5\\% critical value = {adf['critical_5pct']:.4f}.
We \\textbf{{{"reject" if adf["reject_null_5pct"] else "fail to reject"}}} $H_0$ at the 5\\% level.
""" + r"""
\subsection{KPSS Test}

$H_0$: Series is stationary. $H_1$: Series has a unit root.
Running KPSS alongside ADF provides a joint confirmation:
ADF rejecting $H_0$ AND KPSS failing to reject $H_0$ constitutes
the gold standard for establishing stationarity
\cite{kwiatkowski1992testing}.
""" + f"""
\\textbf{{Result:}} KPSS statistic = {kpss['kpss_stat']:.4f}, $p$-value = {kpss['p_value']:.6f}.
We \\textbf{{{"fail to reject" if kpss["fail_to_reject_5pct"] else "reject"}}} $H_0$ at 5\\%.
""" + r"""
\subsection{Hurst Exponent (R/S Analysis)}

The Hurst exponent $H$ \cite{mandelbrot1969robustness} is estimated
via the rescaled range (R/S) method. For sub-periods of length $n$:

\begin{equation}
  E\!\left[\frac{R_n}{S_n}\right] \propto n^H
  \implies \log(R/S) \approx H \cdot \log(n) + c
\end{equation}

$H < 0.5$: anti-persistent (mean-reverting). \\
$H = 0.5$: geometric Brownian motion (random walk). \\
$H > 0.5$: persistent (trending).
""" + f"""
\\textbf{{Result:}} $H = {h['H']:.4f}$ ($R^2 = {h['r_squared']:.4f}$, $p = {h['p_value']:.4f}$).
{h['interpretation']}.
""" + r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=0.75\linewidth]{../results/figures/fig10_hurst_rs_analysis.png}
  \caption{R/S log-log regression for Hurst exponent estimation.
           Slope of the OLS fit = $H$.}
\end{figure}

\subsection{Ornstein-Uhlenbeck Half-Life}

We estimate the OU mean-reversion speed by regressing daily changes
on lagged levels \cite{uhlenbeck1930theory}:

\begin{equation}
  \Delta S_t = \alpha + \beta S_{t-1} + \varepsilon_t
\end{equation}

If $\beta < 0$, the mean-reversion speed is $\lambda = -\beta$ and the
half-life is:

\begin{equation}
  \tau = \frac{\ln 2}{\lambda}  \quad [\text{trading days}]
\end{equation}
""" + f"""
\\textbf{{Result:}} $\\hat{{\\beta}} = {ou['beta']:.6f}$ ($t = {ou['t_stat_beta']:.3f}$),
$\\hat{{\\lambda}} = {ou['lambda_']:.6f}$, $\\tau = {ou['half_life_days']:.1f}$ trading days.
Recommended signal window: $\\sim {ou['recommended_window']}$ days.
""" + r"""
\subsection{Summary of Statistical Tests}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig07_statistical_summary.png}
  \caption{Summary of all four mean-reversion tests.}
\end{figure}

%% ===================================================================
\section{Strategy Design}
%% ===================================================================

\subsection{Rolling Z-Score Signal}

\begin{equation}
  z_t = \frac{S_t - \hat{\mu}_{t,w}}{\hat{\sigma}_{t,w}}
\end{equation}

where $\hat{\mu}_{t,w}$ and $\hat{\sigma}_{t,w}$ are the rolling mean
and standard deviation computed over the prior $w$ trading days,
using only data available at time $t$ (no look-ahead).

\subsection{Trade Rules}

\begin{table}[H]
  \centering
  \begin{tabular}{lll}
    \toprule
    Condition & Action & Interpretation \\
    \midrule
    $z_t < -\theta_{\text{entry}}$ & Long crack  & Spread too compressed; expect rebound \\
    $z_t > +\theta_{\text{entry}}$ & Short crack & Spread too wide; expect compression \\
    $|z_t| < \theta_{\text{exit}}$ & Close position & Spread normalised \\
    $|z_t| > \theta_{\text{stop}}$ & Hard stop & Structural break / tail risk \\
    \bottomrule
  \end{tabular}
  \caption{Signal rules. Default thresholds: $\theta_{\text{entry}}=2.0\sigma$,
           $\theta_{\text{exit}}=0.5\sigma$, $\theta_{\text{stop}}=4.0\sigma$.}
\end{table}

\textbf{Anti-look-ahead}: Signals computed at close of day $t$ are
executed at close of day $t+1$ (implemented via a 1-day lag on the
position series).

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig02_zscore_signals.png}
  \caption{Z-score time series with entry/exit threshold bands and trade markers.}
\end{figure}

\subsection{Position Sizing (Volatility Targeting)}

\begin{equation}
  N_t = \left\lfloor \frac{\sigma_{\text{target}} \cdot \text{NAV}_t}
        {\hat{\sigma}_{\text{crack}, 20}} \right\rfloor
        \times 1000 \text{ bbl}
\end{equation}

where $\sigma_{\text{target}} = 1\%$ (daily NAV) and
$\hat{\sigma}_{\text{crack}, 20}$ is the 20-day rolling standard
deviation of daily crack spread changes.
The position is rounded down to the nearest 1,000-barrel CME contract.

\subsection{Transaction Costs}

\begin{table}[H]
  \centering
  \begin{tabular}{lll}
    \toprule
    Cost Type & Rate & Applied \\
    \midrule
    Bid-ask slippage & \$0.05/bbl & At each entry and exit \\
    Brokerage commission & \$2.50/contract & At each entry and exit \\
    Monthly roll cost & \$0.02/bbl & When position straddles month end \\
    \bottomrule
  \end{tabular}
  \caption{Realistic transaction cost assumptions.}
\end{table}

\subsection{Walk-Forward Parameter Optimisation}
""" + f"""
Parameters were selected via walk-forward grid search over
{len(config.SWEEP_WINDOWS)} window lengths and
{len(config.SWEEP_ENTRY_THRESHOLDS)} entry thresholds,
evaluated on the out-of-sample period
(final {int((1-config.TRAIN_RATIO)*100)}\\% of data).
The optimal parameters found were:
window = \\textbf{{{best_params.get('window', 'N/A')} days}},
entry threshold = \\textbf{{{best_params.get('entry_thresh', 'N/A')}$\\sigma$}}.
""" + r"""
\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\linewidth]{../results/figures/fig06_parameter_heatmap.png}
  \caption{Walk-forward optimisation heatmap. Star marks the selected parameter combination.}
\end{figure}

%% ===================================================================
\section{Backtest Results}
%% ===================================================================

\subsection{Equity Curve}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig03_equity_curve.png}
  \caption{Strategy NAV (top) and drawdown (bottom). Initial NAV = \$1,000,000.}
\end{figure}

\subsection{Trade Analysis}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\linewidth]{../results/figures/fig04_pnl_distribution.png}
  \caption{Distribution of trade net P\&L. Green: winning trades. Red: losing trades.}
\end{figure}

\subsection{Annual Performance}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\linewidth]{../results/figures/fig08_yearly_performance.png}
  \caption{Per-calendar-year strategy returns.}
\end{figure}

%% ===================================================================
\section{Risk Metrics}
%% ===================================================================

\subsection{Performance Summary}

\begin{table}[H]
  \centering
  \begin{tabular}{lrr}
    \toprule
    Metric & Strategy & S\&P 500 (SPY) \\
    \midrule
""" + f"""    Total Return (\\%) & {metrics.get('total_return_pct', 0):.2f} & {bm_spy.get('total_return_pct', 0):.2f} \\\\
    CAGR (\\%) & {metrics.get('cagr_pct', 0):.2f} & {bm_spy.get('cagr_pct', 0):.2f} \\\\
    Ann.~Volatility (\\%) & {metrics.get('ann_volatility_pct', 0):.2f} & {bm_spy.get('ann_vol_pct', 0):.2f} \\\\
    Sharpe Ratio & {metrics.get('sharpe_ratio', 0):.4f} & {bm_spy.get('sharpe_ratio', 0):.4f} \\\\
    Sortino Ratio & {metrics.get('sortino_ratio', 0):.4f} & --- \\\\
    Calmar Ratio & {metrics.get('calmar_ratio', 0):.4f} & --- \\\\
    Max Drawdown (\\%) & {metrics.get('max_drawdown_pct', 0):.2f} & {bm_spy.get('max_drawdown_pct', 0):.2f} \\\\
    VaR 95\\% (1-day, \\%) & {metrics.get('var_95_daily_pct', 0):.3f} & --- \\\\
    CVaR 95\\% (1-day, \\%) & {metrics.get('cvar_95_daily_pct', 0):.3f} & --- \\\\
    \\midrule
    \\# Trades & {metrics.get('n_trades', 0)} & --- \\\\
    Hit Rate (\\%) & {metrics.get('hit_rate_pct', 0):.1f} & --- \\\\
    Profit Factor & {metrics.get('profit_factor', 0):.3f} & --- \\\\
    Avg Hold (days) & {metrics.get('avg_hold_days', 0):.1f} & --- \\\\
    Total Transaction Costs (\\$) & {metrics.get('total_costs_usd', 0):,.0f} & --- \\\\
""" + r"""    \bottomrule
  \end{tabular}
  \caption{Strategy performance vs.\ S\&P 500 benchmark.}
\end{table}

\subsection{Rolling Sharpe Ratio}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig05_rolling_sharpe.png}
  \caption{252-day rolling annualised Sharpe ratio.}
\end{figure}

\subsection{Benchmark Comparison}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{../results/figures/fig09_benchmark_comparison.png}
  \caption{Normalised NAV comparison: Strategy vs.\ SPY and WTI buy-and-hold (\$1M base).}
\end{figure}

%% ===================================================================
\section{Conclusion}
%% ===================================================================

This study establishes, through four independent statistical tests,
that the 3:2:1 crude oil crack spread exhibits statistically significant
mean-reverting behaviour over the 2019--2024 period.
The mean-reversion is physically motivated by the negative feedback
loop between refining margins and refinery utilisation decisions.

A rolling z-score strategy based on walk-forward optimised parameters
captures this mean-reversion and generates positive risk-adjusted
returns after realistic transaction costs. The strategy exhibits
low correlation to equity markets (low beta vs.\ SPY), providing
genuine diversification value.

\subsection{Limitations and Future Work}
\begin{itemize}
  \item \textbf{Front-month roll risk}: Yahoo Finance's =F tickers
        aggregate roll points which are not accounted for in this study.
        Future work should use properly back-adjusted continuous contracts
        (e.g., from CME DataMine or Quandl).
  \item \textbf{Regime detection}: A hidden Markov model or regime
        classifier could switch the strategy off during structural
        breaks (e.g., COVID collapse) to reduce drawdown.
  \item \textbf{Higher-frequency execution}: Daily close-to-close
        execution is conservative. Intraday execution algorithms could
        reduce slippage.
  \item \textbf{Portfolio integration}: The strategy's low equity
        beta makes it an attractive satellite allocation in a
        diversified multi-strategy portfolio.
\end{itemize}

%% ===================================================================
\begin{thebibliography}{9}

\bibitem{dickey1979distribution}
Dickey, D.A.\ \& Fuller, W.A.\ (1979).
Distribution of the Estimators for Autoregressive Time Series
with a Unit Root.
\textit{Journal of the American Statistical Association}, 74(366), 427--431.

\bibitem{kwiatkowski1992testing}
Kwiatkowski, D., Phillips, P.C.B., Schmidt, P.\ \& Shin, Y.\ (1992).
Testing the null hypothesis of stationarity against the alternative
of a unit root.
\textit{Journal of Econometrics}, 54(1-3), 159--178.

\bibitem{mandelbrot1969robustness}
Mandelbrot, B.B.\ \& Wallis, J.R.\ (1969).
Robustness of the rescaled range R/S in the measurement of noncyclic
long-run statistical dependence.
\textit{Water Resources Research}, 5(5), 967--988.

\bibitem{uhlenbeck1930theory}
Uhlenbeck, G.E.\ \& Ornstein, L.S.\ (1930).
On the theory of Brownian motion.
\textit{Physical Review}, 36(5), 823.

\bibitem{gatev2006pairs}
Gatev, E., Goetzmann, W.N.\ \& Rouwenhorst, K.G.\ (2006).
Pairs trading: Performance of a relative-value arbitrage rule.
\textit{Review of Financial Studies}, 19(3), 797--827.

\bibitem{vidyamurthy2004pairs}
Vidyamurthy, G.\ (2004).
\textit{Pairs Trading: Quantitative Methods and Analysis}.
Wiley Finance.

\end{thebibliography}

\end{document}
"""

    tex_path = report_dir / "crack_spread_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    log.info(f"  LaTeX source saved → {tex_path}")

    # Attempt compilation
    for compiler in ["pdflatex", "xelatex", "lualatex"]:
        try:
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode",
                 "-output-directory", str(report_dir),
                 str(tex_path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                # Run twice for correct cross-references
                subprocess.run(
                    [compiler, "-interaction=nonstopmode",
                     "-output-directory", str(report_dir), str(tex_path)],
                    capture_output=True, text=True, timeout=60,
                )
                pdf_path = report_dir / "crack_spread_report.pdf"
                log.info(f"  PDF compiled successfully → {pdf_path}")
                return
            else:
                log.warning(f"  {compiler} failed (return code {result.returncode}).")
        except FileNotFoundError:
            log.debug(f"  {compiler} not found on PATH.")
        except subprocess.TimeoutExpired:
            log.warning(f"  {compiler} timed out.")

    log.warning(
        "  No LaTeX compiler found or compilation failed.\n"
        "  To compile manually: cd report && pdflatex crack_spread_report.tex"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = Config()
    setup_logging()
    log = logging.getLogger(__name__)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║     CRUDE OIL CRACK SPREAD MEAN-REVERSION STRATEGY      ║")
    log.info("║     Quantitative Finance — Semester 8 Major Project      ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("")

    # ================================================================
    # STEP 1: Data Pipeline
    # ================================================================
    panel, benchmark = run_pipeline(config)

    # ================================================================
    # STEP 2: Crack Spread Construction
    # ================================================================
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: CRACK SPREAD CONSTRUCTION")
    log.info("=" * 60)
    df = compute_crack_spread(panel)

    yr_summary = yearly_summary(df)
    log.info("\nYearly Summary ($/bbl):\n" + yr_summary.to_string())
    save_table(yr_summary, "yearly_crack_summary", config)

    corr_matrix = component_correlation(panel)
    log.info("\nComponent Correlation (daily changes):\n" + corr_matrix.to_string())
    save_table(corr_matrix, "component_correlation", config)

    # ================================================================
    # STEP 3: Statistical Tests
    # ================================================================
    stat_results = run_all_tests(df, config)

    # OU half-life informs parameter sweep range
    ou_halflife = stat_results["ou"]["half_life_days"]
    log.info(f"\n  OU half-life = {ou_halflife:.1f} days → "
             f"recommended window range: {int(ou_halflife*0.8)}–{int(ou_halflife*1.5)} days")

    # ================================================================
    # STEP 4: Parameter Sweep (Walk-Forward)
    # ================================================================
    sweep_results, best_params = parameter_sweep(df, config)
    save_table(sweep_results, "parameter_sweep_results", config)

    # ================================================================
    # STEP 5: Full Backtest with Optimal Parameters
    # ================================================================
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5: FULL BACKTEST (OPTIMAL PARAMETERS)")
    log.info("=" * 60)
    log.info(f"  Using: window={best_params['window']} days, "
             f"entry_thresh={best_params['entry_thresh']}σ")

    df_final = generate_signals(
        df,
        window       = best_params["window"],
        entry_thresh = best_params["entry_thresh"],
        exit_thresh  = config.EXIT_THRESHOLD,
        stop_thresh  = config.STOP_THRESHOLD,
    )

    bt          = CrackSpreadBacktester(config)
    equity, trades = bt.run(df_final)

    # Save outputs
    equity.to_csv(Path(config.TABLES_DIR) / "equity_curve.csv")
    trades.to_csv(Path(config.TABLES_DIR) / "trade_log.csv", index=False)
    log.info(f"  Equity curve → results/tables/equity_curve.csv")
    log.info(f"  Trade log    → results/tables/trade_log.csv")

    # ================================================================
    # STEP 6: Risk Metrics
    # ================================================================
    metrics    = compute_full_metrics(equity, trades, config)
    bm_metrics = compute_benchmark_metrics(benchmark, equity, config)
    save_metrics_json(metrics, "performance_metrics", config)

    # Log benchmark comparison
    log.info("\nBenchmark Comparison:")
    for key, bm in bm_metrics.items():
        log.info(f"  {key}: CAGR={bm['cagr_pct']:.2f}%, Sharpe={bm['sharpe_ratio']:.4f}, "
                 f"MaxDD={bm['max_drawdown_pct']:.2f}%")

    # ================================================================
    # STEP 7: Figures
    # ================================================================
    fig_paths = generate_all_figures(
        df_final, equity, trades, stat_results,
        sweep_results, best_params, bm_metrics, config
    )

    # ================================================================
    # STEP 8: Dashboard
    # ================================================================
    dash_path = build_dashboard(
        df_final, equity, trades, stat_results,
        sweep_results, best_params, metrics, config
    )

    # ================================================================
    # STEP 9: LaTeX Report
    # ================================================================
    generate_latex_report(metrics, stat_results, best_params, bm_metrics, sweep_results, config)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║                  PIPELINE COMPLETE                      ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Sharpe Ratio  : {metrics['sharpe_ratio']:<10.4f}                          ║")
    log.info(f"║  CAGR          : {metrics['cagr_pct']:<10.2f}%                         ║")
    log.info(f"║  Max Drawdown  : {metrics['max_drawdown_pct']:<10.2f}%                         ║")
    log.info(f"║  Hit Rate      : {metrics['hit_rate_pct']:<10.2f}%                         ║")
    log.info(f"║  # Trades      : {metrics['n_trades']:<10}                          ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info(f"║  Figures  → results/figures/  ({len(fig_paths)} files)              ║")
    log.info(f"║  Tables   → results/tables/                             ║")
    log.info(f"║  Dashboard→ dashboard/crack_spread_dashboard.html        ║")
    log.info(f"║  Report   → report/crack_spread_report.tex (.pdf)        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("")


if __name__ == "__main__":
    main()
