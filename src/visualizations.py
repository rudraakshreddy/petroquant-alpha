"""
src/visualizations.py — Publication-Quality Matplotlib Figures.

Generates 10 figures saved at 300 DPI in results/figures/.
Each figure function is self-contained; pass the relevant data and config.

Figure Inventory
----------------
fig01_crack_spread_history   : Crack spread + rolling mean ± 2σ bands
fig02_zscore_signals         : Z-score with entry/exit markers
fig03_equity_curve           : NAV equity curve with drawdown shaded
fig04_pnl_distribution       : Trade P&L histogram + KDE
fig05_rolling_sharpe         : 252-day rolling Sharpe ratio
fig06_parameter_heatmap      : Window × entry_thresh → OOS Sharpe
fig07_statistical_summary    : Test results table (rendered as figure)
fig08_yearly_performance     : Per-year returns bar chart
fig09_benchmark_comparison   : Strategy vs. SPY and WTI buy-and-hold
fig10_hurst_rs_analysis      : R/S log-log regression (Hurst estimation)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)

# ---- Style constants ----
BLUE   = "#2E86AB"
RED    = "#E84855"
GREEN  = "#3BB273"
ORANGE = "#F18F01"
GREY   = "#6B7280"
LGREY  = "#D1D5DB"
DARK   = "#1F2937"
BG     = "#F9FAFB"


def _save(fig: plt.Figure, name: str, config: Config) -> str:
    """Save figure at 300 DPI and return absolute path."""
    path = Path(config.FIGURES_DIR) / name
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")
    return str(path)


def _base_style(fig, axes=None):
    """Apply consistent publication style to a figure."""
    fig.patch.set_facecolor(BG)
    if axes is not None:
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(BG)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(LGREY)
            ax.tick_params(colors=GREY, labelsize=9)
            ax.xaxis.label.set_color(DARK)
            ax.yaxis.label.set_color(DARK)
            ax.title.set_color(DARK)


# ---------------------------------------------------------------------------
# Figure 1: Crack Spread History
# ---------------------------------------------------------------------------

def fig01_crack_spread_history(df: pd.DataFrame, config: Config) -> str:
    """
    Full time series of the 3:2:1 crack spread overlaid with
    252-day rolling mean ± 2σ bands.
    Annotates COVID crash (Apr 2020) and 2022 energy shock peaks.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    _base_style(fig, ax)

    crack = df["crack"]
    ax.fill_between(df.index, df["roll_lower_252"], df["roll_upper_252"],
                    alpha=0.15, color=BLUE, label="Rolling mean ± 2σ (252-day)")
    ax.plot(df.index, df["roll_mean_252"], color=BLUE, linewidth=1.4,
            linestyle="--", label="Rolling mean (252-day)")
    ax.plot(df.index, crack, color=DARK, linewidth=0.9, alpha=0.85, label="3:2:1 Crack Spread")

    # Annotate notable events
    covid_trough = crack.loc["2020-03":"2020-05"].idxmin()
    energy_peak  = crack.loc["2022-05":"2022-07"].idxmax()
    ax.annotate(
        f"COVID demand collapse\n${crack.loc[covid_trough]:.1f}/bbl",
        xy=(covid_trough, crack.loc[covid_trough]),
        xytext=(covid_trough + pd.Timedelta(days=40), crack.loc[covid_trough] + 5),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
        fontsize=8, color=RED,
    )
    ax.annotate(
        f"2022 Energy Shock\n${crack.loc[energy_peak]:.1f}/bbl",
        xy=(energy_peak, crack.loc[energy_peak]),
        xytext=(energy_peak - pd.Timedelta(days=200), crack.loc[energy_peak] - 10),
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
        fontsize=8, color=ORANGE,
    )

    ax.axhline(crack.mean(), color=RED, linewidth=0.8, linestyle=":", alpha=0.7,
               label=f"Full-period mean (${crack.mean():.2f}/bbl)")
    ax.set_title("3:2:1 Crude Oil Crack Spread — Daily Settlement (2019–2024)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Crack Spread ($/bbl)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    fig.tight_layout()
    return _save(fig, "fig01_crack_spread_history.png", config)


# ---------------------------------------------------------------------------
# Figure 2: Z-Score and Trade Signals
# ---------------------------------------------------------------------------

def fig02_zscore_signals(df: pd.DataFrame, config: Config) -> str:
    """
    Z-score time series with horizontal entry/exit threshold lines
    and trade entry/exit markers overlaid.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.8]})
    _base_style(fig, axes)
    ax_crack, ax_z = axes

    # Top panel: crack spread
    ax_crack.plot(df.index, df["crack"], color=DARK, linewidth=0.8, alpha=0.85)
    ax_crack.set_ylabel("Crack Spread ($/bbl)", fontsize=9)
    ax_crack.set_title("Rolling Z-Score Signal with Trade Entry/Exit Markers",
                        fontsize=12, fontweight="bold", pad=10)
    ax_crack.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

    # Bottom panel: z-score
    z = df["z_score"]
    ax_z.plot(df.index, z, color=BLUE, linewidth=0.7, alpha=0.85, label="Z-Score")
    ax_z.fill_between(df.index, z, 0, where=(z < 0), alpha=0.12, color=GREEN)
    ax_z.fill_between(df.index, z, 0, where=(z > 0), alpha=0.12, color=RED)

    # Threshold lines
    for thresh, col, ls in [(config.ENTRY_THRESHOLD, RED, "--"),
                             (-config.ENTRY_THRESHOLD, GREEN, "--"),
                             (config.EXIT_THRESHOLD, GREY, ":"),
                             (-config.EXIT_THRESHOLD, GREY, ":"),
                             (config.STOP_THRESHOLD, DARK, "-."),
                             (-config.STOP_THRESHOLD, DARK, "-.")]:
        ax_z.axhline(thresh, color=col, linewidth=0.9, linestyle=ls, alpha=0.7)

    # Entry markers: transitions in position_exec
    if "position_exec" in df.columns:
        pos = df["position_exec"]
        entries = pos[(pos != pos.shift(1)) & (pos != 0)]
        exits   = pos[(pos.shift(1) != 0) & (pos == 0)]
        long_entries  = entries[entries == +1]
        short_entries = entries[entries == -1]

        ax_crack.scatter(long_entries.index,  df.loc[long_entries.index,  "crack"],
                         marker="^", color=GREEN, s=50, zorder=5, label="Long entry")
        ax_crack.scatter(short_entries.index, df.loc[short_entries.index, "crack"],
                         marker="v", color=RED,   s=50, zorder=5, label="Short entry")
        ax_crack.scatter(exits.index, df.loc[exits.index, "crack"],
                         marker="x", color=GREY, s=30, zorder=5, label="Exit")
        ax_crack.legend(fontsize=7, framealpha=0.7, loc="upper right")

    ax_z.set_ylabel("Z-Score", fontsize=9)
    ax_z.set_xlabel("Date", fontsize=9)
    ax_z.set_ylim(-6, 6)

    # Legend for thresholds
    handles = [
        mpatches.Patch(color=RED,   label=f"Entry ±{config.ENTRY_THRESHOLD}σ"),
        mpatches.Patch(color=GREY,  label=f"Exit ±{config.EXIT_THRESHOLD}σ"),
        mpatches.Patch(color=DARK,  label=f"Stop ±{config.STOP_THRESHOLD}σ"),
    ]
    ax_z.legend(handles=handles, fontsize=7, framealpha=0.7, loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig02_zscore_signals.png", config)


# ---------------------------------------------------------------------------
# Figure 3: Equity Curve + Drawdown
# ---------------------------------------------------------------------------

def fig03_equity_curve(equity: pd.DataFrame, dd_series: pd.Series,
                        config: Config) -> str:
    """
    Dual-panel: NAV equity curve (top) + drawdown (bottom, shaded red).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2.5, 1]})
    _base_style(fig, axes)
    ax_nav, ax_dd = axes

    nav = equity["nav"]
    ax_nav.plot(nav.index, nav / 1e6, color=BLUE, linewidth=1.5, label="Strategy NAV")
    ax_nav.axhline(config.INITIAL_NAV / 1e6, color=GREY, linewidth=0.8,
                   linestyle=":", alpha=0.6, label="Initial NAV ($1M)")
    ax_nav.set_ylabel("Net Asset Value ($M)", fontsize=10)
    ax_nav.set_title("Strategy Equity Curve and Drawdown Profile", fontsize=12,
                      fontweight="bold", pad=10)
    ax_nav.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}M"))
    ax_nav.legend(fontsize=9, framealpha=0.7)

    # Shaded drawdown
    ax_dd.fill_between(dd_series.index, dd_series * 100, 0,
                       color=RED, alpha=0.5, label="Drawdown")
    ax_dd.plot(dd_series.index, dd_series * 100, color=RED, linewidth=0.6)
    ax_dd.axhline(0, color=GREY, linewidth=0.5)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=9)
    ax_dd.set_xlabel("Date", fontsize=9)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_dd.invert_yaxis()
    ax_dd.legend(fontsize=8, framealpha=0.7, loc="lower right")

    fig.tight_layout()
    return _save(fig, "fig03_equity_curve.png", config)


# ---------------------------------------------------------------------------
# Figure 4: Trade P&L Distribution
# ---------------------------------------------------------------------------

def fig04_pnl_distribution(trades: pd.DataFrame, config: Config) -> str:
    """
    Histogram of trade net P&L with KDE overlay.
    Separates winning (green) and losing (red) trades.
    """
    if len(trades) == 0:
        logger.warning("No trades to plot in fig04.")
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    _base_style(fig, ax)

    pnl = trades["net_pnl"].dropna()
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    bins = np.linspace(pnl.min() * 1.1, pnl.max() * 1.1, 40)
    ax.hist(wins,   bins=bins, color=GREEN, alpha=0.65, label="Winning trades")
    ax.hist(losses, bins=bins, color=RED,   alpha=0.65, label="Losing trades")

    # KDE overlay
    from scipy.stats import gaussian_kde
    if len(pnl) > 5:
        kde    = gaussian_kde(pnl)
        x_kde  = np.linspace(pnl.min() * 1.2, pnl.max() * 1.2, 300)
        y_kde  = kde(x_kde)
        ax2    = ax.twinx()
        ax2.plot(x_kde, y_kde, color=BLUE, linewidth=1.8, label="KDE")
        ax2.set_ylabel("Density", fontsize=9, color=BLUE)
        ax2.tick_params(axis="y", labelcolor=BLUE)
        ax2.set_facecolor(BG)
        ax2.spines[["top", "right"]].set_color(LGREY)
        ax2.legend(fontsize=8, loc="upper right")

    ax.axvline(0,        color=DARK, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(pnl.mean(), color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Mean P&L (${pnl.mean():,.0f})")

    ax.set_xlabel("Trade Net P&L (USD)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"Trade P&L Distribution (n={len(trades)} trades)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=9, framealpha=0.7)
    fig.tight_layout()
    return _save(fig, "fig04_pnl_distribution.png", config)


# ---------------------------------------------------------------------------
# Figure 5: Rolling Sharpe Ratio
# ---------------------------------------------------------------------------

def fig05_rolling_sharpe(equity: pd.DataFrame, config: Config,
                          window: int = 252) -> str:
    """252-day rolling annualised Sharpe ratio."""
    from src.risk_metrics import compute_rolling_sharpe

    fig, ax = plt.subplots(figsize=(14, 4))
    _base_style(fig, ax)

    returns      = equity["nav"].pct_change().dropna()
    roll_sharpe  = compute_rolling_sharpe(returns, window, config.RISK_FREE_RATE)

    ax.plot(roll_sharpe.index, roll_sharpe, color=BLUE, linewidth=1.2, label="Rolling Sharpe")
    ax.fill_between(roll_sharpe.index, roll_sharpe, 0,
                    where=(roll_sharpe >= 0), color=GREEN, alpha=0.15)
    ax.fill_between(roll_sharpe.index, roll_sharpe, 0,
                    where=(roll_sharpe < 0),  color=RED,   alpha=0.15)
    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.axhline(1, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.7, label="Sharpe = 1.0")

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio (Annualised)", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Sharpe Ratio", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.7)
    fig.tight_layout()
    return _save(fig, "fig05_rolling_sharpe.png", config)


# ---------------------------------------------------------------------------
# Figure 6: Parameter Heatmap
# ---------------------------------------------------------------------------

def fig06_parameter_heatmap(sweep_results: pd.DataFrame, best_params: dict,
                              config: Config) -> str:
    """
    Window × entry threshold → OOS Sharpe ratio heatmap.
    Marks the selected (best) parameter combination with a star.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _base_style(fig, ax)

    pivot = sweep_results.pivot_table(
        values="oos_sharpe", index="window", columns="entry_thresh"
    )

    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, linecolor=LGREY,
                cbar_kws={"label": "OOS Sharpe Ratio"},
                annot_kws={"fontsize": 9})

    # Mark best parameters
    if best_params:
        rows = list(pivot.index)
        cols = list(pivot.columns)
        r_idx = rows.index(best_params["window"])   if best_params["window"] in rows else None
        c_idx = cols.index(best_params["entry_thresh"]) if best_params["entry_thresh"] in cols else None
        if r_idx is not None and c_idx is not None:
            ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1,
                                       fill=False, edgecolor=BLUE, lw=3, zorder=5))
            # Put [BEST] label in the top-right corner of the cell so it doesn't cover the value
            ax.text(c_idx + 0.96, r_idx + 0.90, "[BEST]", ha="right", va="top",
                    fontsize=7.5, color=BLUE, fontweight="bold", zorder=6)

    ax.set_title("Walk-Forward Parameter Optimisation — OOS Sharpe by Window & Entry Threshold",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Entry Threshold (σ)", fontsize=10)
    ax.set_ylabel("Rolling Window (days)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "fig06_parameter_heatmap.png", config)


# ---------------------------------------------------------------------------
# Figure 7: Statistical Test Summary Table
# ---------------------------------------------------------------------------

def fig07_statistical_summary(stat_results: dict, config: Config) -> str:
    """
    Renders a formatted summary table of all four statistical tests as a figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    _base_style(fig, ax)
    ax.axis("off")

    adf  = stat_results["adf"]["primary"]
    kpss = stat_results["kpss"]["level"]
    h    = stat_results["hurst"]
    ou   = stat_results["ou"]

    rows = [
        ["Test", "Statistic", "p-value / Value", "Critical (5%)", "Result"],
        ["ADF (AIC lags)", f"{adf['adf_stat']:.4f}",
         f"p = {adf['p_value']:.6f}", f"{adf['critical_5pct']:.4f}",
         "PASS  Reject unit root" if adf["reject_null_5pct"] else "FAIL  Cannot reject"],
        ["KPSS (level)", f"{kpss['kpss_stat']:.4f}",
         f"p = {kpss['p_value']:.6f}", f"{kpss['critical_5pct']:.4f}",
         "PASS  Stationary" if kpss["fail_to_reject_5pct"] else "FAIL  Non-stationary"],
        ["Hurst Exponent (R/S)", f"H = {h['H']:.4f}",
         f"R2 = {h['r_squared']:.4f}", "H < 0.5",
         "PASS  Mean-reverting" if h["H"] < 0.5 else "NOTE  Trending (level)"],
        ["OU Half-Life", f"beta = {ou['beta']:.5f}",
         f"tau = {ou['half_life_days']:.1f} days", "beta < 0",
         "PASS  Mean-reverting" if ou["beta"] < 0 else "FAIL  No reversion"],
    ]

    col_widths = [0.26, 0.18, 0.22, 0.16, 0.22]
    table = ax.table(
        cellText=rows[1:], colLabels=rows[0],
        colWidths=col_widths,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(LGREY)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EFF6FF")
        else:
            cell.set_facecolor(BG)
        # Colour result column
        if c == 4 and r > 0:
            txt = cell.get_text().get_text()
            cell.set_facecolor("#D1FAE5" if "✓" in txt else "#FEE2E2")

    ax.set_title("Statistical Tests for Mean Reversion in the 3:2:1 Crack Spread",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    return _save(fig, "fig07_statistical_summary.png", config)


# ---------------------------------------------------------------------------
# Figure 8: Yearly Performance Attribution
# ---------------------------------------------------------------------------

def fig08_yearly_performance(equity: pd.DataFrame, config: Config) -> str:
    """
    Per-calendar-year returns bar chart.
    Bars are coloured green/red by sign.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    _base_style(fig, ax)

    nav = equity["nav"].copy()
    nav.index = pd.to_datetime(nav.index)

    # Compute year-end NAVs
    year_end = nav.resample("YE").last()
    year_beg = nav.resample("YE").first().shift(1)
    year_ret = ((year_end - year_beg) / year_beg * 100).dropna()

    colors = [GREEN if r >= 0 else RED for r in year_ret.values]
    bars   = ax.bar(year_ret.index.year, year_ret.values, color=colors, width=0.6, alpha=0.85)

    for bar, val in zip(bars, year_ret.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.4 if val >= 0 else -1.5),
                f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, color=DARK, fontweight="bold")

    ax.axhline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_title("Annual Strategy Returns (%)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Annual Return (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xticks(year_ret.index.year)
    fig.tight_layout()
    return _save(fig, "fig08_yearly_performance.png", config)


# ---------------------------------------------------------------------------
# Figure 9: Benchmark Comparison
# ---------------------------------------------------------------------------

def fig09_benchmark_comparison(equity: pd.DataFrame, bm_metrics: dict,
                                 config: Config) -> str:
    """
    Normalised equity curves: Strategy vs SPY vs WTI buy-and-hold.
    All indexed to $1M at backtest start.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    _base_style(fig, ax)

    nav = equity["nav"]
    ax.plot(nav.index, nav / 1e6, color=BLUE, linewidth=1.8,
            label="Crack Spread Strategy", zorder=3)

    bm_colors = {"SPY": ORANGE, "WTI_BH": RED}
    bm_labels = {"SPY": "S&P 500 (SPY)", "WTI_BH": "WTI Crude (Buy-and-Hold)"}
    for key, bm in bm_metrics.items():
        if "nav_series" in bm:
            bm_nav = bm["nav_series"].reindex(nav.index).ffill()
            ax.plot(bm_nav.index, bm_nav / 1e6,
                    color=bm_colors.get(key, GREY), linewidth=1.2,
                    linestyle="--", alpha=0.8,
                    label=bm_labels.get(key, key), zorder=2)

    ax.axhline(config.INITIAL_NAV / 1e6, color=GREY, linewidth=0.6,
               linestyle=":", alpha=0.5)
    ax.set_title("Strategy vs. Benchmarks — Normalised NAV ($1M Base)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("NAV ($M)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}M"))
    ax.legend(fontsize=9, framealpha=0.7)
    fig.tight_layout()
    return _save(fig, "fig09_benchmark_comparison.png", config)


# ---------------------------------------------------------------------------
# Figure 10: Hurst R/S Log-Log Plot
# ---------------------------------------------------------------------------

def fig10_hurst_rs_analysis(hurst_result: dict, config: Config) -> str:
    """
    Log-log scatter of sub-period size (n) vs. mean R/S with OLS fit line.
    Slope of the fit line = Hurst exponent H.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    _base_style(fig, ax)

    log_n  = np.array(hurst_result["log_n"])
    log_rs = np.array(hurst_result["log_rs"])
    H      = hurst_result["H"]
    r2     = hurst_result["r_squared"]
    c      = hurst_result["intercept"]

    ax.scatter(log_n, log_rs, color=BLUE, s=50, zorder=3, label="Observed R/S")

    # Fitted line
    x_fit = np.linspace(log_n.min(), log_n.max(), 200)
    ax.plot(x_fit, H * x_fit + c, color=RED, linewidth=1.8,
            label=f"OLS fit: H = {H:.4f}  (R² = {r2:.4f})")

    # Reference lines for H=0.5 (BM) and the fitted H
    ax.plot(x_fit, 0.5 * x_fit + (np.mean(log_rs) - 0.5 * np.mean(log_n)),
            color=GREY, linewidth=1.0, linestyle=":", alpha=0.7, label="H = 0.5 (Random Walk)")

    ax.set_title("R/S Analysis — Hurst Exponent Estimation (log-log scale)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("log(Sub-period size n)", fontsize=10)
    ax.set_ylabel("log(Mean Rescaled Range E[R/S])", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)

    verdict = "Mean-Reverting" if H < 0.5 else ("Random Walk" if abs(H - 0.5) < 0.02 else "Trending")
    ax.text(0.97, 0.07, f"H = {H:.4f} → {verdict}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=RED,
            bbox=dict(boxstyle="round", facecolor=BG, alpha=0.9, edgecolor=LGREY))
    fig.tight_layout()
    return _save(fig, "fig10_hurst_rs_analysis.png", config)


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

def generate_all_figures(df: pd.DataFrame, equity: pd.DataFrame,
                          trades: pd.DataFrame, stat_results: dict,
                          sweep_results: pd.DataFrame, best_params: dict,
                          bm_metrics: dict, config: Config) -> dict:
    """
    Generate all 10 figures and return a dict of {fig_name: abs_path}.
    """
    from src.risk_metrics import compute_max_drawdown

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7: GENERATING FIGURES")
    logger.info("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")

    dd_info = compute_max_drawdown(equity["nav"])
    paths   = {}

    paths["fig01"] = fig01_crack_spread_history(df, config)
    paths["fig02"] = fig02_zscore_signals(df, config)
    paths["fig03"] = fig03_equity_curve(equity, dd_info["drawdown_series"], config)
    paths["fig04"] = fig04_pnl_distribution(trades, config)
    paths["fig05"] = fig05_rolling_sharpe(equity, config)
    paths["fig06"] = fig06_parameter_heatmap(sweep_results, best_params, config)
    paths["fig07"] = fig07_statistical_summary(stat_results, config)
    paths["fig08"] = fig08_yearly_performance(equity, config)
    paths["fig09"] = fig09_benchmark_comparison(equity, bm_metrics, config)
    paths["fig10"] = fig10_hurst_rs_analysis(stat_results["hurst"], config)

    logger.info(f"  All {len(paths)} figures saved to {config.FIGURES_DIR}")
    return paths
