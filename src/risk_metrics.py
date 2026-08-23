"""
src/risk_metrics.py — Full Risk and Performance Metrics Suite.

All metrics are computed from first principles; no external risk library.

Conventions
-----------
- Annual scaling: 252 trading days per year (CME standard).
- Risk-free rate sourced from Config (default 5% p.a. — approx US T-bill avg 2019-2024).
- All percentage metrics reported as decimals internally; converted to % in the report.
- Drawdown is computed on NAV levels (absolute), not returns.

Metrics Computed
----------------
Returns:         Total Return, CAGR
Adjusted:        Sharpe Ratio, Sortino Ratio, Calmar Ratio
Risk:            Annualised Volatility, Max Drawdown, VaR (95%, 99%), CVaR (95%)
Trade-level:     Hit Rate, Profit Factor, Avg Win/Loss, Avg Hold Duration
Benchmark:       Alpha, Beta (vs SPY and WTI)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def compute_cagr(nav: pd.Series) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (V_final / V_initial)^(252 / n_days) − 1
    """
    n = len(nav.dropna())
    if n < 2:
        return 0.0
    v0, vt = float(nav.dropna().iloc[0]), float(nav.dropna().iloc[-1])
    if v0 <= 0:
        return 0.0
    return float((vt / v0) ** (252.0 / n) - 1.0)


def compute_sharpe(returns: pd.Series, rf_annual: float = 0.05) -> float:
    """
    Annualised Sharpe ratio using daily excess returns.

    Sharpe = (E[rₜ − rf_daily]) / σ[rₜ − rf_daily] × √252

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (fractions, NOT percentages).
    rf_annual : float
        Annual risk-free rate.
    """
    rf_daily = rf_annual / 252.0
    excess   = returns.dropna() - rf_daily
    if len(excess) < 5 or excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252.0))


def compute_sortino(returns: pd.Series, rf_annual: float = 0.05) -> float:
    """
    Sortino ratio: penalises only downside deviation.

    Sortino = (E[rₜ − rf]) / σ_downside × √252
    where σ_downside = std of (rₜ − rf) when (rₜ − rf) < 0.
    """
    rf_daily = rf_annual / 252.0
    excess   = returns.dropna() - rf_daily
    down     = excess[excess < 0]
    if len(down) < 2 or down.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / down.std() * np.sqrt(252.0))


def compute_max_drawdown(nav: pd.Series) -> dict:
    """
    Maximum peak-to-trough drawdown and related timeline metadata.

    Returns
    -------
    dict
        max_drawdown   : worst drawdown as a fraction (negative number)
        peak_date      : date of the pre-drawdown peak
        trough_date    : date of the worst drawdown trough
        recovery_date  : first date NAV recovered to prior peak (None if not yet)
        drawdown_dur   : days from peak to trough
        recovery_dur   : days from trough to recovery (None if not yet)
        drawdown_series: full drawdown series (for plotting)
    """
    nav_clean    = nav.dropna()
    rolling_max  = nav_clean.cummax()
    drawdown     = (nav_clean - rolling_max) / rolling_max

    max_dd       = float(drawdown.min())
    trough_idx   = drawdown.idxmin()
    peak_idx     = rolling_max.loc[:trough_idx].idxmax()

    # Recovery: first date after trough where NAV ≥ rolling max at trough date
    peak_nav     = float(rolling_max.loc[trough_idx])
    post_trough  = nav_clean.loc[trough_idx:]
    recovered    = post_trough[post_trough >= peak_nav]
    recovery_idx = recovered.index[0] if len(recovered) > 0 else None

    dd_dur  = int((trough_idx - peak_idx).days)
    rec_dur = int((recovery_idx - trough_idx).days) if recovery_idx is not None else None

    return {
        "max_drawdown":   round(max_dd * 100.0, 2),  # as %
        "peak_date":      peak_idx,
        "trough_date":    trough_idx,
        "recovery_date":  recovery_idx,
        "drawdown_dur":   dd_dur,
        "recovery_dur":   rec_dur,
        "drawdown_series": drawdown,
    }


def compute_calmar(nav: pd.Series) -> float:
    """
    Calmar ratio = Annualised Return / |Max Drawdown|.
    Higher is better.
    """
    ann_ret = compute_cagr(nav)
    dd_info = compute_max_drawdown(nav)
    max_dd  = abs(dd_info["max_drawdown"] / 100.0)
    if max_dd < 1e-9:
        return float("inf")
    return round(float(ann_ret / max_dd), 4)


def compute_var(returns: pd.Series, confidence: float = 0.95) -> dict:
    """
    Value at Risk — historical and parametric (normal distribution).

    Parameters
    ----------
    confidence : float
        e.g. 0.95 for 95% VaR.

    Returns
    -------
    dict
        'historical' : empirical quantile VaR (% of portfolio)
        'parametric' : normal-distribution VaR (% of portfolio)
    """
    r       = returns.dropna()
    alpha   = 1.0 - confidence
    h_var   = float(r.quantile(alpha))
    p_var   = float(r.mean() - stats.norm.ppf(confidence) * r.std())
    return {
        "confidence":  confidence,
        "historical":  round(h_var * 100.0, 4),
        "parametric":  round(p_var * 100.0, 4),
    }


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall): mean of returns below VaR threshold.
    Expressed as % of portfolio.
    """
    r      = returns.dropna()
    cutoff = r.quantile(1.0 - confidence)
    tail   = r[r <= cutoff]
    if len(tail) == 0:
        return 0.0
    return round(float(tail.mean()) * 100.0, 4)


def compute_hit_rate(trades: pd.DataFrame) -> dict:
    """
    Trade-level performance statistics.

    Returns
    -------
    dict
        hit_rate_pct, profit_factor, avg_win_usd, avg_loss_usd,
        n_wins, n_losses, expectancy_usd
    """
    if trades is None or len(trades) == 0:
        return {k: 0.0 for k in ["hit_rate_pct", "profit_factor", "avg_win_usd",
                                   "avg_loss_usd", "n_wins", "n_losses", "expectancy_usd"]}

    wins   = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    hit_rate      = float(len(wins)) / float(len(trades))
    avg_win       = float(wins["net_pnl"].mean())   if len(wins)   > 0 else 0.0
    avg_loss      = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
    gross_profit  = float(wins["net_pnl"].sum())    if len(wins)   > 0 else 0.0
    gross_loss    = abs(float(losses["net_pnl"].sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy    = hit_rate * avg_win + (1.0 - hit_rate) * avg_loss

    return {
        "hit_rate_pct":   round(hit_rate * 100.0, 2),
        "profit_factor":  round(profit_factor, 4),
        "avg_win_usd":    round(avg_win, 2),
        "avg_loss_usd":   round(avg_loss, 2),
        "n_wins":         int(len(wins)),
        "n_losses":       int(len(losses)),
        "expectancy_usd": round(expectancy, 2),
    }


def compute_rolling_sharpe(returns: pd.Series, window: int = 252,
                            rf_annual: float = 0.05) -> pd.Series:
    """Rolling Sharpe ratio over `window` days."""
    rf_daily  = rf_annual / 252.0
    excess    = returns - rf_daily
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).rename("rolling_sharpe")


# ---------------------------------------------------------------------------
# Full Report
# ---------------------------------------------------------------------------

def compute_full_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                          config: Config) -> dict:
    """
    Compute and log the complete performance and risk metrics report.

    Parameters
    ----------
    equity : pd.DataFrame
        Output of CrackSpreadBacktester.run() — must contain column 'nav'.
    trades : pd.DataFrame
        Trade log from backtester.
    config : Config

    Returns
    -------
    dict
        All computed metrics as a flat dictionary.
    """
    nav     = equity["nav"].dropna()
    returns = nav.pct_change().dropna()

    sharpe   = compute_sharpe(returns,  config.RISK_FREE_RATE)
    sortino  = compute_sortino(returns, config.RISK_FREE_RATE)
    dd_info  = compute_max_drawdown(nav)
    calmar   = compute_calmar(nav)
    cagr     = compute_cagr(nav)
    var95    = compute_var(returns, 0.95)
    var99    = compute_var(returns, 0.99)
    cvar95   = compute_cvar(returns, 0.95)
    trade_m  = compute_hit_rate(trades)
    ann_vol  = float(returns.std() * np.sqrt(252.0))
    total_ret = float((nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0])

    avg_hold = float(trades["hold_days"].mean()) if len(trades) > 0 else 0.0
    long_t   = trades[trades["direction"] ==  1] if len(trades) > 0 else pd.DataFrame()
    short_t  = trades[trades["direction"] == -1] if len(trades) > 0 else pd.DataFrame()
    total_costs = float(trades["total_cost"].sum()) if len(trades) > 0 else 0.0

    metrics = {
        # ---- Return Metrics ----
        "total_return_pct":      round(total_ret * 100.0, 2),
        "cagr_pct":              round(cagr * 100.0, 2),
        "ann_volatility_pct":    round(ann_vol * 100.0, 2),
        # ---- Risk-Adjusted ----
        "sharpe_ratio":          round(sharpe, 4),
        "sortino_ratio":         round(sortino, 4),
        "calmar_ratio":          round(calmar, 4),
        # ---- Drawdown ----
        "max_drawdown_pct":      dd_info["max_drawdown"],
        "peak_date":             str(dd_info["peak_date"].date()),
        "trough_date":           str(dd_info["trough_date"].date()),
        "drawdown_dur_days":     dd_info["drawdown_dur"],
        "recovery_dur_days":     dd_info["recovery_dur"],
        # ---- Tail Risk ----
        "var_95_daily_pct":      var95["historical"],
        "var_99_daily_pct":      var99["historical"],
        "cvar_95_daily_pct":     cvar95,
        # ---- Trade Stats ----
        "n_trades":              int(len(trades)),
        "n_long_trades":         int(len(long_t)),
        "n_short_trades":        int(len(short_t)),
        "hit_rate_pct":          trade_m["hit_rate_pct"],
        "profit_factor":         trade_m["profit_factor"],
        "avg_win_usd":           trade_m["avg_win_usd"],
        "avg_loss_usd":          trade_m["avg_loss_usd"],
        "expectancy_usd":        trade_m["expectancy_usd"],
        "avg_hold_days":         round(avg_hold, 1),
        "total_costs_usd":       round(total_costs, 2),
        # ---- NAV ----
        "initial_nav_usd":       config.INITIAL_NAV,
        "final_nav_usd":         round(float(nav.iloc[-1]), 2),
        "total_pnl_usd":         round(float(nav.iloc[-1] - nav.iloc[0]), 2),
    }

    # Log table
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: PERFORMANCE METRICS")
    logger.info("=" * 60)
    col_w = 30
    for k, v in metrics.items():
        logger.info(f"  {k:<{col_w}}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# Benchmark Comparison
# ---------------------------------------------------------------------------

def compute_benchmark_metrics(benchmark: pd.DataFrame, equity: pd.DataFrame,
                               config: Config) -> dict:
    """
    Compute Sharpe, CAGR, max drawdown for SPY and WTI buy-and-hold.
    Also compute strategy alpha and beta relative to each benchmark.

    Parameters
    ----------
    benchmark : pd.DataFrame
        Columns: 'SPY', 'WTI_BH'. Price series (not NAV).
    equity : pd.DataFrame
        Strategy equity curve with 'nav' column.

    Returns
    -------
    dict
        Keys: 'SPY', 'WTI_BH'. Each is a dict of performance metrics + nav_series.
    """
    start = equity.index[0]
    end   = equity.index[-1]
    bm    = benchmark.loc[start:end].copy()

    results = {}
    strat_rets = equity["nav"].pct_change().dropna()

    for col in ["SPY", "WTI_BH"]:
        if col not in bm.columns:
            continue
        price  = bm[col].dropna()
        if len(price) < 20:
            continue

        # Align to strategy dates
        price = price.reindex(equity.index).ffill().dropna()

        # NAV-normalised series (starting at strategy initial NAV)
        nav_bm = config.INITIAL_NAV * (price / price.iloc[0])
        rets   = price.pct_change().dropna()

        # Alpha / Beta
        common_idx = strat_rets.index.intersection(rets.index)
        s_r = strat_rets.loc[common_idx].values
        b_r = rets.loc[common_idx].values
        if len(s_r) > 10:
            slope_, intercept_, _, _, _ = stats.linregress(b_r, s_r)
            beta_vs  = round(float(slope_), 4)
            alpha_vs = round(float(intercept_ * 252.0), 4)   # annualised alpha
        else:
            beta_vs, alpha_vs = None, None

        results[col] = {
            "nav_series":       nav_bm,
            "total_return_pct": round(float((price.iloc[-1] / price.iloc[0] - 1) * 100.0), 2),
            "cagr_pct":         round(float(compute_cagr(nav_bm) * 100.0), 2),
            "sharpe_ratio":     round(float(compute_sharpe(rets, config.RISK_FREE_RATE)), 4),
            "max_drawdown_pct": compute_max_drawdown(nav_bm)["max_drawdown"],
            "ann_vol_pct":      round(float(rets.std() * np.sqrt(252.0) * 100.0), 2),
            "strategy_beta":    beta_vs,
            "strategy_alpha_ann": alpha_vs,
        }

    return results
