"""
src/spread_construction.py — 3:2:1 Crack Spread Computation.

Physical Rationale
------------------
A typical North American refinery operates roughly on a 3:2:1 yield:
    3 barrels of crude oil  →  2 barrels of gasoline  +  1 barrel of distillate

The 3:2:1 crack spread is therefore the synthetic *gross refining margin*
per barrel of crude processed:

    Crack_t = (2 × RBOB_t + 1 × HO_t  −  3 × WTI_t) / 3   [$/bbl]

All prices must be in $/barrel BEFORE applying this formula.

Why Mean Reversion Is Expected
-------------------------------
When crack spreads are high (refining margins are fat):
  1. Existing refineries run at higher utilisation rates.
  2. Economically marginal refineries (mothballed) come back online.
  3. Increased crude demand pushes WTI higher.
  4. Increased product supply pushes RBOB/HO lower.
  → Margins compress back toward long-run equilibrium.

The reverse holds when spreads are depressed (e.g., COVID demand collapse
in Apr 2020 briefly sent crack spreads negative).

This module computes the spread and its full set of rolling statistics
needed by the signal generator.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Spread Formula
# ---------------------------------------------------------------------------

def compute_crack_spread(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 3:2:1 crack spread and all derivative statistics.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns 'WTI', 'RBOB', 'HO' in $/barrel.

    Returns
    -------
    pd.DataFrame
        Original panel plus the following columns:

        crack            : 3:2:1 crack spread ($/bbl)
        crack_chg        : Day-over-day change ($/bbl)
        crack_pct        : Day-over-day % change
        crack_vol_20     : 20-day rolling std of crack changes (used for vol-targeting)
        crack_vol_60     : 60-day rolling std of crack changes (longer context)
        roll_mean_252    : 252-day rolling mean (annual context band)
        roll_std_252     : 252-day rolling std
        roll_upper_252   : roll_mean_252 + 2 × roll_std_252
        roll_lower_252   : roll_mean_252 − 2 × roll_std_252
    """
    df = panel.copy()

    # ---- Core formula ----
    df["crack"] = (2.0 * df["RBOB"] + df["HO"] - 3.0 * df["WTI"]) / 3.0

    # ---- Daily changes ----
    df["crack_chg"] = df["crack"].diff()
    df["crack_pct"] = df["crack"].pct_change() * 100.0

    # ---- Rolling volatility (for position sizing) ----
    df["crack_vol_20"] = df["crack_chg"].rolling(20,  min_periods=10).std()
    df["crack_vol_60"] = df["crack_chg"].rolling(60,  min_periods=30).std()

    # ---- Long-horizon context band (±2σ Bollinger-style) ----
    df["roll_mean_252"]  = df["crack"].rolling(252, min_periods=126).mean()
    df["roll_std_252"]   = df["crack"].rolling(252, min_periods=126).std()
    df["roll_upper_252"] = df["roll_mean_252"] + 2.0 * df["roll_std_252"]
    df["roll_lower_252"] = df["roll_mean_252"] - 2.0 * df["roll_std_252"]

    # ---- Logging ----
    crack = df["crack"].dropna()
    logger.info("3:2:1 Crack Spread Summary:")
    logger.info(f"  Mean   : ${crack.mean():.3f}/bbl")
    logger.info(f"  Std    : ${crack.std():.3f}/bbl")
    logger.info(f"  Min    : ${crack.min():.3f}/bbl  (on {crack.idxmin().date()})")
    logger.info(f"  Max    : ${crack.max():.3f}/bbl  (on {crack.idxmax().date()})")
    logger.info(f"  Skew   : {crack.skew():.3f}")
    logger.info(f"  Kurtosis: {crack.kurt():.3f}")

    return df


# ---------------------------------------------------------------------------
# Descriptive Tables
# ---------------------------------------------------------------------------

def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-year descriptive statistics for the crack spread.

    Returns
    -------
    pd.DataFrame
        Rows = calendar years. Columns = mean, std, min, median, max, Q25, Q75.
    """
    crack = df["crack"].copy()
    crack.index = pd.to_datetime(crack.index)

    summary = (
        crack.groupby(crack.index.year)
        .agg(
            Mean   = ("mean"),
            Std    = ("std"),
            Min    = ("min"),
            Q25    = (lambda x: x.quantile(0.25)),
            Median = ("median"),
            Q75    = (lambda x: x.quantile(0.75)),
            Max    = ("max"),
        )
        .round(2)
    )
    summary.index.name = "Year"
    return summary


def component_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlations of daily *price changes* for WTI, RBOB, HO.

    Using changes (not levels) avoids spurious correlation from shared trends.
    """
    changes = panel[["WTI", "RBOB", "HO"]].diff().dropna()
    return changes.corr().round(4)


def crack_vs_components_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regression of crack spread changes on component changes.
    Reports beta coefficients to quantify each leg's sensitivity.
    """
    chg = df[["WTI", "RBOB", "HO", "crack_chg"]].diff().dropna()

    rows = []
    for col in ["WTI", "RBOB", "HO"]:
        x   = chg[col].values
        y   = chg["crack_chg"].values
        ok  = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        x_, y_ = x[ok], y[ok]
        X       = np.column_stack([np.ones(len(x_)), x_])
        betas, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
        corr    = np.corrcoef(x_, y_)[0, 1]
        rows.append({"Component": col, "Beta": round(betas[1], 4), "Correlation": round(corr, 4)})

    return pd.DataFrame(rows).set_index("Component")
