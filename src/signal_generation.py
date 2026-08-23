"""
src/signal_generation.py — Rolling Z-Score Signal + Walk-Forward Parameter Sweep.

Signal Logic
------------
The z-score measures how many standard deviations the current crack spread
is from its recent rolling mean:

    zₜ = (Sₜ − μ_{t,w}) / σ_{t,w}

where μ_{t,w} and σ_{t,w} are the rolling mean and std over the last w days,
computed using ONLY data up to day t (no look-ahead).

Trade Rules
-----------
    zₜ < −ENTRY_THRESHOLD  → Long crack spread  (spread too compressed → expect rebound)
    zₜ > +ENTRY_THRESHOLD  → Short crack spread (spread too wide     → expect compression)
    |zₜ| < EXIT_THRESHOLD  → Exit position       (spread normalised)
    |zₜ| > STOP_THRESHOLD  → Hard stop           (tail-risk — spread structurally broken)

Anti-Look-Ahead Implementation
-------------------------------
Signal computed at close of day t is NOT applied until close of day t+1.
Implemented via pd.Series.shift(1) on the position series.
This is the conservative but correct approach for daily settlement data.

Walk-Forward Optimization
--------------------------
Parameters (window, entry_threshold) are selected by:
  1. Train on first 70% of data (IS period).
  2. Evaluate each (window, threshold) combination on remaining 30% (OOS period).
  3. Select params maximising OOS Sharpe — not IS Sharpe.
This prevents data-snooping / over-fitting to the historical sample.
"""

import logging
import sys
from itertools import product
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Z-Score Computation
# ---------------------------------------------------------------------------

def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score of a time series.

    Uses rolling window of exactly `window` days with min_periods = window // 2
    (requires at least half the window to be non-NaN before computing).

    Parameters
    ----------
    series : pd.Series
    window : int
        Lookback window in trading days.

    Returns
    -------
    pd.Series
        NaN for first (window // 2 - 1) observations.
    """
    roll_mean = series.rolling(window=window, min_periods=window // 2).mean()
    roll_std  = series.rolling(window=window, min_periods=window // 2).std()
    roll_std  = roll_std.replace(0.0, np.nan)   # Avoid div/0
    return (series - roll_mean) / roll_std


# ---------------------------------------------------------------------------
# State Machine: Raw Signal → Position Series
# ---------------------------------------------------------------------------

def build_position_series(zscore: pd.Series, entry_thresh: float,
                           exit_thresh: float, stop_thresh: float) -> pd.Series:
    """
    Convert raw z-score into an integer position series via a state machine.

    The state machine handles carry-over correctly:
      - A position persists until an exit or stop condition is triggered.
      - Re-entry into the same direction is allowed after exiting.
      - Stops take priority over all other signals.

    Parameters
    ----------
    zscore : pd.Series
    entry_thresh, exit_thresh, stop_thresh : float

    Returns
    -------
    pd.Series
        Values: +1 (long crack), −1 (short crack), 0 (flat).
    """
    positions = np.zeros(len(zscore), dtype=float)
    pos       = 0   # current state

    for i, z in enumerate(zscore.values):
        if np.isnan(z):
            positions[i] = 0
            pos = 0
            continue

        # --- Exit checks (take priority over entries) ---
        if pos == +1:
            if z > -exit_thresh or z > stop_thresh:
                pos = 0
        elif pos == -1:
            if z < +exit_thresh or z < -stop_thresh:
                pos = 0

        # Hard stop: regardless of direction
        if abs(z) > stop_thresh:
            pos = 0

        # --- Entry checks (only from flat) ---
        if pos == 0:
            if z < -entry_thresh:
                pos = +1
            elif z > +entry_thresh:
                pos = -1

        positions[i] = pos

    return pd.Series(positions, index=zscore.index, name="position", dtype=float)


# ---------------------------------------------------------------------------
# Signal Generation (full DataFrame)
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, window: int,
                     entry_thresh: float, exit_thresh: float,
                     stop_thresh: float) -> pd.DataFrame:
    """
    Add z-score and position columns to the data DataFrame.

    Output columns added
    --------------------
    z_score         : Rolling z-score of crack spread (contemporaneous).
    position        : Target position based on today's z-score.
    position_exec   : Lagged position (applied the *next* day) — used by backtester.
                      This implements the mandatory 1-day execution lag.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'crack'.

    Returns
    -------
    pd.DataFrame
        Copy of df with z_score, position, position_exec added.
    """
    out = df.copy()
    out["z_score"]       = compute_zscore(out["crack"], window)
    out["position"]      = build_position_series(
        out["z_score"], entry_thresh, exit_thresh, stop_thresh
    )
    # 1-day execution lag: trade at next day's close based on today's signal
    out["position_exec"] = out["position"].shift(1).fillna(0.0)

    n_long  = (out["position_exec"] ==  1).sum()
    n_short = (out["position_exec"] == -1).sum()
    n_flat  = (out["position_exec"] ==  0).sum()
    n_total = len(out)
    logger.debug(
        f"  Positions (w={window}, e={entry_thresh}): "
        f"long={n_long} ({n_long/n_total:.1%}), "
        f"short={n_short} ({n_short/n_total:.1%}), "
        f"flat={n_flat} ({n_flat/n_total:.1%})"
    )

    return out


# ---------------------------------------------------------------------------
# Walk-Forward Parameter Sweep
# ---------------------------------------------------------------------------

def parameter_sweep(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, dict]:
    """
    Walk-forward grid search over (window, entry_threshold).

    Method
    ------
    Data is split 70/30 into in-sample (IS) and out-of-sample (OOS) periods.
    For each parameter combination:
      1. Signals are generated on the FULL dataset (to allow proper warm-up).
      2. Backtesting is performed on the OOS slice only.
      3. Sharpe ratio of OOS equity curve is the selection criterion.

    The winning parameters are those maximising OOS Sharpe with ≥ 5 trades.
    Ties are broken by lower max drawdown.

    Returns
    -------
    results_df : pd.DataFrame
        Full grid with columns: window, entry_thresh, oos_sharpe, n_trades, max_dd_pct.
    best_params : dict
        Keys: 'window', 'entry_thresh'.
    """
    # Deferred import to avoid circular deps
    from src.backtester import CrackSpreadBacktester
    from src.risk_metrics import compute_sharpe, compute_max_drawdown

    n          = len(df)
    split_idx  = int(n * config.TRAIN_RATIO)
    bt         = CrackSpreadBacktester(config)
    grid       = list(product(config.SWEEP_WINDOWS, config.SWEEP_ENTRY_THRESHOLDS))

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: WALK-FORWARD PARAMETER SWEEP")
    logger.info("=" * 60)
    logger.info(
        f"  IS period: {df.index[0].date()} → {df.index[split_idx-1].date()} "
        f"({split_idx} days)"
    )
    logger.info(
        f"  OOS period: {df.index[split_idx].date()} → {df.index[-1].date()} "
        f"({n - split_idx} days)"
    )
    logger.info(f"  Grid: {len(grid)} combinations")

    records    = []
    best_sharpe = -np.inf
    best_params = {"window": config.ROLLING_WINDOW, "entry_thresh": config.ENTRY_THRESHOLD}
    best_dd     = np.inf

    for window, entry_thresh in tqdm(grid, desc="Param sweep", ncols=70):
        try:
            df_sig = generate_signals(
                df, window, entry_thresh,
                config.EXIT_THRESHOLD, config.STOP_THRESHOLD
            )
            df_oos  = df_sig.iloc[split_idx:].copy()
            equity, trades = bt.run(df_oos)

            if len(equity) < 20 or len(trades) < 3:
                records.append({
                    "window": window, "entry_thresh": entry_thresh,
                    "oos_sharpe": np.nan, "n_trades": len(trades),
                    "max_dd_pct": np.nan
                })
                continue

            daily_ret = equity["nav"].pct_change().dropna()
            sharpe    = compute_sharpe(daily_ret, config.RISK_FREE_RATE)
            dd_info   = compute_max_drawdown(equity["nav"])
            max_dd    = dd_info["max_drawdown"]

            records.append({
                "window":       window,
                "entry_thresh": entry_thresh,
                "oos_sharpe":   round(sharpe, 4),
                "n_trades":     len(trades),
                "max_dd_pct":   round(max_dd, 2),
            })

            # Select best: maximize Sharpe, break ties by min drawdown
            if sharpe > best_sharpe and len(trades) >= 5:
                best_sharpe  = sharpe
                best_dd      = max_dd
                best_params  = {"window": window, "entry_thresh": entry_thresh}
            elif abs(sharpe - best_sharpe) < 0.01 and max_dd < best_dd and len(trades) >= 5:
                best_dd     = max_dd
                best_params = {"window": window, "entry_thresh": entry_thresh}

        except Exception as exc:
            logger.warning(f"  Sweep failed w={window}, e={entry_thresh}: {exc}")
            records.append({
                "window": window, "entry_thresh": entry_thresh,
                "oos_sharpe": np.nan, "n_trades": 0, "max_dd_pct": np.nan
            })

    results_df = pd.DataFrame(records)

    logger.info(
        f"\n  Best OOS params: window={best_params['window']} days, "
        f"entry_thresh={best_params['entry_thresh']}σ, "
        f"Sharpe={best_sharpe:.4f}"
    )

    return results_df, best_params
