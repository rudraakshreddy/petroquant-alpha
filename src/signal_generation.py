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
                           exit_thresh: float, stop_thresh: float,
                           max_hold: int | None = None) -> pd.Series:
    """
    Convert the z-score into an integer position series via a state machine.

    Rules
    -----
      - Enter long when z < -entry_thresh, short when z > +entry_thresh.
      - Exit when |z| < exit_thresh (the deviation has normalised).
      - Stop out when |z| > stop_thresh, and SUPPRESS RE-ENTRY until |z| falls
        back below entry_thresh.
      - Close any position held longer than `max_hold` sessions.

    Why the lockout is required
    ---------------------------
    Because stop_thresh > entry_thresh by construction, every bar on which the
    stop fires also satisfies the entry condition. A stop implemented as a bare
    flattening would therefore close and immediately reopen the same position on
    the same bar, leaving exposure unchanged and making the control vacuous. The
    `locked` flag is what gives the stop effect: once triggered, no new position
    is opened until the deviation has returned to a range in which the
    mean-reversion hypothesis is again tenable.

    Why the time stop
    -----------------
    The fitted Ornstein-Uhlenbeck half-life implies a deviation should decay by
    half within ~28 sessions. A position held far beyond that horizon without
    reverting is evidence against the model that motivated it, so it is closed.
    The threshold is derived from the fitted process (2 x half-life), not tuned.

    Parameters
    ----------
    zscore : pd.Series
    entry_thresh, exit_thresh, stop_thresh : float
    max_hold : int, optional
        Maximum holding period in sessions. None disables the time stop.

    Returns
    -------
    pd.Series
        Values: +1 (long crack), -1 (short crack), 0 (flat).
    """
    positions = np.zeros(len(zscore), dtype=float)
    pos       = 0       # current state
    held      = 0       # sessions the current position has been open
    locked    = False   # re-entry suppressed after a stop

    for i, z in enumerate(zscore.values):
        if np.isnan(z):
            positions[i] = 0
            pos, held = 0, 0
            continue

        # --- Normal exit: the deviation has normalised ---
        if pos == +1 and z > -exit_thresh:
            pos, held = 0, 0
        elif pos == -1 and z < +exit_thresh:
            pos, held = 0, 0

        # --- Hard stop, with re-entry lockout ---
        if abs(z) > stop_thresh:
            pos, held = 0, 0
            locked = True

        # --- Time stop ---
        if max_hold is not None and pos != 0 and held >= max_hold:
            pos, held = 0, 0

        # --- Release the lockout once the deviation is tradeable again ---
        if locked and abs(z) < entry_thresh:
            locked = False

        # --- Entry (only from flat, and only when not locked out) ---
        if pos == 0 and not locked:
            if z < -entry_thresh:
                pos, held = +1, 0
            elif z > +entry_thresh:
                pos, held = -1, 0

        if pos != 0:
            held += 1
        positions[i] = pos

    return pd.Series(positions, index=zscore.index, name="position", dtype=float)


# ---------------------------------------------------------------------------
# Signal Generation (full DataFrame)
# ---------------------------------------------------------------------------

def _harmonic_design(doy: np.ndarray, K: int) -> np.ndarray:
    """Design matrix of K sine/cosine pairs on the annual cycle, plus intercept."""
    cols = [np.ones(len(doy))]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25),
                 np.cos(2 * np.pi * k * doy / 365.25)]
    return np.column_stack(cols)


def deseasonalise(series: pd.Series, harmonics: int = 1,
                  min_train: int = 504) -> pd.Series:
    """
    Remove the annual seasonal component of the refining margin from the signal.

    Refining margins follow a documented annual cycle driven by the northern
    hemisphere driving season, winter heating demand and the spring/autumn
    refinery turnaround periods. That component is predictable, so a z-score
    computed on the raw level would read it as disequilibrium and trade against
    it. The signal is therefore computed on the residual of

        S_t = c0 + sum_k [ a_k sin(2 pi k d_t / 365.25)
                         + b_k cos(2 pi k d_t / 365.25) ] + u_t

    Two properties are essential to validity:

      1. Coefficients are estimated on an EXPANDING window using only
         observations strictly prior to t, refitted once per calendar year. No
         future information enters the seasonal estimate.
      2. The adjustment applies to the SIGNAL ONLY. Profit and loss continues to
         be marked on the actual traded spread. Backtesting on the
         deseasonalised series would credit the strategy with a return it cannot
         realise.

    The first `min_train` observations have no seasonal estimate and are
    returned as NaN, so no position is taken until the fit is defined.

    Parameters
    ----------
    series : pd.Series
        Crack spread indexed by date.
    harmonics : int
        Number of sine/cosine pairs. K=1 is the a priori choice: the mechanism
        is a single annual cycle, and each extra harmonic adds two parameters
        estimated on a short history.
    min_train : int
        Minimum observations before the first seasonal fit.

    Returns
    -------
    pd.Series
        Deseasonalised residual, NaN over the warm-up period.
    """
    doy = pd.to_datetime(series.index).dayofyear.values
    v = series.values
    out = np.full(len(v), np.nan)
    beta, last_year = None, None

    for i in range(len(v)):
        year = pd.Timestamp(series.index[i]).year
        if i >= min_train and year != last_year:      # refit annually, history only
            X, y = _harmonic_design(doy[:i], harmonics), v[:i]
            ok = np.isfinite(y)
            if ok.sum() > min_train:
                beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                last_year = year
        if beta is not None:
            out[i] = v[i] - _harmonic_design(np.array([doy[i]]), harmonics).dot(beta)[0]

    return pd.Series(out, index=series.index, name="crack_deseason")


def generate_signals(df: pd.DataFrame, window: int,
                     entry_thresh: float, exit_thresh: float,
                     stop_thresh: float, max_hold: int | None = None,
                     harmonics: int = 1, min_train: int = 504) -> pd.DataFrame:
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

    # Signal is computed on the DESEASONALISED spread; P&L is always marked on
    # the untransformed traded spread. See deseasonalise() for the rationale.
    if "crack_deseason" not in out.columns:
        out["crack_deseason"] = deseasonalise(out["crack"], harmonics, min_train)
    out["z_score"]       = compute_zscore(out["crack_deseason"], window)
    out["position"]      = build_position_series(
        out["z_score"], entry_thresh, exit_thresh, stop_thresh, max_hold
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
                config.EXIT_THRESHOLD, config.STOP_THRESHOLD,
                max_hold  = config.MAX_HOLD_DAYS,
                harmonics = config.SEASONAL_HARMONICS,
                min_train = config.SEASONAL_MIN_TRAIN,
            )
            # Selection uses the TRAINING segment only. Choosing parameters by
            # their performance on the held-out segment would turn that segment
            # into a selection set and bias the reported result upward.
            df_is   = df_sig.iloc[:split_idx].copy()
            df_oos  = df_sig.iloc[split_idx:].copy()
            equity, trades = bt.run(df_is)

            if len(equity) < 20 or len(trades) < 3:
                records.append({
                    "window": window, "entry_thresh": entry_thresh,
                    "is_sharpe": np.nan, "oos_sharpe": np.nan,
                    "n_trades": len(trades), "max_dd_pct": np.nan
                })
                continue

            daily_ret = equity["nav"].pct_change().dropna()
            sharpe    = compute_sharpe(daily_ret, config.RISK_FREE_RATE)
            dd_info   = compute_max_drawdown(equity["nav"])
            max_dd    = dd_info["max_drawdown"]

            # Held-out performance is RECORDED for reporting, never used to select.
            eq_o, tr_o = bt.run(df_oos)
            oos_sharpe = (compute_sharpe(eq_o["nav"].pct_change().dropna(),
                                         config.RISK_FREE_RATE)
                          if len(eq_o) >= 20 and len(tr_o) >= 3 else np.nan)

            records.append({
                "window":       window,
                "entry_thresh": entry_thresh,
                "is_sharpe":    round(sharpe, 4),
                "oos_sharpe":   round(oos_sharpe, 4) if oos_sharpe == oos_sharpe else np.nan,
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
        f"\n  Selected on training segment: window={best_params['window']} days, "
        f"entry_thresh={best_params['entry_thresh']}σ, "
        f"IS Sharpe={best_sharpe:.4f}"
    )

    return results_df, best_params
