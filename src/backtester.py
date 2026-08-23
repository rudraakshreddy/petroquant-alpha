"""
src/backtester.py — Custom Event-Loop Backtester.

Design Philosophy
-----------------
We deliberately avoid third-party backtesting frameworks (backtrader, zipline,
vectorbt) because:
  1. We must be able to explain every computational step in an interview.
  2. We need precise control over transaction cost accounting.
  3. We need to document the anti-look-ahead conventions unambiguously.

The backtester simulates a desk that:
  - Receives daily settlement prices for WTI, RBOB, and HO.
  - Synthetically trades the 3:2:1 crack spread by simultaneously
    buying/selling the three legs (replicating the physical refinery margin).
  - Sizes positions to target a fixed daily NAV risk (volatility targeting).
  - Pays realistic costs: bid-ask slippage + brokerage commission + roll cost.

Timing Convention
-----------------
Signal generated at: close of day t   (using z-score from t)
Trade executed at:   close of day t+1 (position_exec = position.shift(1))
P&L realised at:     close of day t+1 via mark-to-market

This means if the signal fires on Monday's close, we enter at Tuesday's close.
We capture the Tuesday close-to-Wednesday close move as our first P&L day.
This is conservative and prevents look-ahead bias.

Position Sizing
---------------
We use volatility targeting:
    N_bbl = floor( VOL_TARGET_DAILY × NAV / σ_crack )
where σ_crack is the 20-day rolling std of daily crack changes.
This automatically scales down in volatile markets (e.g., COVID March 2020).
Minimum size is always 1 contract (1,000 bbl).

Transaction Costs
-----------------
One-way cost (paid at entry AND at exit):
    Slippage   = SLIPPAGE_PER_BBL × N_bbl         ($/position)
    Commission = COMMISSION_PER_CONTRACT × N_contracts  ($/position)
Roll cost (paid once per month-boundary crossed while in position):
    Roll       = ROLL_COST_PER_BBL × N_bbl
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


class CrackSpreadBacktester:
    """
    Single-spread event-loop backtester.

    Parameters
    ----------
    config : Config
        All strategy and cost parameters.
    """

    def __init__(self, config: Config):
        self.cfg = config

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the backtest on a signal-annotated DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns:
              - 'crack'         : crack spread in $/bbl
              - 'z_score'       : contemporaneous z-score
              - 'position_exec' : 1-day lagged target position (+1/−1/0)
              - 'crack_vol_20'  : 20-day rolling vol of crack changes (for sizing)

        Returns
        -------
        equity : pd.DataFrame
            Daily equity curve. Columns:
            nav, daily_pnl, position, pos_size_bbl, roll_cost, action, crack, z_score.
        trades : pd.DataFrame
            One row per completed round-trip trade. Columns:
            entry_date, exit_date, direction, entry_spread, exit_spread,
            spread_chg, pos_size_bbl, gross_pnl, total_cost, net_pnl,
            hold_days, exit_reason.
        """
        cfg = self.cfg
        n   = len(df)

        if n == 0:
            logger.warning("Backtester received empty DataFrame. Returning empty results.")
            return pd.DataFrame(), pd.DataFrame()

        # ---- State ----
        nav           = float(cfg.INITIAL_NAV)
        position      = 0       # +1 long, −1 short, 0 flat
        entry_spread  = 0.0
        entry_date    = None
        pos_size_bbl  = 0       # locked at entry (barrels of crude-equiv)
        entry_cost_   = 0.0     # transaction cost paid at entry (for trade record)

        equity_rows = []
        trade_rows  = []

        for i in range(n):
            date       = df.index[i]
            crack      = float(df["crack"].iloc[i])
            z          = float(df.get("z_score", pd.Series(0.0, index=df.index)).iloc[i] or 0.0)
            target_pos = int(df.get("position_exec", pd.Series(0.0, index=df.index)).iloc[i] or 0)
            vol_20     = df.get("crack_vol_20", pd.Series(np.nan, index=df.index)).iloc[i]

            # ------------------------------------------------------------
            # 1. Mark-to-market open position
            # ------------------------------------------------------------
            if position != 0 and i > 0:
                prev_crack = float(df["crack"].iloc[i - 1])
                daily_pnl  = float(position) * (crack - prev_crack) * float(pos_size_bbl)
            else:
                daily_pnl = 0.0

            nav += daily_pnl

            # ------------------------------------------------------------
            # 2. Roll cost: charged when a month boundary is crossed
            # ------------------------------------------------------------
            roll_cost = 0.0
            if position != 0 and i > 0:
                prev_date = df.index[i - 1]
                if date.month != prev_date.month:
                    roll_cost = cfg.ROLL_COST_PER_BBL * float(pos_size_bbl)
                    nav      -= roll_cost

            # ------------------------------------------------------------
            # 3. Execute position changes (based on lagged signal)
            # ------------------------------------------------------------
            action     = "hold"
            exit_cost  = 0.0
            entry_cost_new = 0.0

            if target_pos != position:

                # ---- Close existing position ----
                if position != 0:
                    exit_cost = self._oneway_cost(pos_size_bbl)
                    nav      -= exit_cost

                    gross_pnl = float(position) * (crack - entry_spread) * float(pos_size_bbl)
                    net_pnl   = gross_pnl - entry_cost_ - exit_cost

                    hold_days = int((date - entry_date).days) if entry_date is not None else 0

                    trade_rows.append({
                        "entry_date":    entry_date,
                        "exit_date":     date,
                        "direction":     int(position),
                        "entry_spread":  round(entry_spread, 4),
                        "exit_spread":   round(crack, 4),
                        "spread_chg":    round(crack - entry_spread, 4),
                        "pos_size_bbl":  int(pos_size_bbl),
                        "gross_pnl":     round(gross_pnl, 2),
                        "total_cost":    round(entry_cost_ + exit_cost, 2),
                        "net_pnl":       round(net_pnl, 2),
                        "hold_days":     hold_days,
                        "exit_reason":   self._exit_reason(z, position),
                    })

                    position     = 0
                    entry_spread = 0.0
                    entry_date   = None
                    pos_size_bbl = 0
                    entry_cost_  = 0.0
                    action       = "exit"

                # ---- Open new position ----
                if target_pos != 0:
                    pos_size_bbl = self._vol_target_size(vol_20, nav, cfg)
                    entry_cost_new = self._oneway_cost(pos_size_bbl)
                    nav           -= entry_cost_new

                    position     = target_pos
                    entry_spread = crack
                    entry_date   = date
                    entry_cost_  = entry_cost_new
                    action       = "enter" if action != "exit" else "flip"

            # ------------------------------------------------------------
            # 4. Snapshot
            # ------------------------------------------------------------
            equity_rows.append({
                "date":         date,
                "nav":          round(nav, 2),
                "daily_pnl":    round(daily_pnl, 2),
                "position":     int(position),
                "pos_size_bbl": int(pos_size_bbl),
                "roll_cost":    round(roll_cost, 2),
                "action":       action,
                "crack":        round(crack, 4),
                "z_score":      round(z, 4),
            })

        # ---- Force-close any open position at end of backtest ----
        if position != 0:
            last_crack = float(df["crack"].iloc[-1])
            exit_cost  = self._oneway_cost(pos_size_bbl)
            nav       -= exit_cost
            gross_pnl  = float(position) * (last_crack - entry_spread) * float(pos_size_bbl)
            net_pnl    = gross_pnl - entry_cost_ - exit_cost
            trade_rows.append({
                "entry_date":    entry_date,
                "exit_date":     df.index[-1],
                "direction":     int(position),
                "entry_spread":  round(entry_spread, 4),
                "exit_spread":   round(last_crack, 4),
                "spread_chg":    round(last_crack - entry_spread, 4),
                "pos_size_bbl":  int(pos_size_bbl),
                "gross_pnl":     round(gross_pnl, 2),
                "total_cost":    round(entry_cost_ + exit_cost, 2),
                "net_pnl":       round(net_pnl, 2),
                "hold_days":     int((df.index[-1] - entry_date).days) if entry_date else 0,
                "exit_reason":   "end_of_backtest",
            })

        equity = pd.DataFrame(equity_rows).set_index("date")
        trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["entry_date", "exit_date", "direction", "entry_spread",
                     "exit_spread", "spread_chg", "pos_size_bbl",
                     "gross_pnl", "total_cost", "net_pnl", "hold_days", "exit_reason"]
        )

        total_ret = (nav - cfg.INITIAL_NAV) / cfg.INITIAL_NAV * 100
        logger.info(
            f"  Backtest: {df.index[0].date()} → {df.index[-1].date()} | "
            f"NAV: ${cfg.INITIAL_NAV:,.0f} → ${nav:,.0f} ({total_ret:+.2f}%) | "
            f"Trades: {len(trades)}"
        )

        return equity, trades

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _oneway_cost(self, pos_size_bbl: int) -> float:
        """
        One-way transaction cost = slippage + commission.

        Paid at both entry and exit (round-trip = 2 × one-way).
        """
        cfg         = self.cfg
        n_contracts = max(1, int(pos_size_bbl) // cfg.CONTRACT_SIZE_BBL)
        slippage    = cfg.SLIPPAGE_PER_BBL    * float(pos_size_bbl)
        commission  = cfg.COMMISSION_PER_CONTRACT * n_contracts
        return slippage + commission

    def _vol_target_size(self, vol_20: float, nav: float, cfg: Config) -> int:
        """
        Compute volatility-targeted position size in barrels.

        Target: 1-day P&L std = VOL_TARGET_DAILY × NAV
        N_bbl = (VOL_TARGET_DAILY × NAV) / σ_crack

        Rounded DOWN to nearest contract (1,000 bbl).
        Bounded below by 1 contract.
        """
        if pd.isna(vol_20) or vol_20 <= 0:
            return cfg.CONTRACT_SIZE_BBL   # fallback: 1 contract

        raw_bbl = int((cfg.VOL_TARGET_DAILY * nav) / float(vol_20))
        # Round down to integer number of contracts
        n_contracts  = max(1, raw_bbl // cfg.CONTRACT_SIZE_BBL)
        return n_contracts * cfg.CONTRACT_SIZE_BBL

    def _exit_reason(self, z: float, position: int) -> str:
        """Classify exit for trade attribution analysis."""
        cfg = self.cfg
        if abs(z) > cfg.STOP_THRESHOLD:
            return "stop_loss"
        if position == +1 and z > -cfg.EXIT_THRESHOLD:
            return "profit_target"
        if position == -1 and z < +cfg.EXIT_THRESHOLD:
            return "profit_target"
        return "signal_reversal"
