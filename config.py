"""
config.py — Single source of truth for all strategy parameters.

Every tuneable parameter lives here. Modifying values here propagates
through the entire pipeline without touching module code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------
    BASE_DIR:         str = str(Path(__file__).parent)
    DATA_DIR:         str = str(Path(__file__).parent / "data")
    RESULTS_DIR:      str = str(Path(__file__).parent / "results")
    FIGURES_DIR:      str = str(Path(__file__).parent / "results" / "figures")
    TABLES_DIR:       str = str(Path(__file__).parent / "results" / "tables")
    REPORT_DIR:       str = str(Path(__file__).parent / "report")
    DASHBOARD_DIR:    str = str(Path(__file__).parent / "dashboard")

    # -----------------------------------------------------------------------
    # Data Parameters
    # -----------------------------------------------------------------------
    TICKERS: dict = field(default_factory=lambda: {
        "WTI":  "CL=F",    # WTI Crude — already in $/bbl
        "RBOB": "RB=F",    # RBOB Gasoline — Yahoo Finance returns $/gallon
        "HO":   "HO=F",    # Heating Oil (diesel proxy) — Yahoo Finance $/gallon
    })

    START_DATE:          str   = "2019-01-01"   # 5 full calendar years
    END_DATE:            str   = "2024-12-31"   # includes COVID crash + 2022 energy shock
    GALLONS_PER_BARREL:  int   = 42             # CME standard conversion factor

    # -----------------------------------------------------------------------
    # Signal Parameters (defaults; overridden by sweep)
    # -----------------------------------------------------------------------
    ROLLING_WINDOW:      int   = 40     # z-score lookback (days)
    ENTRY_THRESHOLD:     float = 2.0    # |z| > 2.0 → enter position
    EXIT_THRESHOLD:      float = 0.5    # |z| < 0.5 → exit position
    STOP_THRESHOLD:      float = 4.0    # |z| > 4.0 → hard stop (tail protection)

    # -----------------------------------------------------------------------
    # Parameter Sweep Grid (walk-forward)
    # -----------------------------------------------------------------------
    SWEEP_WINDOWS:            List[int]   = field(default_factory=lambda: [20, 30, 40, 50, 60])
    SWEEP_ENTRY_THRESHOLDS:   List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    TRAIN_RATIO:              float       = 0.70   # 70% IS / 30% OOS split

    # -----------------------------------------------------------------------
    # Backtester Parameters
    # -----------------------------------------------------------------------
    INITIAL_NAV:              float = 1_000_000.0  # $1M starting capital
    VOL_TARGET_DAILY:         float = 0.01          # 1% daily NAV risk target
    CONTRACT_SIZE_BBL:        int   = 1_000          # 1 CME contract = 1,000 bbl

    # Transaction Costs (one-way, per trade)
    SLIPPAGE_PER_BBL:         float = 0.05    # $/bbl bid-ask slippage
    COMMISSION_PER_CONTRACT:  float = 2.50    # $/contract round-trip (IB-style)
    ROLL_COST_PER_BBL:        float = 0.02    # $/bbl monthly roll cost when position open

    # -----------------------------------------------------------------------
    # Risk Parameters
    # -----------------------------------------------------------------------
    RISK_FREE_RATE:           float = 0.05    # 5% annual (approx US Treasury 2019-2024 avg)

    # -----------------------------------------------------------------------
    # Figure Aesthetics
    # -----------------------------------------------------------------------
    FIGURE_DPI:               int   = 300
    FIGURE_STYLE:             str   = "seaborn-v0_8-whitegrid"
    FIGURE_PALETTE:           str   = "deep"

    def __post_init__(self):
        """Create all output directories on instantiation."""
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.FIGURES_DIR,
                  self.TABLES_DIR, self.REPORT_DIR, self.DASHBOARD_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)
        # Raw and processed sub-directories
        for sub in ["raw", "processed"]:
            Path(self.DATA_DIR, sub).mkdir(parents=True, exist_ok=True)


# Default singleton used throughout the pipeline
default_config = Config()
