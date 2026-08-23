"""
src/data_pipeline.py — Data Acquisition, Validation, and Unit Conversion.

Responsibility:
    Download raw CME front-month futures settlements from Yahoo Finance,
    run quality checks, convert units to $/barrel, align on common
    trading calendar, and persist clean data to disk.

Unit Economics:
    WTI (CL=F)   : Yahoo Finance → $/barrel  (no conversion needed)
    RBOB (RB=F)  : Yahoo Finance → $/gallon  → ×42 → $/barrel
    HO (HO=F)    : Yahoo Finance → $/gallon  → ×42 → $/barrel

Note on continuous contracts:
    Yahoo Finance's =F tickers are rolling front-month proxies. They
    roll to the next contract as expiry approaches. This introduces small
    roll gaps (typically ≤$0.10/bbl) that are visible as overnight jumps
    in the price series. We document these but do NOT attempt to back-adjust
    prices; the strategy operates on the observable quoted spread, which is
    what a trading desk would use.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_futures_data(config: Config) -> dict:
    """
    Download daily Adjusted Close prices for WTI, RBOB, and HO.

    Parameters
    ----------
    config : Config

    Returns
    -------
    dict[str, pd.Series]
        Keys: 'WTI', 'RBOB', 'HO'. Values: Close price Series indexed by date.
    """
    data = {}
    for name, ticker in config.TICKERS.items():
        logger.info(f"  Downloading {name} ({ticker}) ...")
        raw = yf.download(
            ticker,
            start=config.START_DATE,
            end=config.END_DATE,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for {name} ({ticker}). "
                "Check ticker symbol and internet connectivity."
            )

        # yfinance may return MultiIndex columns when downloading single tickers
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        close = raw["Close"].copy()
        close.name = name
        close.index = pd.to_datetime(close.index)
        data[name] = close

        logger.info(
            f"    {name}: {len(close)} trading days, "
            f"range [{close.min():.3f}, {close.max():.3f}]"
        )

    return data


def download_benchmarks(config: Config) -> pd.DataFrame:
    """
    Download SPY (S&P 500) and WTI buy-and-hold benchmark series.

    Returns
    -------
    pd.DataFrame
        Columns: 'SPY', 'WTI_BH'. Index: datetime.
    """
    logger.info("  Downloading benchmark series (SPY, WTI) ...")
    spy = yf.download(
        "SPY",
        start=config.START_DATE,
        end=config.END_DATE,
        progress=False,
        auto_adjust=True,
    )
    wti = yf.download(
        config.TICKERS["WTI"],
        start=config.START_DATE,
        end=config.END_DATE,
        progress=False,
        auto_adjust=True,
    )

    # Handle MultiIndex columns
    for df_ in [spy, wti]:
        if isinstance(df_.columns, pd.MultiIndex):
            df_.columns = df_.columns.droplevel(1)

    bm = pd.DataFrame(
        {
            "SPY":    spy["Close"],
            "WTI_BH": wti["Close"],
        }
    )
    bm.index = pd.to_datetime(bm.index)
    bm = bm.dropna(how="any")
    logger.info(f"    Benchmark: {len(bm)} rows, {bm.index[0].date()} → {bm.index[-1].date()}")
    return bm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_raw_data(data: dict) -> None:
    """
    Run sanity checks on raw downloaded price series.

    Checks
    ------
    1. No instrument is entirely NaN.
    2. No instrument has > 5 consecutive NaN values.
    3. Raw price ranges are physically plausible.

    Raises
    ------
    ValueError
        On critical data quality failures.

    Warns
    -----
    On non-critical anomalies (logged at WARNING level).
    """
    # Expected price ranges BEFORE unit conversion
    price_ranges = {
        "WTI":  (-50.0, 200.0),  # $/bbl — allow negative (COVID Apr 2020 went to -$37)
        "RBOB": (0.40,    6.0),  # $/gallon
        "HO":   (0.40,    6.0),  # $/gallon
    }

    for name, series in data.items():
        if series.isna().all():
            raise ValueError(f"All prices are NaN for {name}.")

        # Check consecutive NaN
        nan_mask   = series.isna().astype(int)
        run_lengths = nan_mask.groupby((nan_mask != nan_mask.shift()).cumsum()).sum()
        max_consec  = run_lengths.max()
        if max_consec > 5:
            raise ValueError(
                f"{name}: Found {max_consec} consecutive NaN days. "
                "Data quality insufficient for strategy."
            )

        # Range check
        non_null = series.dropna()
        lo, hi   = price_ranges[name]
        if non_null.min() < lo or non_null.max() > hi:
            logger.warning(
                f"  {name}: Raw price range [{non_null.min():.4f}, {non_null.max():.4f}] "
                f"outside expected [{lo}, {hi}]. Proceeding with caution."
            )

    # Check calendar alignment
    date_sets = [set(s.dropna().index) for s in data.values()]
    common    = date_sets[0].intersection(*date_sets[1:])
    total     = date_sets[0].union(*date_sets[1:])
    pct_common = len(common) / len(total) * 100
    logger.info(f"  Trading calendar alignment: {pct_common:.1f}% dates shared across all instruments.")
    if pct_common < 90:
        logger.warning(f"  Low date alignment ({pct_common:.1f}%) — check for stale or missing data.")


# ---------------------------------------------------------------------------
# Unit Conversion
# ---------------------------------------------------------------------------

def convert_to_per_barrel(data: dict, config: Config) -> pd.DataFrame:
    """
    Convert RBOB and HO from $/gallon to $/barrel and build aligned panel.

    Conversion:   $/gallon × 42 gallons/barrel = $/barrel

    Alignment:
        1. Take the intersection of all three trading calendars.
        2. Forward-fill at most 1 day to handle settlement lags.
        3. Drop remaining NaN rows.

    Post-conversion validation:
        All $/bbl prices are checked against physically plausible ranges.

    Returns
    -------
    pd.DataFrame
        Columns: 'WTI', 'RBOB', 'HO' — all in $/barrel.
    """
    g = config.GALLONS_PER_BARREL

    wti  = data["WTI"].copy()
    rbob = data["RBOB"].copy() * g   # $/gallon → $/bbl
    ho   = data["HO"].copy()   * g   # $/gallon → $/bbl

    panel = pd.DataFrame({"WTI": wti, "RBOB": rbob, "HO": ho})
    panel.sort_index(inplace=True)

    # Forward-fill at most 1 day (handles occasional settlement reporting lags)
    panel = panel.ffill(limit=1)
    panel = panel.dropna(how="any")

    # Post-conversion range sanity check ($/bbl)
    post_ranges = {
        "WTI":  (10.0,  250.0),
        "RBOB": (20.0,  400.0),   # 42× $/gallon range
        "HO":   (20.0,  400.0),
    }
    for col, (lo, hi) in post_ranges.items():
        out_of_range = (~panel[col].between(lo, hi)).sum()
        if out_of_range > 0:
            logger.warning(
                f"  {col}: {out_of_range} out-of-range values after conversion "
                f"(expected [{lo}, {hi}] $/bbl). Check for unit errors."
            )

    logger.info(
        f"  Aligned panel: {len(panel)} rows, "
        f"{panel.index[0].date()} to {panel.index[-1].date()}"
    )
    logger.info(
        f"  WTI  $/bbl: [{panel['WTI'].min():.2f}, {panel['WTI'].max():.2f}]"
    )
    logger.info(
        f"  RBOB $/bbl: [{panel['RBOB'].min():.2f}, {panel['RBOB'].max():.2f}]"
    )
    logger.info(
        f"  HO   $/bbl: [{panel['HO'].min():.2f}, {panel['HO'].max():.2f}]"
    )

    return panel


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw_data(data: dict, config: Config) -> None:
    """Save each raw series with a download timestamp for reproducibility."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(config.DATA_DIR) / "raw"
    for name, series in data.items():
        path = raw_dir / f"{name}_{ts}.csv"
        series.to_csv(path, header=True)
    logger.info(f"  Raw data saved to {raw_dir} (timestamp: {ts})")


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data pipeline end-to-end.

    Steps
    -----
    1. Download raw futures data from Yahoo Finance.
    2. Validate data quality.
    3. Convert to $/barrel and align on common calendar.
    4. Download benchmark data (SPY, WTI buy-and-hold).
    5. Persist to disk.

    Returns
    -------
    panel : pd.DataFrame
        Aligned $/bbl prices: columns WTI, RBOB, HO.
    benchmark : pd.DataFrame
        Benchmark series: columns SPY, WTI_BH.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PIPELINE")
    logger.info("=" * 60)

    # Download
    raw_data  = download_futures_data(config)
    benchmark = download_benchmarks(config)

    # Validate
    logger.info("  Validating raw data ...")
    validate_raw_data(raw_data)
    logger.info("  Validation passed.")

    # Save raw
    save_raw_data(raw_data, config)

    # Convert units
    logger.info("  Converting units to $/barrel ...")
    panel = convert_to_per_barrel(raw_data, config)

    # Persist processed data
    proc_dir = Path(config.DATA_DIR) / "processed"
    panel.to_csv(proc_dir / "panel.csv")
    benchmark.to_csv(proc_dir / "benchmark.csv")
    logger.info(f"  Processed panel saved -> {proc_dir / 'panel.csv'}")
    logger.info(f"  Benchmark saved       -> {proc_dir / 'benchmark.csv'}")

    return panel, benchmark
