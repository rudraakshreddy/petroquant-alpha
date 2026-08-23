"""
src/statistical_tests.py — Four-Test Statistical Framework for Mean Reversion.

Tests Conducted
---------------
1. Augmented Dickey-Fuller (ADF)
   H₀: Unit root present (random walk / non-stationary)
   H₁: Stationary (mean-reverting)
   → Reject H₀ at p < 0.05 to claim stationarity.

2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
   H₀: Series is stationary
   H₁: Unit root present
   → Fail to reject H₀ (p > 0.05) to confirm stationarity.
   ADF + KPSS agreement is the gold standard (Kwiatkowski et al., 1992).

3. Hurst Exponent (R/S Analysis)
   H < 0.5: Anti-persistent / mean-reverting
   H = 0.5: Geometric Brownian motion (random walk)
   H > 0.5: Persistent / trending
   → Quantifies the DEGREE of mean-reversion, not just yes/no.

4. Ornstein-Uhlenbeck Half-Life
   Model: ΔSₜ = α + β·Sₜ₋₁ + εₜ
   If β < 0 → mean-reverting speed λ = −β
   Half-life τ = ln(2) / λ  [trading days]
   → Calibrates the rolling window for the z-score signal.

References
----------
- Dickey, D.A. & Fuller, W.A. (1979). Distribution of the Estimators for
  Autoregressive Time Series with a Unit Root. JASA, 74(366), 427–431.
- Kwiatkowski, D. et al. (1992). Testing the null hypothesis of stationarity
  against the alternative of a unit root. JoE, 54(1-3), 159–178.
- Mandelbrot, B.B. & Wallis, J.R. (1969). Robustness of the rescaled range
  R/S in the measurement of noncyclic long-run statistical dependence.
  Water Resources Research, 5(5), 967–988.
- Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of Brownian motion.
  Physical Review, 36(5), 823.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Augmented Dickey-Fuller Test
# ---------------------------------------------------------------------------

def run_adf(series: pd.Series, lags: Optional[List] = None) -> dict:
    """
    Augmented Dickey-Fuller test for a unit root.

    Run with four lag specifications for robustness:
      - AIC-selected lags (data-driven, our primary result)
      - Fixed lags 1, 5, 10 (sensitivity check)

    Parameters
    ----------
    series : pd.Series
        Level of the crack spread (NOT first differences).
    lags : list, optional
        Lag specifications to test. Defaults to [1, 5, 10, 'AIC'].

    Returns
    -------
    dict
        'primary' : AIC-selected result (main reference)
        'by_lag'  : dict of results for each lag specification
    """
    if lags is None:
        lags = [1, 5, 10, "AIC"]

    clean = series.dropna()
    by_lag = {}

    for lag in lags:
        if lag == "AIC":
            result = adfuller(clean, maxlag=20, autolag="AIC", regression="c")
            key = "AIC"
        else:
            result = adfuller(clean, maxlag=int(lag), autolag=None, regression="c")
            key = f"lag_{lag}"

        stat, pval, n_lags, n_obs, crit = result[0], result[1], result[2], result[3], result[4]

        by_lag[key] = {
            "adf_stat":          round(float(stat), 4),
            "p_value":           round(float(pval), 6),
            "n_lags_used":       int(n_lags),
            "n_obs":             int(n_obs),
            "critical_1pct":     round(float(crit["1%"]),  4),
            "critical_5pct":     round(float(crit["5%"]),  4),
            "critical_10pct":    round(float(crit["10%"]), 4),
            "reject_null_5pct":  bool(pval < 0.05),
            "reject_null_1pct":  bool(pval < 0.01),
            "interpretation": (
                "Stationary (reject unit root @ 5%)"
                if pval < 0.05
                else "Cannot reject unit root @ 5%"
            ),
        }

    primary = by_lag["AIC"]
    logger.info("ADF Test (primary: AIC lags):")
    logger.info(f"  Statistic : {primary['adf_stat']:.4f}")
    logger.info(f"  p-value   : {primary['p_value']:.6f}")
    logger.info(f"  Lags used : {primary['n_lags_used']}")
    logger.info(f"  Critical  : 1%={primary['critical_1pct']}, 5%={primary['critical_5pct']}, 10%={primary['critical_10pct']}")
    logger.info(f"  Result    : {primary['interpretation']}")

    return {"primary": primary, "by_lag": by_lag}


# ---------------------------------------------------------------------------
# 2. KPSS Test
# ---------------------------------------------------------------------------

def run_kpss(series: pd.Series) -> dict:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test.

    Run with two regression specifications:
      - 'c'  (level stationarity) — our primary concern
      - 'ct' (trend stationarity) — to rule out deterministic trend

    Parameters
    ----------
    series : pd.Series
        Level of the crack spread.

    Returns
    -------
    dict
        Keys: 'level' and 'trend'. Each contains test statistics and interpretation.
    """
    import warnings
    clean = series.dropna()
    results = {}

    for regression in ["c", "ct"]:
        label = "level" if regression == "c" else "trend"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, n_lags, crit = kpss(clean, regression=regression, nlags="auto")

        results[label] = {
            "kpss_stat":           round(float(stat),  4),
            "p_value":             round(float(pval),  6),
            "n_lags":              int(n_lags),
            "critical_1pct":       round(float(crit["1%"]),  4),
            "critical_5pct":       round(float(crit["5%"]),  4),
            "critical_10pct":      round(float(crit["10%"]), 4),
            "fail_to_reject_5pct": bool(pval > 0.05),
            "interpretation": (
                "Confirms stationarity (fail to reject H₀ @ 5%)"
                if pval > 0.05
                else "Rejects stationarity @ 5% — possible unit root"
            ),
        }

    logger.info("KPSS Test:")
    for spec, r in results.items():
        logger.info(f"  [{spec}] stat={r['kpss_stat']:.4f}, p={r['p_value']:.6f} → {r['interpretation']}")

    return results


# ---------------------------------------------------------------------------
# 3. Hurst Exponent (R/S Analysis)
# ---------------------------------------------------------------------------

def compute_hurst(series: pd.Series, min_window: int = 10,
                  max_window: Optional[int] = None,
                  n_points: int = 25) -> dict:
    """
    Estimate the Hurst exponent via the Rescaled Range (R/S) method.

    Algorithm (Mandelbrot & Wallis, 1969):
      For each window size n:
        1. Divide series into M non-overlapping sub-periods of length n.
        2. For each sub-period i:
           a. Compute mean μᵢ.
           b. Compute cumulative deviation: Xₜ = Σ(sⱼ − μᵢ) for j=1..t
           c. Compute range R = max(Xₜ) − min(Xₜ)
           d. Compute std S = std(sⱼ)
           e. RS_i = R / S   (rescaled range)
        3. RS(n) = mean(RS_i) over all sub-periods.
      Plot log(n) vs log(RS(n)) and fit slope H by OLS.

    Parameters
    ----------
    series : pd.Series
        Price LEVEL of the crack spread (not differences).
    min_window : int
        Minimum sub-period length. Default 10.
    max_window : int, optional
        Maximum sub-period length. Defaults to N // 4.
    n_points : int
        Number of log-spaced window sizes to test.

    Returns
    -------
    dict
        'H', 'r_squared', 'p_value', 'std_err',
        'log_n', 'log_rs', 'interpretation'
    """
    arr = series.dropna().values.astype(float)
    N   = len(arr)

    if max_window is None:
        max_window = N // 4

    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_points).astype(int)
    )

    log_n_list  = []
    log_rs_list = []

    for n in windows:
        n_sub = N // n
        if n_sub < 2:
            continue

        rs_vals = []
        for i in range(n_sub):
            sub  = arr[i * n : (i + 1) * n]
            mu   = sub.mean()
            dev  = np.cumsum(sub - mu)
            R    = dev.max() - dev.min()
            S    = sub.std(ddof=1)
            if S > 1e-10 and np.isfinite(R / S):
                rs_vals.append(R / S)

        if len(rs_vals) >= 2:
            log_n_list.append(np.log(n))
            log_rs_list.append(np.log(np.mean(rs_vals)))

    log_n  = np.array(log_n_list)
    log_rs = np.array(log_rs_list)

    if len(log_n) < 3:
        logger.warning("Hurst: too few valid windows — returning H=0.5 (undefined).")
        return {"H": 0.5, "r_squared": 0.0, "interpretation": "Undefined"}

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
    H = float(slope)

    if H < 0.45:
        interp = f"H = {H:.4f} — strong mean-reversion (anti-persistent)"
    elif H < 0.5:
        interp = f"H = {H:.4f} — mild mean-reversion (anti-persistent)"
    elif H < 0.55:
        interp = f"H = {H:.4f} — approximately random walk"
    else:
        interp = f"H = {H:.4f} — trending / persistent"

    logger.info(f"Hurst Exponent:  H = {H:.4f}  (R² = {r_value**2:.4f}, p = {p_value:.4f})")
    logger.info(f"  {interp}")

    return {
        "H":             round(H, 4),
        "r_squared":     round(float(r_value ** 2), 4),
        "p_value":       round(float(p_value), 6),
        "std_err":       round(float(std_err), 4),
        "intercept":     round(float(intercept), 4),
        "log_n":         log_n.tolist(),
        "log_rs":        log_rs.tolist(),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# 4. Ornstein-Uhlenbeck Half-Life
# ---------------------------------------------------------------------------

def compute_ou_halflife(series: pd.Series) -> dict:
    """
    Estimate the Ornstein-Uhlenbeck mean-reversion half-life.

    Model
    -----
        ΔSₜ = α + β·Sₜ₋₁ + εₜ

    Estimation: OLS regression of ΔS on lagged level S.

    Parameters
    ----------
    β < 0 → mean-reverting process.
    Mean-reversion speed: λ = −β
    Half-life: τ = ln(2) / λ  (in trading days)

    The half-life is the expected time for the spread to move
    halfway from its current value back to its long-run mean.
    This quantity directly calibrates the optimal z-score window:
      Recommended window ≈ 1.0 × half-life to 1.5 × half-life.

    Parameters
    ----------
    series : pd.Series
        Price level of the crack spread (NOT differences).

    Returns
    -------
    dict
        'alpha', 'beta', 'lambda_', 'half_life_days', 't_stat_beta',
        'r_squared', 'interpretation', 'recommended_window'
    """
    s      = series.dropna()
    delta  = s.diff().dropna()
    lagged = s.shift(1).dropna()
    idx    = delta.index.intersection(lagged.index)
    delta  = delta.loc[idx]
    lagged = lagged.loc[idx]

    X    = np.column_stack([np.ones(len(lagged)), lagged.values])
    y    = delta.values

    # OLS via normal equations
    XtX  = X.T @ X
    Xty  = X.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])

    # Residuals and R²
    y_hat  = X @ beta_hat
    resid  = y - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard error of beta (for t-statistic)
    n, k   = len(y), 2
    sigma2 = ss_res / (n - k)
    cov    = sigma2 * np.linalg.inv(XtX)
    se_beta = float(np.sqrt(cov[1, 1]))
    t_stat  = beta / se_beta if se_beta > 0 else np.nan

    lambda_ = -beta
    if lambda_ > 0:
        half_life = float(np.log(2.0) / lambda_)
        rec_window = int(np.clip(round(half_life * 1.2), 10, 120))
        interp = (
            f"β = {beta:.6f} < 0 → mean-reverting. "
            f"Half-life ≈ {half_life:.1f} trading days. "
            f"Recommended z-score window: ~{rec_window} days."
        )
    else:
        half_life  = float("inf")
        rec_window = int(40)
        interp     = f"β = {beta:.6f} ≥ 0 → no mean reversion detected in OU framework."

    logger.info(f"OU Half-Life:   β = {beta:.6f}  (t = {t_stat:.3f})")
    logger.info(f"  λ (speed)    = {lambda_:.6f}")
    logger.info(f"  Half-life    = {half_life:.1f} days")
    logger.info(f"  {interp}")

    return {
        "alpha":              round(alpha, 6),
        "beta":               round(beta, 6),
        "lambda_":            round(lambda_, 6),
        "half_life_days":     round(half_life, 2),
        "t_stat_beta":        round(float(t_stat), 4) if np.isfinite(t_stat) else None,
        "r_squared":          round(r_sq, 4),
        "recommended_window": rec_window,
        "interpretation":     interp,
    }


# ---------------------------------------------------------------------------
# Combined Runner
# ---------------------------------------------------------------------------

def run_all_tests(df: pd.DataFrame, config: Config) -> dict:
    """
    Run all four statistical tests on the crack spread series and
    return a consolidated results dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'crack'.
    config : Config

    Returns
    -------
    dict with keys 'adf', 'kpss', 'hurst', 'ou', 'n_confirm',
    'overall_conclusion'
    """
    series = df["crack"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: STATISTICAL TESTS FOR MEAN REVERSION")
    logger.info("=" * 60)

    adf_res   = run_adf(series)
    kpss_res  = run_kpss(series)
    hurst_res = compute_hurst(series)
    ou_res    = compute_ou_halflife(series)

    # Tally confirmations
    c1 = adf_res["primary"]["reject_null_5pct"]       # ADF rejects unit root
    c2 = kpss_res["level"]["fail_to_reject_5pct"]     # KPSS confirms stationarity
    c3 = hurst_res["H"] < 0.5                         # Hurst < 0.5
    c4 = ou_res["half_life_days"] < 120               # Finite half-life

    n_confirm = sum([c1, c2, c3, c4])

    if n_confirm >= 3:
        conclusion = (
            f"STRONG evidence for mean reversion ({n_confirm}/4 tests confirm). "
            "Strategy statistical rationale is solid."
        )
    elif n_confirm == 2:
        conclusion = (
            f"MODERATE evidence for mean reversion ({n_confirm}/4 tests confirm). "
            "Strategy is defensible with appropriate caveats."
        )
    else:
        conclusion = (
            f"WEAK evidence for mean reversion ({n_confirm}/4 tests confirm). "
            "Review data quality and consider alternative regimes."
        )

    logger.info("")
    logger.info(f"OVERALL: {n_confirm}/4 tests confirm mean reversion.")
    logger.info(f"  {conclusion}")

    return {
        "adf":                adf_res,
        "kpss":               kpss_res,
        "hurst":              hurst_res,
        "ou":                 ou_res,
        "n_confirm":          n_confirm,
        "overall_conclusion": conclusion,
        "checks": {
            "adf_rejects_unit_root":      c1,
            "kpss_confirms_stationarity": c2,
            "hurst_below_half":           c3,
            "finite_ou_halflife":         c4,
        },
    }
