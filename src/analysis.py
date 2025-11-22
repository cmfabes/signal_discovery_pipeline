"""Statistical analysis functions for signal discovery.

This module will contain functions to compute pairwise correlations,
lead/lag relationships, and bootstrap confidence intervals.  These tools
help identify meaningful signals and quantify their statistical
significance and effect sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Dict, Any, List, Optional
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from numpy.random import default_rng
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, coint


def compute_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute correlation and p-value between two series.

    - method: 'pearson' or 'spearman'.
    Returns (coefficient, p_value).
    """
    a = series_a.astype(float)
    b = series_b.astype(float)
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan, np.nan
    if method == "spearman":
        coef, p = spearmanr(a, b, nan_policy="omit")
        return float(coef), float(p)
    elif method == "pearson":
        coef, p = pearsonr(a, b)
        return float(coef), float(p)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")


def lagged_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    lags: Iterable[int],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlations and p-values at different lags.

    Positive lags shift `series_b` into the future relative to `series_a`.

    Returns a DataFrame with index=lag and columns=['coef','p_value'].
    """
    rows: List[Dict[str, Any]] = []
    for lag in lags:
        shifted_b = series_b.shift(lag)
        coef, p = compute_correlation(series_a, shifted_b, method=method)
        rows.append({"lag": lag, "coef": coef, "p_value": p})
    out = pd.DataFrame(rows).set_index("lag").sort_index()
    return out


def bootstrap_ci(
    data: np.ndarray,
    stat_func,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        data: 1‑D array of data points.
        stat_func: Function that computes the statistic on the data.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (e.g. 0.05 for a 95% CI).

    Returns:
        Lower and upper bounds of the confidence interval.
    """
    n = len(data)
    stats = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        stats.append(stat_func(sample))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper


def fdr_adjust(p_values: pd.Series, alpha: float = 0.05, method: str = "fdr_bh") -> pd.DataFrame:
    """Apply multiple-testing correction to a vector of p-values.

    Returns DataFrame with columns ['p_adj','rejected'] using statsmodels.multipletests.
    """
    # Replace NaNs with 1.0 for conservative adjustment
    p = p_values.fillna(1.0).values
    rejected, p_adj, _, _ = multipletests(p, alpha=alpha, method=method)
    return pd.DataFrame({"p_adj": p_adj, "rejected": rejected}, index=p_values.index)


def rolling_stability(
    series_a: pd.Series,
    series_b: pd.Series,
    lead_lag: int,
    window: int = 90,
    step: int = 14,
    method: str = "spearman",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Estimate stability: share of windows where correlation is significant.

    Returns dict with 'stability_share' and 'n_windows'.
    """
    b_shift = series_b.shift(lead_lag)
    n = len(series_a)
    idx = series_a.index
    significant = 0
    total = 0
    start = 0
    while start + window <= n:
        sl = slice(idx[start], idx[start + window - 1])
        coef, p = compute_correlation(series_a.loc[sl], b_shift.loc[sl], method=method)
        if not np.isnan(p) and p < alpha:
            significant += 1
        total += 1
        start += step
    share = (significant / total) if total > 0 else np.nan
    return {"stability_share": share, "n_windows": total}


def circular_shift_permutation_pvalue(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    lag: int = 0,
    method: str = "spearman",
    n_perm: int = 500,
    alternative: str = "two-sided",
    seed: Optional[int] = 0,
) -> float:
    """Permutation p-value via random circular shifts of series_b.

    Preserves autocorrelation while destroying alignment. Conservative and time-series friendly.
    alternative: 'two-sided' | 'greater' | 'less'
    """
    a = series_a.astype(float)
    b = series_b.astype(float).shift(lag)
    mask = a.notna() & b.notna()
    a = a[mask].to_numpy()
    b = b[mask].to_numpy()
    n = len(a)
    if n < 5:
        return float("nan")
    if method == "spearman":
        obs, _ = spearmanr(a, b)
    else:
        obs = np.corrcoef(a, b)[0, 1]
    rng = default_rng(seed)
    # precompute a single copy of b to rotate
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        k = rng.integers(1, n)  # exclude 0 shift
        b_shift = np.roll(b, k)
        if method == "spearman":
            stat, _ = spearmanr(a, b_shift)
        else:
            stat = np.corrcoef(a, b_shift)[0, 1]
        perm_stats[i] = stat
    if alternative == "two-sided":
        p = (np.sum(np.abs(perm_stats) >= abs(obs)) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(perm_stats <= obs) + 1) / (n_perm + 1)
    return float(p)


def moving_block_bootstrap_corr_ci(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    lag: int = 0,
    method: str = "spearman",
    block_len: int = 5,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> Tuple[float, float]:
    """Moving block bootstrap CI for correlation at a given lag.

    Draws blocks with wrap-around to retain local dependence.
    """
    a = series_a.astype(float)
    b = series_b.astype(float).shift(lag)
    mask = a.notna() & b.notna()
    a = a[mask].to_numpy()
    b = b[mask].to_numpy()
    n = len(a)
    if n < 5:
        return float("nan"), float("nan")
    rng = default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, n)
            block = [(start + j) % n for j in range(block_len)]
            idx.extend(block)
        idx = np.array(idx[:n])
        a_b = a[idx]
        b_b = b[idx]
        if method == "spearman":
            stat, _ = spearmanr(a_b, b_b)
        else:
            stat = np.corrcoef(a_b, b_b)[0, 1]
        stats[i] = stat
    lower = float(np.percentile(stats, 100 * (alpha / 2)))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lower, upper


def newey_west_pvalue(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    lag: int = 0,
    use_returns: bool = True,
    nw_lags: int = 5,
) -> Tuple[float, float]:
    """Newey-West robust regression of b on a (at given lag).

    Returns (beta, robust_pvalue). If use_returns, regress returns of b on returns of a.
    """
    a = series_a.astype(float)
    b = series_b.astype(float)
    if use_returns:
        a = a.pct_change()
        b = b.pct_change()
    b = b.shift(lag)
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    if len(a) < (nw_lags + 3):
        return float("nan"), float("nan")
    X = sm.add_constant(a.values)
    y = b.values
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    beta = float(res.params[1]) if len(res.params) > 1 else float("nan")
    pval = float(res.pvalues[1]) if len(res.pvalues) > 1 else float("nan")
    return beta, pval


def to_log_returns(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return np.log(s).diff()


def run_granger(
    series_a: pd.Series,
    series_b: pd.Series,
    maxlag: int = 10,
    use_returns: bool = True,
) -> Dict[str, Any]:
    """Run Granger causality tests of a -> b for lags 1..maxlag.

    Returns dict with 'min_p', 'best_lag', and 'per_lag' p-values for the ssr_ftest.
    """
    a = series_a.copy()
    b = series_b.copy()
    if use_returns:
        a = to_log_returns(a)
        b = to_log_returns(b)
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(df) < (maxlag + 5):
        return {"min_p": float("nan"), "best_lag": None, "per_lag": {}}
    try:
        res = grangercausalitytests(df[["b", "a"]], maxlag=maxlag, verbose=False)
        pvals = {lag: float(res[lag][0]["ssr_ftest"][1]) for lag in res.keys()}
        best_lag = min(pvals, key=pvals.get)
        return {"min_p": pvals[best_lag], "best_lag": int(best_lag), "per_lag": pvals}
    except Exception:
        return {"min_p": float("nan"), "best_lag": None, "per_lag": {}}


def run_cointegration(
    series_a: pd.Series,
    series_b: pd.Series,
) -> Dict[str, Any]:
    """Engle–Granger cointegration test on levels.

    Returns dict with 't_stat','p_value','crit'.
    """
    a = series_a.astype(float)
    b = series_b.astype(float)
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(df) < 20:
        return {"t_stat": float("nan"), "p_value": float("nan"), "crit": {}}
    try:
        t_stat, p_value, crit = coint(df["a"], df["b"])
        return {"t_stat": float(t_stat), "p_value": float(p_value), "crit": {k: float(v) for k, v in crit.items()}}
    except Exception:
        return {"t_stat": float("nan"), "p_value": float("nan"), "crit": {}}


def neutralize_against(series: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Return residuals of series after regressing on given factor(s).

    Aligns on index intersection, adds constant, and fits OLS. Returns residuals reindexed to
    original index (NaN where no alignment).
    """
    y = series.astype(float)
    X = factors.astype(float)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df.empty:
        return series * float("nan")
    Xc = sm.add_constant(df.drop(columns=["y"]).values)
    model = sm.OLS(df["y"].values, Xc)
    res = model.fit()
    resid = pd.Series(res.resid, index=df.index)
    # Reindex to original, leaving NaN where we didn't have data
    resid_full = resid.reindex(series.index)
    resid_full.name = series.name
    return resid_full
