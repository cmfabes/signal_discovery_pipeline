"""End‑to‑end pipeline orchestration.

This module stitches together the data ingestion, transformation, and
analysis steps defined in the other modules. It exposes a simple
run_pipeline function that accepts lists of operational file paths and
market tickers along with a date range and computes rolling statistics,
anomaly flags, and lead/lag correlations.
"""

from __future__ import annotations

import pandas as pd
from typing import Iterable, List, Dict, Any, Tuple

from .data_ingest import load_operational_data, load_market_data
from .transforms import rolling_zscore, anomaly_flag
from .analysis import (
    lagged_correlation,
    fdr_adjust,
    rolling_stability,
    circular_shift_permutation_pvalue,
    moving_block_bootstrap_corr_ci,
    newey_west_pvalue,
)


def run_pipeline(
    op_files: List[str],
    date_col: str,
    value_cols: List[str],
    tickers: List[str],
    start_date: str,
    end_date: str,
    lags: Iterable[int] = (0,),
    zscore_window: int = 14,
    anomaly_threshold: float = 3.0,
    returns_mode: bool = False,
    do_granger: bool = False,
    do_coint: bool = False,
    neutralize_benchmark_ticker: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Execute the signal discovery pipeline.

    Args:
        op_files: List of CSV file paths containing operational metrics.
        date_col: Name of the date column in the operational files.
        value_cols: List of column names to extract from each operational file.
        tickers: List of market tickers to download from Yahoo Finance.
        start_date: Start date (inclusive) for market data in 'YYYY-MM-DD' format.
        end_date: End date (exclusive) for market data.
        lags: Iterable of integer lags for correlation (in days).
        zscore_window: Window size for rolling z‑score calculation.
        anomaly_threshold: Threshold for anomaly flagging on z‑scores.

    Returns:
        A list of dictionaries summarizing correlations for each (metric, ticker) pair.
    """
    op_df = load_operational_data(op_files, date_col=date_col, value_cols=value_cols)
    market_df = load_market_data(tickers, start_date=start_date, end_date=end_date)

    # Join on the intersection of indices
    joined = op_df.join(market_df, how="inner")

    # Optionally convert to returns for correlation analysis
    if returns_mode:
        from .analysis import to_log_returns
        # Keep original joins for reporting/plots; build a returns dataframe for tests
        corr_frame = joined.apply(to_log_returns)
    else:
        corr_frame = joined

    # Optional neutralization vs benchmark returns
    if neutralize_benchmark_ticker:
        try:
            from .analysis import neutralize_against
            if neutralize_benchmark_ticker not in corr_frame.columns:
                # If benchmark not in market_df columns (e.g., typo), skip
                pass
            else:
                bench = corr_frame[[neutralize_benchmark_ticker]].rename(columns={neutralize_benchmark_ticker: "bench"})
                new_cols = {}
                for col in corr_frame.columns:
                    if col == neutralize_benchmark_ticker:
                        continue
                    new_cols[col] = neutralize_against(corr_frame[col], bench)
                corr_frame = pd.DataFrame(new_cols)
        except Exception:
            # Fail open: keep original corr_frame
            pass

    # Compute rolling z‑scores and anomaly flags for each operational metric
    for op_col in op_df.columns:
        zscores = rolling_zscore(joined[op_col], window=zscore_window)
        joined[f"{op_col}_z"] = zscores
        joined[f"{op_col}_anomaly"] = anomaly_flag(zscores, threshold=anomaly_threshold)

    # First pass: compute correlations and p-values for all pairs/lags
    # Accumulate p-values globally for FDR adjustment
    results: List[Dict[str, Any]] = []
    pvals_index: list[Tuple[str, str, int]] = []
    pvals_values: list[float] = []
    pair_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for op_col in op_df.columns:
        for ticker in tickers:
            df = lagged_correlation(corr_frame[op_col], corr_frame[ticker], lags=lags, method="spearman")
            pair_cache[(op_col, ticker)] = df
            for lag, row in df.iterrows():
                pvals_index.append((op_col, ticker, int(lag)))
                pvals_values.append(float(row.get("p_value", float("nan"))))

    # Apply FDR across all tests
    pvals_series = pd.Series(pvals_values, index=pd.MultiIndex.from_tuples(pvals_index, names=["metric","ticker","lag"]))
    fdr_df = fdr_adjust(pvals_series)

    # Build result records with FDR, peak, and stability metrics
    for (op_col, ticker), df in pair_cache.items():
        # attach adjusted p and significance flags
        # Align fdr rows for this pair
        sub_fdr = fdr_df.loc[(op_col, ticker)] if (op_col, ticker) in fdr_df.index else None
        if sub_fdr is not None:
            df = df.copy()
            # sub_fdr index is lag, columns p_adj/rejected
            df["p_adj"] = sub_fdr.loc[df.index, "p_adj"].values
            df["rejected"] = sub_fdr.loc[df.index, "rejected"].values

        # Pick peak by absolute coefficient among significant if any, else overall
        peak_row = None
        if "rejected" in df.columns and df["rejected"].any():
            peak_idx = df.loc[df["rejected"]].index[df.loc[df["rejected"]]["coef"].abs().argmax()]
            peak_row = df.loc[peak_idx]
        else:
            peak_idx = df["coef"].abs().idxmax()
            peak_row = df.loc[peak_idx]

        # Stability at peak lag
        stab = rolling_stability(
            corr_frame[op_col], corr_frame[ticker], lead_lag=int(peak_idx), window=90, step=14, method="spearman", alpha=0.05
        )

        # Additional significance: permutation p-value and block-bootstrap CI
        perm_p = circular_shift_permutation_pvalue(
            corr_frame[op_col], corr_frame[ticker], lag=int(peak_idx), method="spearman", n_perm=500, alternative="two-sided"
        )
        ci_lo, ci_hi = moving_block_bootstrap_corr_ci(
            corr_frame[op_col], corr_frame[ticker], lag=int(peak_idx), method="spearman", block_len=5, n_boot=500, alpha=0.05
        )
        # Newey-West regression p-value on returns
        beta_nw, p_nw = newey_west_pvalue(joined[op_col], joined[ticker], lag=int(peak_idx), use_returns=True, nw_lags=5)

        # Optional: Granger and cointegration
        extra: Dict[str, Any] = {}
        if do_granger:
            from .analysis import run_granger
            g = run_granger(joined[op_col], joined[ticker], maxlag=max(lags) if hasattr(lags, '__iter__') else 10, use_returns=True)
            extra["granger"] = g
        if do_coint:
            from .analysis import run_cointegration
            c = run_cointegration(joined[op_col], joined[ticker])
            extra["cointegration"] = c

        results.append(
            {
                "metric": op_col,
                "ticker": ticker,
                "correlations": df["coef"].to_dict(),
                "p_values": df["p_value"].to_dict(),
                "p_adj": (df["p_adj"].to_dict() if "p_adj" in df.columns else {}),
                "significant_lags": (
                    [int(l) for l, v in df["rejected"].items() if bool(v)] if "rejected" in df.columns else []
                ),
                "peak": {
                    "lag": int(peak_idx),
                    "coef": float(peak_row["coef"]),
                    "p_value": float(peak_row.get("p_value", float("nan"))),
                    "p_adj": float(peak_row.get("p_adj", float("nan"))) if not pd.isna(peak_row.get("p_adj", float("nan"))) else None,
                    "significant": bool(peak_row.get("rejected", False)) if "rejected" in df.columns else False,
                    "stability_share": float(stab.get("stability_share", float("nan"))),
                    "stability_windows": int(stab.get("n_windows", 0)),
                    "perm_p": float(perm_p) if perm_p == perm_p else None,
                    "ci_lower": float(ci_lo) if ci_lo == ci_lo else None,
                    "ci_upper": float(ci_hi) if ci_hi == ci_hi else None,
                    "beta_nw": float(beta_nw) if beta_nw == beta_nw else None,
                    "p_newey_west": float(p_nw) if p_nw == p_nw else None,
                    "perm_p": float(perm_p) if perm_p == perm_p else None,
                    "ci_lower": float(ci_lo) if ci_lo == ci_lo else None,
                    "ci_upper": float(ci_hi) if ci_hi == ci_hi else None,
                    "beta_nw": float(beta_nw) if beta_nw == beta_nw else None,
                    "p_newey_west": float(p_nw) if p_nw == p_nw else None,
                },
                **({"granger": extra.get("granger")} if "granger" in extra else {}),
                **({"cointegration": extra.get("cointegration")} if "cointegration" in extra else {}),
            }
        )

    return results


if __name__ == "__main__":
    # Example usage when running this module directly. Users can modify
    # these parameters or integrate this function into a Streamlit UI.
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Run the signal discovery pipeline")
    parser.add_argument(
        "--op-files", nargs="+", required=True, help="Paths to operational CSV files"
    )
    parser.add_argument(
        "--date-col", default="date", help="Name of the date column in operational files"
    )
    parser.add_argument(
        "--value-cols", nargs="+", required=True, help="Value columns to extract"
    )
    parser.add_argument(
        "--tickers", nargs="+", required=True, help="List of market tickers"
    )
    parser.add_argument(
        "--start-date", required=True, help="Start date for market data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date for market data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--lags", nargs="*", type=int, default=[0], help="List of integer lags"
    )
    args = parser.parse_args()

    results = run_pipeline(
        op_files=args.op_files,
        date_col=args.date_col,
        value_cols=args.value_cols,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lags=args.lags,
    )
    pprint.pprint(results)
