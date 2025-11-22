"""Time‑series transformation functions.

This module provides functions for computing rolling statistics, percentage
changes, z‑scores, and anomaly flags on time‑series data.  These
transformations help normalize the data and identify unusual patterns.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling mean of a series."""
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling standard deviation of a series."""
    return series.rolling(window=window, min_periods=1).std(ddof=0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute the rolling z‑score of a series.

    The z‑score is calculated as (series - rolling_mean) / rolling_std.
    A small epsilon is added to the denominator to avoid division by zero.
    """
    mean = rolling_mean(series, window)
    std = rolling_std(series, window)
    epsilon = 1e-8
    return (series - mean) / (std + epsilon)


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Compute percentage change over given periods."""
    return series.pct_change(periods=periods)


def anomaly_flag(zscores: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Generate a boolean flag for anomalies based on z‑score threshold.

    Observations where the absolute z‑score exceeds the threshold are
    marked as True.
    """
    return zscores.abs() > threshold


def detect_momentum_shift(series: pd.Series,
                        window: int = 14,
                        threshold: float = 2.0) -> pd.DataFrame:
    """
    Detect significant momentum shifts in a time series.
    
    Args:
        series: Input time series
        window: Rolling window for momentum calculation
        threshold: Z-score threshold for significance
    
    Returns:
        DataFrame with columns:
        - direction: 1 for upward shift, -1 for downward
        - magnitude: Size of the shift in original units
        - zscore: Statistical significance (z-score)
    """
    # Compute momentum indicators
    ma = rolling_mean(series, window)
    std = rolling_std(series, window)
    zscore = rolling_zscore(series, window)
    
    # Detect shifts
    shifts = pd.DataFrame(index=series.index)
    
    # Momentum direction
    shifts["direction"] = np.where(zscore > threshold, 1,
                                 np.where(zscore < -threshold, -1, 0))
    
    # Magnitude of shift
    shifts["magnitude"] = series - ma
    
    # Statistical significance
    shifts["zscore"] = zscore
    
    # Filter only significant shifts
    return shifts[shifts["direction"] != 0]