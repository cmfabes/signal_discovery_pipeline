"""Data ingestion utilities for the signal discovery pipeline.

This module centralizes robust I/O for operational features and market
data so the rest of the app stays simple. It includes helpers to:

- read CSV/Parquet tables with consistent column handling
- standardize/clean columns and coerce numeric types
- align/aggregate to daily frequency with configurable fill or aggregation
- load market data (Yahoo Finance) with a simple in‑memory cache
- assemble multiple operational files/globs into a single features table
"""

from __future__ import annotations

from datetime import timezone
from typing import Iterable, Optional
import glob
import logging
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGER: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        try:
            out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "logs"))
            os.makedirs(out_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(out_dir, "ingest.log"), encoding="utf-8")
            fh.setLevel(logging.INFO)
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # Fall back silently if logs dir not writable
            pass
    _LOGGER = logger
    return logger


# ---------------------------------------------------------------------------
# Column handling & generic readers
# ---------------------------------------------------------------------------


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower‑snake‑case column names stripped of spaces."""
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_").lower() for c in out.columns]
    return out


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric (errors='coerce')."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_csv(file_path: str, *, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with basic standardization."""
    df = pd.read_csv(file_path, parse_dates=parse_dates)
    return df


def read_table(file_path: str, *, date_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Read a CSV or Parquet and return standardized DataFrame.

    The date column is parsed to datetime; numeric columns coerced.
    """
    log = _get_logger()
    if file_path.lower().endswith((".parquet", ".pq")):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    df = standardize_columns(df)
    if date_col.lower() in df.columns and df[date_col.lower()].dtype.kind != "M":
        df[date_col.lower()] = pd.to_datetime(df[date_col.lower()], errors="coerce")
    df = coerce_numeric(df, [c.lower() for c in value_cols if c.lower() in df.columns])
    # Warn about missing columns
    missing = [c.lower() for c in value_cols if c.lower() not in df.columns]
    if missing:
        log.warning("Missing columns in %s: %s", file_path, ", ".join(missing))
    return df


def align_daily(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    *,
    method: str = "ffill",
    agg: str | None = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Align a DataFrame to daily frequency with configurable strategy.

    - method: 'ffill' (default) fills missing values forward; ignored if agg provided.
    - agg: if provided (e.g., 'sum'/'mean'/'last'), resamples per day with that agg.
    """
    log = _get_logger()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.set_index(date_col).sort_index()
    if start is None:
        start = d.index.min()
    if end is None:
        end = d.index.max()
    d = d.loc[start:end]
    if agg:
        if agg == "sum":
            out = d[value_cols].resample("D").sum()
        elif agg == "mean":
            out = d[value_cols].resample("D").mean()
        else:
            out = d[value_cols].resample("D").last()
    else:
        daily_index = pd.date_range(start=start, end=end, freq="D")
        out = d[value_cols].reindex(daily_index)
        if method == "ffill":
            out = out.ffill()
        elif method == "bfill":
            out = out.bfill()
        else:
            out = out.interpolate(limit_direction="both")
    out.index.name = "Date"
    log.info("Aligned to daily: rows=%d, cols=%d, method=%s, agg=%s", out.shape[0], out.shape[1], method, agg)
    return out
_MARKET_CACHE: dict[tuple[tuple[str, ...], str, str], pd.DataFrame] = {}


def load_market_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download market data for a list of tickers from Yahoo Finance.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf
    key = (tuple(sorted(tickers)), start_date, end_date)
    if key in _MARKET_CACHE:
        return _MARKET_CACHE[key].copy()
    log = _get_logger()
    data: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No market data returned for ticker {ticker}")
        # Prefer 'Adj Close', fall back to 'Close'
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        elif isinstance(df.columns, pd.MultiIndex):
            # Handle unexpected multi-index columns
            if "Adj Close" in df.columns.get_level_values(-1):
                series = df.xs("Adj Close", level=-1, axis=1)
            elif "Close" in df.columns.get_level_values(-1):
                series = df.xs("Close", level=-1, axis=1)
            else:
                raise KeyError("No 'Adj Close' or 'Close' column found in market data")
        else:
            raise KeyError("No 'Adj Close' or 'Close' column found in market data")
        # If we pulled a DataFrame slice, squeeze it down to a Series
        if hasattr(series, "ndim") and series.ndim > 1:
            series = series.squeeze("columns")
        # Name the Series so concat uses the ticker as the column name
        series.name = ticker
        data[ticker] = series
    market_df = pd.concat(data.values(), axis=1)
    market_df.index.name = "Date"
    _MARKET_CACHE[key] = market_df.copy()
    log.info("Loaded market data: tickers=%s, rows=%d", ",".join(tickers), len(market_df))
    return market_df




def load_operational_data(file_paths: list[str], date_col: str, value_cols: list[str]) -> pd.DataFrame:
    """
    Load and align multiple operational data files.

    Each CSV in `file_paths` must contain a date column and one or more
    value columns. The function reads each file, aligns it to daily
    frequency using `align_daily`, prefixes columns with the file name
    (to avoid collisions), and concatenates them horizontally.
    """
    log = _get_logger()
    aligned_dfs: list[pd.DataFrame] = []
    for path in file_paths:
        df = read_table(path, date_col=date_col, value_cols=value_cols)
        aligned = align_daily(df, date_col=date_col, value_cols=[c.lower() for c in value_cols])
        prefix = path.split("/")[-1].split(".")[0]
        aligned = aligned.add_prefix(f"{prefix}_")
        aligned_dfs.append(aligned)
    if not aligned_dfs:
        raise ValueError("No operational files provided")
    combined = pd.concat(aligned_dfs, axis=1)
    combined.index.name = "Date"
    log.info("Loaded operational features: files=%d, rows=%d, cols=%d", len(file_paths), combined.shape[0], combined.shape[1])
    return combined


def join_operational_market(op_df: pd.DataFrame, market_df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """Join operational and market panels on date index.

    Ensures the index name is 'Date' on output for consistency.
    """
    out = op_df.join(market_df, how=how)
    out.index.name = "Date"
    return out


def load_operational_glob(pattern: str, *, date_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Load multiple operational files matching a glob pattern.

    Example: load_operational_glob('data/ops/*.csv', date_col='date', value_cols=['value'])
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"No files matched pattern: {pattern}")
    return load_operational_data(files, date_col=date_col, value_cols=value_cols)


def load_intraday_data(tickers: list[str], interval: str = '15m') -> pd.DataFrame:
    """
    Load intraday market data for the given tickers.
    Returns most recent intraday data using a more reliable interval.
    
    Args:
        tickers: List of ticker symbols
        interval: Data interval ('1m','2m','5m','15m','30m','60m','90m')
                 Default is 15m as it's more reliably available
    
    Returns:
        DataFrame with OHLCV columns, or empty DataFrame if no data available
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    log = _get_logger()
    all_data = []
    
    # Use today and previous trading day to ensure we get some data
    end = datetime.now()
    start = end - timedelta(days=1)
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    for ticker in tickers:
        try:
            # Get intraday data
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False
            )
            
            if df.empty:
                log.warning(f"No intraday data returned for {ticker}")
                continue
                
            # Add ticker identifier
            df['ticker'] = ticker
            all_data.append(df)
            
        except Exception as e:
            log.error(f"Error fetching intraday data for {ticker}: {str(e)}")
            continue
    
    if not all_data:
        log.warning("No intraday data available, returning empty DataFrame")
        # Return empty DataFrame with expected columns instead of raising error
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'ticker'])
        
    # Combine all tickers
    result = pd.concat(all_data)
    
    # Ensure basic OHLCV columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    for col in required_cols:
        if col not in result.columns:
            result[col] = None
    
    log.info(f"Loaded intraday data for {len([d for d in all_data if not d.empty])} tickers at {interval} interval")
    return result
