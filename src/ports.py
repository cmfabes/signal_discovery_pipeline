from __future__ import annotations

from typing import List

import pandas as pd


def load_port_features_csv(
    file_path: str,
    *,
    date_col: str = "Date",
    port_col: str = "port",
) -> pd.DataFrame:
    """Load a long-format CSV with columns [Date, port, <features>].

    Returns a DataFrame indexed by Date with a simple wide layout for a single port
    once filtered; upstream UI should filter by port and select feature columns.
    """
    df = pd.read_csv(file_path)
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in {file_path}")
    if port_col not in df.columns:
        raise ValueError(f"'{port_col}' column not found in {file_path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    return df


def filter_port(df: pd.DataFrame, port_name: str, date_col: str = "Date", port_col: str = "port") -> pd.DataFrame:
    out = df[df[port_col] == port_name].copy()
    out = out.drop(columns=[port_col])
    out = out.set_index(date_col)
    out.index.name = "Date"
    out = out.sort_index()
    return out


def create_lagged_features(df: pd.DataFrame, feature_cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lagged copies of selected feature columns.

    Returns a DataFrame with columns like '<feature>_lag_<k>'.
    """
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col not in df.columns:
            continue
        for k in lags:
            out[f"{col}_lag_{k}"] = df[col].shift(k)
    return out


def load_duckdb_features(db_path: str, table: str = "port_features") -> pd.DataFrame:
    """Load port features from a DuckDB database table.

    Requires the `duckdb` package to be installed in the environment.
    Expected table schema includes at least: Date (DATE/TIMESTAMP), port (TEXT), feature columns.
    """
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise ImportError("duckdb is not installed. Run: pip install duckdb") from exc

    con = duckdb.connect(database=db_path, read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
    finally:
        con.close()
    if "Date" not in df.columns:
        raise ValueError("DuckDB table must include a 'Date' column")
    if "port" not in df.columns:
        raise ValueError("DuckDB table must include a 'port' column")
    df["Date"] = pd.to_datetime(df["Date"])  
    df = df.sort_values("Date")
    return df

