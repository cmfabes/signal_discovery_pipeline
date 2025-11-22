from __future__ import annotations

from typing import Optional
import os
import pandas as pd


def _connect(db_path: str):
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise ImportError("duckdb is not installed. Run: pip install duckdb") from exc
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(database=db_path, read_only=False)


def ensure_port_features_table(db_path: str, table: str = "port_features") -> None:
    con = _connect(db_path)
    try:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                Date DATE,
                port VARCHAR,
                anchored_count BIGINT,
                arrivals_count BIGINT,
                departures_count BIGINT,
                avg_speed_knots DOUBLE,
                distinct_vessels BIGINT,
                anchored_vessel_hours DOUBLE
            );
            """
        )
    finally:
        con.close()


def write_port_features_df(
    db_path: str,
    df: pd.DataFrame,
    table: str = "port_features",
    mode: str = "append",
) -> int:
    if "Date" not in df.columns or "port" not in df.columns:
        raise ValueError("DataFrame must include 'Date' and 'port' columns")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    con = _connect(db_path)
    try:
        ensure_port_features_table(db_path, table)
        if mode == "replace":
            con.execute(f"DELETE FROM {table}")
        con.register("tmp_df", df)
        con.execute(f"INSERT INTO {table} SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return int(cnt)
    finally:
        con.close()


def read_port_features(
    db_path: str,
    table: str = "port_features",
    ports: Optional[list[str]] = None,
) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        if ports:
            qmarks = ",".join(["?"] * len(ports))
            df = con.execute(f"SELECT * FROM {table} WHERE port IN ({qmarks}) ORDER BY Date", ports).fetchdf()
        else:
            df = con.execute(f"SELECT * FROM {table} ORDER BY Date").fetchdf()
    finally:
        con.close()
    df["Date"] = pd.to_datetime(df["Date"])  
    return df

