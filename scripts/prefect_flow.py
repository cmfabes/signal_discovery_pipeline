#!/usr/bin/env python3
"""
Prefect flow to fetch AIS from Datalastic for configured ports and append daily
features to DuckDB. Writes a simple status JSON for the app.

Usage:
  export DATALASTIC_API_KEY=...
  python -m scripts.prefect_flow --geojson examples/ports.geojson 
    --ports "Shenzhen/Yantian,Ningbo-Zhoushan,Singapore" 
    --hours 24 --db data/ports.duckdb --table port_features

You can schedule via cron or Prefect work queues as needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
from datetime import datetime, timedelta, timezone
from typing import List

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prefect import flow, task
import pandas as pd
from src.ais import load_port_geofences, assign_port_to_points, derive_daily_port_features, polygon_bounds
from src.connectors.ais_datalastic import DatalasticClient
from src.duckdb_io import write_port_features_df


@task
def fetch_and_aggregate(geojson_path: str, ports: List[str], hours: int, api_key: str) -> pd.DataFrame:
    fences = load_port_geofences(geojson_path)
    if ports:
        fences = [f for f in fences if f.name in set(ports)]
    client = DatalasticClient(api_key=api_key)
    t_end = datetime.now(timezone.utc)
    t_start = t_end - timedelta(hours=hours)
    def _iso(dt: datetime) -> str:
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    parts = []
    for fence in fences:
        mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
        try:
            df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, _iso(t_start), _iso(t_end))
        except Exception:
            continue
        if df.empty:
            continue
        tag = assign_port_to_points(df, [fence])
        tag = tag[tag["port"].notna()]
        if tag.empty:
            continue
        feats = derive_daily_port_features(tag)
        parts.append(feats)
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=["Date","port"])  # empty


@task
def write_status(status_path: str, ports: List[str], rows_appended: int) -> None:
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ports": ports,
        "rows_appended": rows_appended,
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@flow(name="ais-dailyrun")
def run_flow(geojson_path: str, ports_csv: str, hours: int, db_path: str, table: str, api_key: str):
    ports = [p.strip() for p in ports_csv.split(",") if p.strip()] if ports_csv else []
    df = fetch_and_aggregate(geojson_path, ports, hours, api_key)
    rows = 0
    if not df.empty:
        rows = write_port_features_df(db_path, df, table=table, mode="append") - 0  # returns table size
    write_status(os.path.join(ROOT, "outputs", "flow_status.json"), ports or ["ALL"], int(len(df)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default="examples/ports.geojson")
    ap.add_argument("--ports", default="")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--db", default="data/ports.duckdb")
    ap.add_argument("--table", default="port_features")
    ap.add_argument("--api-key", default=os.environ.get("DATALASTIC_API_KEY"))
    args = ap.parse_args()
    if not args.api_key:
        print("DATALASTIC_API_KEY missing (use --api-key or env var)")
        sys.exit(1)
    run_flow(args.geojson, args.ports, args.hours, args.db, args.table, args.api_key)


if __name__ == "__main__":
    main()

