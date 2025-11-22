#!/usr/bin/env python3
"""
Backfill AIS features from Datalastic for multiple past days and ports.

Usage:
  export DATALASTIC_API_KEY=...
  python -m scripts.backfill_datalastic --geojson examples/ports.geojson \
    --ports "Shenzhen/Yantian,Ningbo-Zhoushan,Singapore" --days 60 \
    --db data/ports.duckdb --table port_features
"""

import argparse
import os
import sys
import pathlib
from datetime import datetime, timedelta, timezone
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ais import load_port_geofences, assign_port_to_points, derive_daily_port_features, polygon_bounds
from src.connectors.ais_datalastic import DatalasticClient
from src.duckdb_io import write_port_features_df


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default="examples/ports.geojson")
    ap.add_argument("--ports", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--db", default="data/ports.duckdb")
    ap.add_argument("--table", default="port_features")
    ap.add_argument("--api-key", default=os.environ.get("DATALASTIC_API_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        print("Error: DATALASTIC_API_KEY not provided")
        sys.exit(1)

    fences = load_port_geofences(args.geojson)
    if args.ports:
        names = {n.strip() for n in args.ports.split(",") if n.strip()}
        fences = [f for f in fences if f.name in names]
    if not fences:
        print("No ports found to backfill")
        sys.exit(1)

    client = DatalasticClient(api_key=args.api_key)
    t_end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    all_rows = 0
    for d in range(args.days, 0, -1):
        day_start = t_end - timedelta(days=d)
        day_end = day_start + timedelta(days=1)
        day_count = 0
        for fence in fences:
            mnx, mny, mxx, mxy = polygon_bounds(fence.polygons)
            try:
                df = client.fetch_positions_bbox(mnx, mny, mxx, mxy, iso(day_start), iso(day_end))
            except Exception as exc:
                print(f"Fetch failed {fence.name} {day_start.date()}: {exc}")
                continue
            if df.empty:
                continue
            df_tag = assign_port_to_points(df, [fence])
            df_tag = df_tag[df_tag["port"].notna()]
            if df_tag.empty:
                continue
            feats = derive_daily_port_features(df_tag)
            if feats.empty:
                continue
            total = write_port_features_df(args.db, feats, table=args.table, mode="append")
            day_count += len(feats)
        all_rows += day_count
        print(f"{day_start.date()}: appended {day_count} rows (table size ~{total})")
    print(f"Backfill completed, appended total rows: {all_rows}")


if __name__ == "__main__":
    main()

