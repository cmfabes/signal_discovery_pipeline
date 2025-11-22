#!/usr/bin/env python3
"""
Fetch recent AIS positions from Datalastic for configured ports and write daily
features to DuckDB. Uses GeoJSON geofences for ports.

Examples:
  export DATALASTIC_API_KEY=... 
  python -m scripts.fetch_ais_datalastic \
    --geojson examples/ports.geojson \
    --ports "Los Angeles / Long Beach,New York / New Jersey" \
    --hours 24 --db data/ports.duckdb --table port_features
"""

import argparse
import os
import sys
import pathlib
from datetime import datetime, timedelta, timezone
import pandas as pd

# Ensure project root on path
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
    ap.add_argument("--ports", default="", help="Comma-separated port names (empty=all from GeoJSON)")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--db", default="data/ports.duckdb")
    ap.add_argument("--table", default="port_features")
    ap.add_argument("--api-key", default=os.environ.get("DATALASTIC_API_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        print("Error: DATALASTIC_API_KEY not provided (use --api-key or env var)")
        sys.exit(1)

    fences = load_port_geofences(args.geojson)
    if args.ports:
        names = {n.strip() for n in args.ports.split(",") if n.strip()}
        fences = [f for f in fences if f.name in names]
    if not fences:
        print("No ports selected or found in GeoJSON")
        sys.exit(1)

    client = DatalasticClient(api_key=args.api_key)
    t_end = datetime.now(timezone.utc)
    t_start = t_end - timedelta(hours=args.hours)
    all_features = []

    for fence in fences:
        min_lon, min_lat, max_lon, max_lat = polygon_bounds(fence.polygons)
        try:
            df = client.fetch_positions_bbox(min_lon, min_lat, max_lon, max_lat, iso(t_start), iso(t_end))
        except Exception as exc:
            print(f"Fetch failed for {fence.name}: {exc}")
            continue
        if df.empty:
            print(f"No positions for {fence.name} in window")
            continue
        # Assign port by precise polygon test
        df_tagged = assign_port_to_points(df, [fence])
        df_tagged = df_tagged[df_tagged["port"].notna()]
        if df_tagged.empty:
            print(f"No points inside {fence.name} polygons (after bbox fetch)")
            continue
        feats = derive_daily_port_features(df_tagged)
        all_features.append(feats)

    if not all_features:
        print("No features to write")
        return
    out = pd.concat(all_features, ignore_index=True)
    total = write_port_features_df(args.db, out, table=args.table, mode="append")
    print(f"Wrote {len(out)} rows; table now has {total} rows")


if __name__ == "__main__":
    main()

