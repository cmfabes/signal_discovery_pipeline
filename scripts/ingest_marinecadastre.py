#!/usr/bin/env python3
"""
Ingest historical AIS CSVs from MarineCadastre (US waters) and build daily port features.

Notes:
- Download monthly AIS CSVs from https://marinecadastre.gov/ais/ (US waters) and place in a folder.
- This script scans a directory for .csv/.csv.gz files with columns that include:
  - timestamp col (e.g., BaseDateTime)
  - MMSI, LAT, LON, SOG, NavigationalStatus (column names may vary)
  It will auto-detect common names and map to our schema.

Usage:
  python -m scripts.ingest_marinecadastre \
    --data-dir /path/to/ais_csvs \
    --geojson examples/ports.geojson \
    --ports "Los Angeles / Long Beach,New York / New Jersey" \
    --db data/ports.duckdb --table port_features
"""

import argparse
import glob
import os
import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ais import load_port_geofences, assign_port_to_points, derive_daily_port_features
from src.duckdb_io import write_port_features_df


COLMAPS = [
    {
        "timestamp": "BaseDateTime",
        "mmsi": "MMSI",
        "lat": "LAT",
        "lon": "LON",
        "speed_knots": "SOG",
        "status": "NavigationalStatus",
    },
    {
        "timestamp": "timestamp",
        "mmsi": "mmsi",
        "lat": "lat",
        "lon": "lon",
        "speed_knots": "speed_knots",
        "status": "status",
    },
]


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    for m in COLMAPS:
        if all(c in df.columns for c in m.values()):
            out = pd.DataFrame({k: df[v] for k, v in m.items()})
            return out
    raise ValueError("Could not map columns for AIS CSV (unexpected header names)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--geojson", default="examples/ports.geojson")
    ap.add_argument("--ports", default="")
    ap.add_argument("--db", default="data/ports.duckdb")
    ap.add_argument("--table", default="port_features")
    args = ap.parse_args()

    fences = load_port_geofences(args.geojson)
    if args.ports:
        names = {n.strip() for n in args.ports.split(",") if n.strip()}
        fences = [f for f in fences if f.name in names]
    if not fences:
        print("No ports found to process")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv*")))
    if not paths:
        print("No CSV files found in data-dir")
        sys.exit(1)

    total_rows = 0
    for p in paths:
        try:
            df_raw = pd.read_csv(p)
            df = map_columns(df_raw)
        except Exception as exc:
            print(f"Skip {p}: {exc}")
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp", "lat", "lon"]).reset_index(drop=True)
        if df.empty:
            continue
        # Assign ports and aggregate per port
        parts = []
        for fence in fences:
            tag = assign_port_to_points(df, [fence])
            tag = tag[tag["port"].notna()]
            if tag.empty:
                continue
            feats = derive_daily_port_features(tag)
            parts.append(feats)
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        total = write_port_features_df(args.db, out, table=args.table, mode="append")
        total_rows += len(out)
        print(f"{os.path.basename(p)} -> appended {len(out)} rows (table ~{total})")
    print(f"Ingestion done. Appended total rows: {total_rows}")


if __name__ == "__main__":
    main()

