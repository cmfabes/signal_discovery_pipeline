#!/usr/bin/env python3
"""
Build a demo DuckDB with port features from the example CSV.

Usage:
  python -m scripts.build_demo_duckdb --db data/ports.duckdb --table port_features \
    --csv examples/port_features_template.csv --mode replace

Or directly:
  python scripts/build_demo_duckdb.py --db data/ports.duckdb --table port_features --csv examples/port_features_template.csv --mode replace
"""

import argparse
import os
import sys
import pathlib
import pandas as pd

# Ensure project root is on sys.path when run as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.duckdb_io import write_port_features_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/ports.duckdb")
    parser.add_argument("--table", default="port_features")
    parser.add_argument("--csv", default="examples/port_features_template.csv")
    parser.add_argument("--mode", choices=["append", "replace"], default="replace")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    df = pd.read_csv(args.csv)
    total = write_port_features_df(args.db, df, table=args.table, mode=args.mode)
    print(f"Wrote demo features to {args.db}:{args.table}. Row count now: {total}")


if __name__ == "__main__":
    main()
