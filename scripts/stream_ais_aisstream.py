#!/usr/bin/env python3
from __future__ import annotations

"""Stream AIS from AISstream.io and append features to DuckDB.

Usage:
  export AISSTREAM_TOKEN=...  # or pass --token
  python -m scripts.stream_ais_aisstream \
    --geojson examples/ports.geojson \
    --db data/ports.duckdb --table port_features \
    --minutes 5
"""

import argparse
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.connectors.ais_aisstream import stream_and_append


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default="examples/ports.geojson")
    ap.add_argument("--db", default="data/ports.duckdb")
    ap.add_argument("--table", default="port_features")
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    total = stream_and_append(
        token=args.token,
        geojson_path=args.geojson,
        minutes=args.minutes,
        db_path=args.db,
        table=args.table,
    )
    print(f"Append complete. Table row count ~{total}")


if __name__ == "__main__":
    main()

