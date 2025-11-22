from __future__ import annotations

"""Sentinel-1 SAR fetch (beta) via sentinelsat.

Searches and downloads recent Sentinel-1 products intersecting a GeoJSON AOI.
You need Copernicus SciHub/Copernicus Data Space credentials (free).

Note: Network access may be blocked in some environments; this code is provided
for convenience and may need to be run on your machine with internet access.
"""

from typing import Optional
import os
import json
from datetime import datetime


def fetch_sentinel_s1(
    *,
    geojson_path: str,
    start_date: str,
    end_date: str,
    max_items: int = 2,
    out_dir: str = "data/sentinel",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> list[str]:
    try:
        from sentinelsat import SentinelAPI, geojson_to_wkt
    except Exception as exc:
        raise ImportError("sentinelsat is not installed. Run: pip install sentinelsat") from exc

    user = username or os.environ.get("SENTINEL_USER")
    pwd = password or os.environ.get("SENTINEL_PASS")
    if not user or not pwd:
        raise ValueError("Missing Sentinel credentials (use args or SENTINEL_USER/SENTINEL_PASS env vars)")

    # New Copernicus Data Space URL (adjust if needed)
    api = SentinelAPI(user, pwd, "https://apihub.copernicus.eu/apihub")

    with open(geojson_path, "r", encoding="utf-8") as fh:
        geom = json.load(fh)
    footprint = geojson_to_wkt(geom)

    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname="Sentinel-1",
        producttype="GRD",
        limit=max_items,
        order_by="-ingestiondate",
    )
    if not products:
        return []
    os.makedirs(out_dir, exist_ok=True)
    downloaded: list[str] = []
    for uuid, prod in list(products.items())[:max_items]:
        save = api.download(uuid, directory_path=out_dir)
        if save and save.get("path"):
            downloaded.append(save["path"])
    return downloaded

