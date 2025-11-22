from __future__ import annotations

"""Datalastic AIS connector (stubbed for live fetching).

Usage:
  client = DatalasticClient(api_key=os.environ["DATALASTIC_API_KEY"]) 
  df = client.fetch_positions_bbox(min_lon, min_lat, max_lon, max_lat,
                                   start_iso, end_iso)

Notes:
  - This module depends on outbound network access at runtime.
  - Errors are handled gracefully and return empty DataFrames.
  - The response schema is normalized to columns used in src.ais: 
    [timestamp, mmsi, lat, lon, speed_knots, status]
"""

import os
import time
from typing import Optional, Dict, Any
import requests
import pandas as pd


class DatalasticClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.datalastic.com/api/v0",
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATALASTIC_API_KEY")
        if not self.api_key:
            raise ValueError("DATALASTIC_API_KEY not provided (env or argument)")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"x-api-key": self.api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_positions_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        start_iso: str,
        end_iso: str,
        page_size: int = 1000,
        max_pages: int = 10,
        sleep_sec: float = 0.5,
    ) -> pd.DataFrame:
        """Fetch AIS positions inside a bounding box and time window.

        This implementation assumes an endpoint `vessel_positions` that supports
        bbox and time filters. Adjust `path` and parameter names if your plan differs.
        """
        records = []
        page = 1
        while page <= max_pages:
            try:
                payload = {
                    "min_lng": min_lon,
                    "min_lat": min_lat,
                    "max_lng": max_lon,
                    "max_lat": max_lat,
                    "from": start_iso,
                    "to": end_iso,
                    "page": page,
                    "page_size": page_size,
                }
                data = self._get("vessel_positions", payload)
            except Exception:
                break
            items = data.get("data") or data.get("results") or []
            if not items:
                break
            for it in items:
                ts = (
                    it.get("last_position_time")
                    or it.get("timestamp")
                    or it.get("position_timestamp")
                )
                rec = {
                    "timestamp": ts,
                    "mmsi": it.get("mmsi"),
                    "lat": it.get("lat") or it.get("latitude"),
                    "lon": it.get("lon") or it.get("lng") or it.get("longitude"),
                    "speed_knots": it.get("speed") or it.get("speed_knots") or it.get("sog"),
                    "status": it.get("status") or it.get("navigational_status"),
                }
                records.append(rec)
            if len(items) < page_size:
                break
            page += 1
            time.sleep(sleep_sec)
        if not records:
            return pd.DataFrame(columns=["timestamp", "mmsi", "lat", "lon", "speed_knots", "status"])
        df = pd.DataFrame.fromrecords(records)
        # Normalize types
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp", "lat", "lon"])  # minimal sanity
        return df

