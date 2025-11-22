from __future__ import annotations

"""AISstream.io WebSocket connector (beta).

This module subscribes to AISstream.io live AIS and aggregates vessel
positions into daily port features using existing geofencing utilities.

Notes:
- Requires websocket-client
- AISstream token is provided via argument or AISSTREAM_TOKEN env var
- Subscription payload uses a common format with APIKey + BoundingBoxes + FilterMessageTypes
  (Adapt the keys if your plan differs.)
"""

from typing import List, Tuple, Optional
import os
import time
import json
import threading
import pandas as pd


def _build_bboxes_from_geojson(geojson_path: str) -> List[List[List[float]]]:
    from ..ais import load_port_geofences, polygon_bounds
    fences = load_port_geofences(geojson_path)
    bboxes: List[List[List[float]]] = []
    for f in fences:
        mnx, mny, mxx, mxy = polygon_bounds(f.polygons)
        # AISstream uses [[minLng,minLat],[maxLng,maxLat]] per bbox
        bboxes.append([[float(mnx), float(mny)], [float(mxx), float(mxy)]])
    return bboxes


def _subscribe_payload(token: str, bboxes: List[List[List[float]]]) -> dict:
    return {
        "APIKey": token,
        "BoundingBoxes": bboxes,
        "FilterMessageTypes": ["PositionReport"],
    }


def _parse_message(msg: str) -> Optional[dict]:
    try:
        data = json.loads(msg)
    except Exception:
        return None
    # Flexible parsing across potential schemas
    mtype = (data.get("MessageType") or data.get("messageType") or "").lower()
    if "position" not in mtype and "positionreport" not in mtype and data.get("type") != "PositionReport":
        return None
    lat = data.get("Latitude") or data.get("lat") or (data.get("Position") or {}).get("Latitude")
    lon = data.get("Longitude") or data.get("lon") or (data.get("Position") or {}).get("Longitude")
    if lat is None or lon is None:
        return None
    ts = (
        data.get("Timestamp")
        or data.get("timestamp")
        or (data.get("MetaData") or {}).get("timestamp")
    )
    mmsi = data.get("MMSI") or data.get("mmsi")
    sog = data.get("SOG") or data.get("speed_knots") or data.get("speed")
    status = data.get("NavigationalStatus") or data.get("status")
    return {
        "timestamp": ts,
        "mmsi": mmsi,
        "lat": float(lat),
        "lon": float(lon),
        "speed_knots": float(sog) if sog is not None else None,
        "status": status,
    }


def stream_and_append(
    *,
    token: Optional[str],
    geojson_path: str,
    minutes: int,
    db_path: str,
    table: str = "port_features",
) -> int:
    """Stream AIS for a few minutes and append aggregated features to DuckDB.

    Returns the table row count after append.
    """
    import websocket  # type: ignore
    from ..ais import assign_port_to_points, derive_daily_port_features
    from ..duckdb_io import write_port_features_df, ensure_port_features_table
    from datetime import datetime, timezone

    tok = token or os.environ.get("AISSTREAM_TOKEN")
    if not tok:
        raise ValueError("AISstream token not provided (use --token or AISSTREAM_TOKEN env var)")

    url = os.environ.get("AISSTREAM_URL", "wss://stream.aisstream.io/v0/stream")
    bboxes = _build_bboxes_from_geojson(geojson_path)
    subscribe = _subscribe_payload(tok, bboxes)

    rows: List[dict] = []
    stop_ts = time.time() + max(1, minutes) * 60
    ws: websocket.WebSocketApp

    def on_open(ws):
        try:
            ws.send(json.dumps(subscribe))
        except Exception:
            ws.close()

    def on_message(ws, message):
        if time.time() >= stop_ts:
            ws.close();
            return
        rec = _parse_message(message)
        if rec:
            rows.append(rec)

    def on_error(ws, err):
        # Allow exit but keep collected rows
        try:
            ws.close()
        except Exception:
            pass

    def on_close(ws, *args, **kwargs):
        pass

    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)

    # Run in thread so we can timeout
    t = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 30, "ping_timeout": 10})
    t.daemon = True
    t.start()
    while t.is_alive() and time.time() < stop_ts:
        time.sleep(0.2)
    try:
        ws.close()
    except Exception:
        pass

    if not rows:
        return 0
    # Build DataFrame and aggregate to daily features
    df = pd.DataFrame(rows)
    # Ensure timestamp is parsed
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception:
        df["timestamp"] = pd.Timestamp.now(tz=timezone.utc)
    # Assign ports and compute features
    from ..ais import load_port_geofences
    fences = load_port_geofences(geojson_path)
    df_tag = assign_port_to_points(df, fences)
    df_tag = df_tag[df_tag["port"].notna()]
    if df_tag.empty:
        return 0
    feats = derive_daily_port_features(df_tag)
    ensure_port_features_table(db_path, table)
    total = write_port_features_df(db_path, feats, table=table, mode="append")
    return int(total)

