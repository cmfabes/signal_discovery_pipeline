"""AIS connector stub with port geofencing and daily feature builder.

This module provides a minimal, dependency-light implementation for:
- Loading port geofences from a GeoJSON file (Polygon/Multipolygon)
- Assigning AIS points to ports via point-in-polygon
- Deriving daily features per port (anchored vessels, arrivals, departures)

It is designed as a stub for local/offline experimentation. In production,
you likely want to replace the geofencing with a robust spatial stack
using GeoPandas/Shapely and stream/process AIS from your provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional
import json
import math
import pandas as pd


@dataclass
class PortFence:
    name: str
    polygons: List[List[Tuple[float, float]]]  # list of rings (lon, lat)


def _extract_polygons_from_geojson(feature: Dict[str, Any]) -> List[List[Tuple[float, float]]]:
    geom = feature.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    polygons: List[List[Tuple[float, float]]] = []
    if gtype == "Polygon":
        # coords: [ [ [lon,lat], ... ] ]
        if coords:
            rings = coords[0]
            polygons.append([(float(x), float(y)) for x, y in rings])
    elif gtype == "MultiPolygon":
        for poly in coords:
            # poly: [ [lon,lat], ... ] inside an extra nesting
            if poly:
                rings = poly[0]
                polygons.append([(float(x), float(y)) for x, y in rings])
    return polygons


def load_port_geofences(geojson_path: str, *, name_property: str = "name") -> List[PortFence]:
    """Load port geofences from a simple GeoJSON file.

    The GeoJSON is expected to contain FeatureCollection with Polygon or
    MultiPolygon geometries. A 'name' property is used for the port label.
    """
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    features = gj.get("features", [])
    fences: List[PortFence] = []
    for feat in features:
        props = feat.get("properties", {})
        name = props.get(name_property) or props.get("Name") or props.get("PORT_NAME")
        if not name:
            continue
        polys = _extract_polygons_from_geojson(feat)
        if polys:
            fences.append(PortFence(name=name, polygons=polys))
    return fences


def _point_in_polygon(lon: float, lat: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon (lon/lat order)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if ray crosses edge
        cond = ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-12) + x1
        )
        if cond:
            inside = not inside
    return inside


def assign_port_to_points(
    df: pd.DataFrame,
    fences: List[PortFence],
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    out_col: str = "port",
) -> pd.DataFrame:
    """Assign a port name to each AIS point if it falls within any fence.

    This is O(N*M) over points and polygons; acceptable for small batches.
    """
    df = df.copy()
    ports: List[Optional[str]] = []
    for _, row in df.iterrows():
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        assigned = None
        for fence in fences:
            for poly in fence.polygons:
                if _point_in_polygon(lon, lat, poly):
                    assigned = fence.name
                    break
            if assigned:
                break
        ports.append(assigned)
    df[out_col] = ports
    return df


def derive_daily_port_features(
    ais: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    mmsi_col: str = "mmsi",
    port_col: str = "port",
    speed_col: str = "speed_knots",
    status_col: str = "status",
) -> pd.DataFrame:
    """Compute daily features per port from AIS points already tagged with ports.

    Expected columns: [timestamp, mmsi, lat, lon, speed_knots, status, port]
    Output columns per day+port: anchored_count, arrivals_count, departures_count,
    avg_speed_knots, distinct_vessels, anchored_vessel_hours.
    - A vessel is "anchored" if status contains 'anchor' (case-insensitive) or speed < 1 knot.
    - Arrivals/Departures are counted by changes of in_port flag per vessel per day.
    """
    df = ais.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df["Date"] = df[timestamp_col].dt.floor("D")
    df = df.sort_values([mmsi_col, timestamp_col])
    df["is_anchored"] = df[status_col].astype(str).str.lower().str.contains("anchor") | (df[speed_col].fillna(0) < 1.0)
    # In-port flag assumed by presence of a port label
    df["in_port"] = df[port_col].notna()

    # Detect arrivals/departures per vessel based on in_port transitions
    df["prev_in_port"] = df.groupby(mmsi_col)["in_port"].shift(1)
    df["arrival"] = (~df["prev_in_port"].fillna(False)) & (df["in_port"])
    df["departure"] = (df["prev_in_port"].fillna(False)) & (~df["in_port"])  # leaving

    # Aggregate daily per port
    agg = df[df["in_port"]].groupby(["Date", port_col]).agg(
        anchored_count=("is_anchored", "sum"),
        avg_speed_knots=(speed_col, "mean"),
        distinct_vessels=(mmsi_col, pd.Series.nunique),
    )
    arr = df[df["arrival"] & df["in_port"]].groupby(["Date", port_col])["arrival"].sum().rename("arrivals_count")
    dep = df[df["departure"] & ~df["in_port"]].groupby(["Date", port_col])["departure"].sum().rename("departures_count")

    out = agg.join(arr, how="left").join(dep, how="left").fillna(0)
    # Approximate anchored vessel-hours as anchored_count (points) normalized by number of samples per day
    # For a stub, treat anchored_count as proxy; real impl should account for sampling frequency.
    out["anchored_vessel_hours"] = out["anchored_count"]  # placeholder proxy
    out = out.reset_index()
    return out


def polygon_bounds(polygons: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    """Return bounding box (min_lon, min_lat, max_lon, max_lat) for a list of rings."""
    lons = []
    lats = []
    for ring in polygons:
        for x, y in ring:
            lons.append(x)
            lats.append(y)
    return (min(lons), min(lats), max(lons), max(lats))
