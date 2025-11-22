from __future__ import annotations

"""Lightweight satellite imagery helpers (beta).

This module provides basic vessel-like target detection on grayscale
imagery (e.g., SAR-style bright targets) using OpenCV, without heavy
geospatial dependencies. It works on common formats (PNG/JPG/TIFF). If a
GeoTIFF with georeferencing is provided, geographic coordinates are not
computed here (to avoid heavy raster libs); instead, detections are returned
in pixel space. Upstream code can map pixels to lat/lon if needed.

Functions:
- load_image: read image into a grayscale numpy array
- detect_vessels: simple bright-blob detection; returns count and centroids
- annotate_detections: draw markers for quick visualization
"""

from typing import Tuple, List
import numpy as np


def load_image(data: bytes) -> np.ndarray:
    """Load an image from raw bytes as grayscale float32 [0,1]."""
    # Lazy import to avoid hard dependency if unused
    import cv2  # type: ignore
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    img = img.astype(np.float32) / 255.0
    return img


def detect_vessels(
    img: np.ndarray,
    *,
    thresh_rel: float = 0.85,
    min_area_px: int = 4,
    max_area_px: int = 500,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Detect bright compact targets in a grayscale image.

    - thresh_rel: relative threshold in [0,1]; pixels above thresh_rel are candidates
    - min_area_px/max_area_px: filter components by area in pixels
    Returns (count, list of (x,y) centroids in pixel coordinates)
    """
    import cv2  # type: ignore

    if img.ndim != 2:
        raise ValueError("detect_vessels expects a single-channel image")
    # Threshold
    thr_val = float(np.clip(thresh_rel, 0.0, 1.0))
    _, mask = cv2.threshold((img * 255).astype(np.uint8), int(thr_val * 255), 255, cv2.THRESH_BINARY)
    # Morphological clean-up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    points: List[Tuple[int, int]] = []
    for i in range(1, num_labels):  # skip background 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue
        cx, cy = centroids[i]
        points.append((int(cx), int(cy)))
    return len(points), points


def annotate_detections(img: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    """Return an RGB image with small markers at detection points."""
    import cv2  # type: ignore
    if img.ndim == 2:
        rgb = np.dstack([img, img, img])
    else:
        rgb = img.copy()
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    for (x, y) in points:
        cv2.drawMarker(rgb_u8, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    return rgb_u8

