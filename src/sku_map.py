from __future__ import annotations

"""SKU/manifest stubs for mapping features to industries (extensible).

This is a placeholder for future provider integration. For now we use
simple keyword rules; later, wire real manifest/SKU categories and map
to sub‑industries.
"""

from typing import List


KEYWORD_INDUSTRIES = {
    "container": ["Transports", "Retail", "Industrials"],
    "anchor": ["Transports", "Retail", "Industrials"],
    "arrival": ["Retail", "Industrials"],
    "dwell": ["Transports", "Retail"],
    "oil": ["Energy"],
    "tanker": ["Energy"],
    "bulk": ["Materials", "Industrials"],
}


def affected_industries(port: str | None, feature: str) -> List[str]:
    name = (feature or "").lower()
    inds: list[str] = []
    for k, v in KEYWORD_INDUSTRIES.items():
        if k in name:
            inds.extend(v)
    # Deduplicate, keep order
    seen = set(); out: list[str] = []
    for i in inds:
        if i not in seen:
            out.append(i); seen.add(i)
    return out or ["Broad Market"]

