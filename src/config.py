from __future__ import annotations

from typing import Dict, List
import yaml


def load_priority_ports(yaml_path: str = "config/ports_priority.yaml") -> Dict[str, List[str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("priority_ports", {})

