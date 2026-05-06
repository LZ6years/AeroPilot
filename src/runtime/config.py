"""Load runtime configuration files into typed schemas."""

from __future__ import annotations

import json
from pathlib import Path

from src.env_client.schemas import MapPackage, PortPolicy


def load_map_config(path: str | Path) -> tuple[list[MapPackage], PortPolicy]:
    # Read map packages and port policy from one JSON file.
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    port_data = data.get("port_policy", {})
    port_policy = PortPolicy(
        server_port=int(port_data.get("server_port", 30000)),
        airsim_port_start=int(port_data.get("airsim_port_start", 30001)),
        max_scene_num=int(port_data.get("max_scene_num", 8)),
    )
    maps = [MapPackage.from_dict(item) for item in data.get("maps", [])]
    if not maps:
        raise ValueError(f"No maps configured in {config_path}")
    return maps, port_policy
