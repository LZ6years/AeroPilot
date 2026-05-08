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


def build_remote_airsim_config(
    host: str,
    port: int = 41451,
    vehicle_name: str = "Drone_1",
    camera_name: str = "front_0",
    ready_timeout_sec: int = 30,
    mode: str = "Multirotor",
) -> tuple[list[MapPackage], PortPolicy]:
    # Build a temporary MapPackage for an already-running remote Linux AirSim scene.
    data = {
        "maps": [
            {
                "map_id": "remote_linux_airsim",
                "provider": "packaged_unreal",
                "scene_id": "remote_linux",
                "root": None,
                "launch": {
                    "method": "existing_process",
                    "ready_timeout_sec": ready_timeout_sec,
                },
                "control": {"protocol": "airsim"},
                "airsim": {
                    "host": host,
                    "port": port,
                    "mode": mode,
                    "vehicle_name": vehicle_name,
                    "camera_name": camera_name,
                    "image_width": 224,
                    "image_height": 224,
                    "depth_width": 256,
                    "depth_height": 256,
                },
            }
        ],
        "port_policy": {"server_port": 30000, "airsim_port_start": 30001, "max_scene_num": 8},
    }
    maps = [MapPackage.from_dict(item) for item in data["maps"]]
    policy_data = data["port_policy"]
    policy = PortPolicy(
        server_port=policy_data["server_port"],
        airsim_port_start=policy_data["airsim_port_start"],
        max_scene_num=policy_data["max_scene_num"],
    )
    return maps, policy
