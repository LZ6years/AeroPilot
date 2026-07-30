"""Validate and render user-owned AirSim settings templates.

The renderer never edits the supplied template. It creates a scene-specific
copy and only changes deployment-owned fields that must vary between runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.env_client.schemas import MapPackage


@dataclass(frozen=True)
class RenderedAirSimSettings:
    path: Path
    sha256: str


class AirSimSettingsRenderer:
    """Render a validated settings template for one scene instance."""

    def render(
        self,
        package: MapPackage,
        destination: str | Path,
        api_server_port: int,
        template_path: str | Path | None = None,
    ) -> RenderedAirSimSettings:
        template_value = (
            template_path
            if template_path is not None
            else os.path.expandvars(str(package.airsim["settings_template"]))
        )
        source_path = Path(template_value).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(f"AirSim settings template not found: {source_path}")

        settings = json.loads(source_path.read_text(encoding="utf-8-sig"))
        self._validate(package, settings, source_path)
        settings["ApiServerPort"] = int(api_server_port)
        bind_host = package.connection.get("bind_host")
        if bind_host:
            settings["LocalHostIp"] = str(bind_host)

        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(settings, indent=2, ensure_ascii=False).encode("utf-8")
        destination_path.write_bytes(payload)
        return RenderedAirSimSettings(
            path=destination_path,
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    @staticmethod
    def _validate(
        package: MapPackage,
        settings: dict[str, Any],
        template_path: Path,
    ) -> None:
        if "SettingsVersion" not in settings:
            raise ValueError(f"{template_path} missing SettingsVersion")
        expected_mode = package.airsim.get("mode")
        actual_mode = settings.get("SimMode")
        if expected_mode and actual_mode and str(expected_mode) != str(actual_mode):
            raise ValueError(
                f"AirSim SimMode mismatch: map={expected_mode!r}, "
                f"settings={actual_mode!r}"
            )

        vehicle_name = str(package.airsim["vehicle_name"])
        vehicles = settings.get("Vehicles", {})
        if vehicle_name not in vehicles:
            raise ValueError(
                f"AirSim settings missing vehicle {vehicle_name!r}: {template_path}"
            )

        camera_name = str(package.airsim["camera_name"])
        cameras = vehicles[vehicle_name].get("Cameras")
        if cameras is not None and camera_name not in cameras:
            raise ValueError(
                f"AirSim settings vehicle {vehicle_name!r} missing camera "
                f"{camera_name!r}"
            )
