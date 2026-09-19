"""Privileged TravelUAV dynamic-target placement through AirSim RPC.

This controller is evaluation infrastructure.  It receives hidden placement
truth from ``scenario_payload`` and never writes that truth into Runtime memory
or model-facing context.  Production v1 uses managed scenes, so process exit is
the authoritative cleanup mechanism after Runtime closes the scene.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from evaluation.benchmarks.travel_uav.adapter import TravelUavScenarioPayload
from src.env_client import SceneInstance


class TravelUavScenarioController:
    """Spawn and read back one target before the UAV episode begins."""

    def __init__(
        self,
        *,
        client_factory: Callable[[str, int], Any] | None = None,
        airsim_provider: Callable[[], Any] | None = None,
        position_tolerance_m: float = 0.1,
        cleanup_strategy: str = "process_exit",
    ) -> None:
        if position_tolerance_m <= 0.0:
            raise ValueError("TravelUAV position_tolerance_m must be positive")
        if cleanup_strategy not in {"process_exit", "destroy_object"}:
            raise ValueError("TravelUAV cleanup_strategy is invalid")
        self.client_factory = client_factory or _default_client_factory
        self.airsim_provider = airsim_provider or _import_airsim
        self.position_tolerance_m = float(position_tolerance_m)
        self.cleanup_strategy = cleanup_strategy
        self._spawned_names: dict[str, str] = {}

    def prepare_episode(
        self, scene: SceneInstance, payload: TravelUavScenarioPayload | None
    ) -> None:
        if payload is None:
            raise ValueError("TravelUAV scenario payload is required")
        target = payload.target
        airsim = self.airsim_provider()
        client = self.client_factory(scene.control_host, scene.control_port)
        try:
            ping = getattr(client, "ping", None)
            if callable(ping) and not ping():
                raise RuntimeError("TravelUAV AirSim RPC ping returned false")
            pose = airsim.Pose(
                airsim.Vector3r(*target.position_ned),
                airsim.Quaternionr(*target.rotation_xyzw),
            )
            scale = airsim.Vector3r(*target.scale)
            spawned = client.simSpawnObject(
                target.object_name,
                target.asset_name,
                pose,
                scale,
                physics_enabled=target.physics_enabled,
                is_blueprint=target.is_blueprint,
            )
            spawned_name = str(spawned or "").strip()
            if not spawned_name:
                raise RuntimeError(
                    f"TravelUAV failed to spawn asset {target.asset_name!r}"
                )
            actual = client.simGetObjectPose(spawned_name)
            actual_position = (
                float(actual.position.x_val),
                float(actual.position.y_val),
                float(actual.position.z_val),
            )
            if not all(math.isfinite(value) for value in actual_position):
                raise RuntimeError("TravelUAV spawned target pose is unavailable")
            error = math.sqrt(
                sum(
                    (actual_position[index] - target.position_ned[index]) ** 2
                    for index in range(3)
                )
            )
            if error > self.position_tolerance_m:
                raise RuntimeError(
                    "TravelUAV target placement readback error "
                    f"{error:.3f}m exceeds {self.position_tolerance_m:.3f}m"
                )
            self._spawned_names[scene.scene_instance_id] = spawned_name
        finally:
            _close_client(client)

    def cleanup_episode(
        self, scene: SceneInstance, payload: TravelUavScenarioPayload | None
    ) -> None:
        spawned_name = self._spawned_names.pop(scene.scene_instance_id, None)
        if not spawned_name or self.cleanup_strategy == "process_exit":
            return
        client = self.client_factory(scene.control_host, scene.control_port)
        try:
            if not bool(client.simDestroyObject(spawned_name)):
                raise RuntimeError(
                    f"TravelUAV failed to destroy spawned object {spawned_name!r}"
                )
        finally:
            _close_client(client)


def _import_airsim() -> Any:
    import airsim

    return airsim


def _default_client_factory(host: str, port: int) -> Any:
    airsim = _import_airsim()
    return airsim.VehicleClient(ip=host, port=port, timeout_value=10)


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        close = getattr(getattr(client, "client", None), "close", None)
    if callable(close):
        close()
