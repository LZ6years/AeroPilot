"""Run the local SimulationServer and manage scene lifecycle."""

from __future__ import annotations

import json
import platform
import uuid
from pathlib import Path

from src.env_client.schemas import ActionResult, MapPackage, Observation, PortPolicy, SceneInstance
from src.sim_server.backends.airvln_backend import AirVlnBackend
from src.sim_server.backends.base import SimulationBackend
from src.sim_server.backends.local_ue_backend import LocalUeBackend
from src.sim_server.port_manager import PortManager
from src.sim_server.process_manager import ProcessManager
from src.sim_server.scene_manager import SceneManager


class LocalSimulationServer:
    def __init__(
        self,
        map_packages: list[MapPackage],
        port_policy: PortPolicy | None = None,
        run_dir: str | Path = "runs/dev",
        launch_processes: bool = False,
    ) -> None:
        # Initialize scene, process, port, and backend managers.
        self.map_packages = {pkg.map_id: pkg for pkg in map_packages}
        self.port_policy = port_policy or PortPolicy()
        self.port_manager = PortManager(self.port_policy)
        self.process_manager = ProcessManager()
        self.scene_manager = SceneManager()
        self.run_dir = Path(run_dir)
        self.launch_processes = launch_processes
        self.backends: dict[str, SimulationBackend] = {
            "airvln": AirVlnBackend(),
            "local_ue": LocalUeBackend(),
        }

    def health_check(self) -> bool:
        # Report that the in-process server object is alive.
        return True

    def list_maps(self) -> list[MapPackage]:
        # Return all configured static map packages.
        return list(self.map_packages.values())

    def start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        # Create a SceneInstance, write settings, and hand it to the backend.
        package = self._get_package(map_id)
        if scene_id is not None and str(package.scene_id) != str(scene_id):
            raise ValueError(f"Map {map_id} does not contain scene_id {scene_id}")

        instance_id = f"scene_{uuid.uuid4().hex[:8]}"
        airsim_port = self.port_manager.allocate_airsim_port()
        settings_path = self._write_airsim_settings(package, instance_id, airsim_port)
        scene = SceneInstance(
            scene_instance_id=instance_id,
            map_id=package.map_id,
            scene_id=package.scene_id,
            machine_id="local",
            host="127.0.0.1",
            server_port=self.port_policy.server_port,
            airsim_port=airsim_port,
            settings_path=str(settings_path),
            process_id=None,
            status="created",
        )

        if self.launch_processes:
            executable = self._resolve_executable(package)
            scene.process_id = self.process_manager.start(instance_id, executable, package.launch.get("args", []))

        backend = self._backend_for(package)
        scene = backend.start(package, scene)
        self.scene_manager.add(scene)
        return scene

    def stop_scene(self, scene_instance_id: str) -> None:
        # Stop backend state, process state, and release its allocated port.
        scene = self.scene_manager.remove(scene_instance_id)
        package = self._get_package(scene.map_id)
        self._backend_for(package).stop(scene)
        self.process_manager.stop(scene_instance_id)
        self.port_manager.release(scene.airsim_port)

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        # Mark an existing scene as running again for this first slice.
        scene = self.scene_manager.update_status(scene_instance_id, "running")
        return scene

    def get_scene_status(self, scene_instance_id: str) -> str:
        # Read the server-side status for one scene instance.
        return self.scene_manager.get(scene_instance_id).status

    def get_observation(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        # Route observation requests to the configured backend.
        scene = self.scene_manager.get(scene_instance_id)
        package = self._get_package(scene.map_id)
        return self._backend_for(package).get_observation(scene, agent_id, modalities)

    def execute_action(
        self, scene_instance_id: str, agent_id: str, action: str, args: dict
    ) -> ActionResult:
        # Route primitive action requests to the configured backend.
        scene = self.scene_manager.get(scene_instance_id)
        package = self._get_package(scene.map_id)
        return self._backend_for(package).execute_action(scene, agent_id, action, args)

    def _get_package(self, map_id: str) -> MapPackage:
        # Look up a static map package by id.
        try:
            return self.map_packages[map_id]
        except KeyError as exc:
            raise KeyError(f"Unknown map_id: {map_id}") from exc

    def _backend_for(self, package: MapPackage) -> SimulationBackend:
        # Select the backend implementation for a map provider.
        try:
            return self.backends[package.provider]
        except KeyError as exc:
            raise ValueError(f"Unsupported map provider: {package.provider}") from exc

    def _write_airsim_settings(self, package: MapPackage, instance_id: str, airsim_port: int) -> Path:
        # Write the per-scene AirSim settings file into the run directory.
        scene_dir = self.run_dir / "scenes" / instance_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        settings_path = scene_dir / "settings.json"
        vehicle_name = package.airsim["vehicle_name"]
        settings = {
            "SettingsVersion": 1.2,
            "SimMode": package.airsim.get("mode", "Multirotor"),
            "ApiServerPort": airsim_port,
            "Vehicles": {
                vehicle_name: {
                    "VehicleType": "SimpleFlight",
                    "AutoCreate": True,
                }
            },
        }
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return settings_path

    def _resolve_executable(self, package: MapPackage) -> str:
        # Resolve the platform-specific scene executable path.
        system = platform.system().lower()
        key = "windows" if system == "windows" else "linux"
        executable = package.launch.get(key) or package.launch.get(f"{key}_script")
        if executable is None:
            raise ValueError(f"MapPackage {package.map_id} has no {key} executable")
        root = Path(package.root or ".")
        path = Path(executable)
        return str(path if path.is_absolute() else root / path)
