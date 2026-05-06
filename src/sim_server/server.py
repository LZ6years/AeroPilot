"""Run the local SimulationServer and manage scene lifecycle."""

from __future__ import annotations

import json
import platform
import socket
import time
import uuid
from pathlib import Path

from src.env_client.schemas import ActionResult, MapPackage, Observation, PortPolicy, SceneInstance
from src.sim_server.backends.airvln_backend import AirVlnBackend
from src.sim_server.backends.base import SimulationBackend
from src.sim_server.backends.local_ue_backend import LocalUeBackend
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend
from src.sim_server.port_manager import PortManager
from src.sim_server.process_manager import ProcessManager, windows_path_to_wsl
from src.sim_server.scene_manager import SceneManager


class LocalSimulationServer:
    def __init__(
        self,
        map_packages: list[MapPackage],
        port_policy: PortPolicy | None = None,
        run_dir: str | Path = "runs/dev",
        launch_processes: bool = False,
        real_control: bool | None = None,
    ) -> None:
        # Initialize scene, process, port, and backend managers.
        self.map_packages = {pkg.map_id: pkg for pkg in map_packages}
        self.port_policy = port_policy or PortPolicy()
        self.port_manager = PortManager(self.port_policy)
        self.process_manager = ProcessManager()
        self.scene_manager = SceneManager()
        self.run_dir = Path(run_dir)
        self.launch_processes = launch_processes
        self.real_control = launch_processes if real_control is None else real_control
        self.backends: dict[str, SimulationBackend] = {}

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
            launch_method = package.launch.get("method", "direct")
            settings_arg = None
            if package.control_protocol == "airsim":
                settings_arg = str(settings_path.resolve())
                if launch_method == "wsl_bash":
                    settings_arg = windows_path_to_wsl(settings_arg)
            log_path = self.run_dir / "scenes" / instance_id / "process.log"
            scene.process_id = self.process_manager.start(
                instance_id,
                executable,
                package.launch.get("args", []),
                method=launch_method,
                settings_path=settings_arg,
                log_path=str(log_path),
            )
            try:
                self._wait_for_airsim(scene, package)
            except Exception:
                self.process_manager.stop(instance_id)
                self.port_manager.release(scene.airsim_port)
                raise
        elif self.real_control:
            self._wait_for_control(scene, package)

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
        # Select the backend implementation for the configured control protocol.
        backend_key = package.control_protocol
        if backend_key not in self.backends:
            if backend_key == "airsim":
                if package.provider == "local_ue":
                    self.backends[backend_key] = LocalUeBackend(use_real_uav=self.real_control)
                else:
                    self.backends[backend_key] = AirVlnBackend(use_real_uav=self.real_control)
            elif backend_key == "unrealcv":
                self.backends[backend_key] = UnrealCvBackend(use_real_client=self.real_control)
            else:
                raise ValueError(f"Unsupported control protocol: {package.control_protocol}")
        try:
            return self.backends[backend_key]
        except KeyError as exc:
            raise ValueError(f"Unsupported control protocol: {package.control_protocol}") from exc

    def _write_airsim_settings(self, package: MapPackage, instance_id: str, airsim_port: int) -> Path:
        # Write the per-scene AirSim settings file into the run directory.
        scene_dir = (self.run_dir / "scenes" / instance_id).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        settings_path = scene_dir / "settings.json"
        if package.control_protocol == "unrealcv":
            settings_path.write_text(
                json.dumps({"provider": package.provider, "control_protocol": "unrealcv"}, indent=2),
                encoding="utf-8",
            )
            return settings_path
        vehicle_name = package.airsim["vehicle_name"]
        camera_name = package.airsim["camera_name"]
        image_width = int(package.airsim.get("image_width", 224))
        image_height = int(package.airsim.get("image_height", 224))
        depth_width = int(package.airsim.get("depth_width", image_width))
        depth_height = int(package.airsim.get("depth_height", image_height))
        capture_settings = [
            {
                "ImageType": 0,
                "Width": image_width,
                "Height": image_height,
                "FOV_Degrees": package.airsim.get("fov_degrees", 90),
            },
            {
                "ImageType": 3,
                "Width": depth_width,
                "Height": depth_height,
                "FOV_Degrees": package.airsim.get("fov_degrees", 90),
            },
            {
                "ImageType": 2,
                "Width": depth_width,
                "Height": depth_height,
                "FOV_Degrees": package.airsim.get("fov_degrees", 90),
            },
        ]
        settings = {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
            "SettingsVersion": 1.2,
            "SimMode": package.airsim.get("mode", "Multirotor"),
            "ViewMode": package.airsim.get("view_mode", "NoDisplay"),
            "ApiServerPort": airsim_port,
            "Vehicles": {vehicle_name: {"VehicleType": "SimpleFlight", "AutoCreate": True}},
        }
        if package.airsim.get("settings_profile") == "camera_defaults":
            settings["CameraDefaults"] = {"CaptureSettings": capture_settings}
            settings["Vehicles"][vehicle_name].update(
                {
                    "X": package.airsim.get("x", 0),
                    "Y": package.airsim.get("y", 0),
                    "Z": package.airsim.get("z", 0),
                }
            )
        else:
            settings["Vehicles"][vehicle_name]["Cameras"] = {
                camera_name: {"CaptureSettings": capture_settings}
            }
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return settings_path

    def _resolve_executable(self, package: MapPackage) -> str:
        # Resolve the platform-specific scene executable path.
        system = platform.system().lower()
        key = "windows" if system == "windows" else "linux"
        executable = package.launch.get(key) or package.launch.get(f"{key}_script")
        if executable is None and package.launch.get("method") == "wsl_bash":
            executable = package.launch.get("linux")
        if executable is None:
            raise ValueError(f"MapPackage {package.map_id} has no {key} executable")
        root = Path(package.root or ".")
        path = Path(executable)
        return str(path if path.is_absolute() else root / path)

    def _wait_for_airsim(self, scene: SceneInstance, package: MapPackage) -> None:
        # Wait until the AirSim RPC port accepts TCP connections.
        self._wait_for_control(scene, package)

    def _wait_for_control(self, scene: SceneInstance, package: MapPackage) -> None:
        # Wait until the configured control protocol endpoint accepts connections.
        if package.control_protocol == "unrealcv":
            host = str(package.unrealcv.get("host", scene.host))
            port = int(package.unrealcv.get("port", scene.airsim_port))
            label = "UnrealCV"
        else:
            host = str(package.airsim.get("host", scene.host))
            port = int(package.airsim.get("port", scene.airsim_port))
            label = "AirSim"
        self._wait_for_tcp_port(
            host,
            port,
            float(package.launch.get("ready_timeout_sec", 90)),
            scene.settings_path,
            label,
        )

    def _wait_for_tcp_port(
        self, host: str, port: int, timeout: float, settings_path: str, label: str
    ) -> None:
        # Wait until a TCP service accepts connections.
        deadline = time.monotonic() + timeout
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(
            f"{label} port did not become ready: {host}:{port} "
            f"within {timeout:.0f}s; settings={settings_path}; last_error={last_error}"
        )
