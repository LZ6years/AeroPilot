"""Coordinate scene lifecycle, deployment control, and per-scene backends."""

from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path

from src.env_client.schemas import (
    ActionResult,
    MapPackage,
    Observation,
    PortPolicy,
    SceneInstance,
)
from src.runtime.configs.action_config import UavActionConfig

from .airsim_settings import AirSimSettingsRenderer
from .backends.base import SimulationBackend
from .backends.registry import BackendContext, BackendRegistry
from .execution import SimulationExecutionMode
from .port_manager import PortManager
from .process_control import LocalSceneControl
from .remote.control import RemoteSceneControl
from .remote.tunnel import SshTunnel
from .scene_manager import SceneManager


class SimulationServer:
    """Compose deployment and protocol strategies behind one EnvClient service."""

    def __init__(
        self,
        map_packages: list[MapPackage],
        port_policy: PortPolicy | None = None,
        run_dir: str | Path = "runs/dev",
        execution_mode: SimulationExecutionMode | str = (
            SimulationExecutionMode.MANAGED
        ),
        show_display: bool | None = None,
        action_config: UavActionConfig | None = None,
        save_artifacts: bool = False,
        remote_scene_control: RemoteSceneControl | None = None,
        settings_renderer: AirSimSettingsRenderer | None = None,
        backend_registry: BackendRegistry | None = None,
    ) -> None:
        self.map_packages = {pkg.map_id: pkg for pkg in map_packages}
        self.port_policy = port_policy or PortPolicy()
        self.port_manager = PortManager(self.port_policy)
        self.scene_manager = SceneManager()
        self.run_dir = Path(run_dir)
        self.execution_mode = SimulationExecutionMode.parse(execution_mode)
        self.show_display = show_display
        self.action_config = action_config or UavActionConfig()
        self.save_artifacts = save_artifacts
        self.artifact_dir = self.run_dir / "artifacts"
        self.local_scene_control = LocalSceneControl(show_display=show_display)
        self.remote_scene_control = remote_scene_control or RemoteSceneControl()
        self.settings_renderer = settings_renderer or AirSimSettingsRenderer()
        self.backend_registry = backend_registry or BackendRegistry.defaults()

        # Simulator SDK clients and SSH tunnels are process-local resources and
        # therefore remain keyed by SceneInstance rather than serialized.
        self.backends: dict[str, SimulationBackend] = {}
        self.tunnels: dict[str, SshTunnel] = {}
        self._reserved_ports: dict[str, int] = {}

    def health_check(self) -> bool:
        return True

    def list_maps(self) -> list[MapPackage]:
        return list(self.map_packages.values())

    def start_scene(
        self, map_id: str, scene_id: str | None = None
    ) -> SceneInstance:
        package = self._get_package(map_id)
        if scene_id is not None and str(package.scene_id) != str(scene_id):
            raise ValueError(f"Map {map_id} does not contain scene_id {scene_id}")

        instance_id = f"scene_{uuid.uuid4().hex[:8]}"
        scene_dir = (self.run_dir / "scenes" / instance_id).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        backend = self.backend_registry.create(
            package.control_protocol,
            BackendContext(
                action_config=self.action_config,
                artifact_dir=self.artifact_dir / instance_id,
                save_artifacts=self.save_artifacts,
            ),
        )
        remote_owned = False
        local_owned = False
        backend_started = False

        try:
            if package.connection_mode == "remote":
                scene, remote_owned = self._prepare_remote_scene(
                    package, instance_id, scene_dir, backend
                )
            else:
                scene, local_owned = self._prepare_local_scene(
                    package, instance_id, scene_dir, backend
                )

            scene.capabilities = backend.describe_capabilities(package)
            scene = backend.start(package, scene)
            backend_started = True
            self.backends[instance_id] = backend
            self.scene_manager.add(scene)
            return scene
        except Exception:
            if backend_started:
                try:
                    backend.stop(scene)  # type: ignore[possibly-undefined]
                except Exception:
                    pass
            if remote_owned:
                try:
                    self.remote_scene_control.stop(package, instance_id)
                except Exception:
                    pass
            if local_owned:
                self.local_scene_control.stop(instance_id)
            self._close_transport(instance_id)
            raise

    def get_or_start_scene(
        self, map_id: str, scene_id: str | None = None
    ) -> SceneInstance:
        running = self.scene_manager.find_running(map_id, scene_id)
        return running if running is not None else self.start_scene(map_id, scene_id)

    def stop_scene(self, scene_instance_id: str) -> None:
        scene = self.scene_manager.get(scene_instance_id)
        package = self._get_package(scene.map_id)
        backend = self.backends.pop(scene_instance_id)
        backend.stop(scene)
        try:
            if scene.process_ownership == "remote_managed":
                self.remote_scene_control.stop(package, scene_instance_id)
            elif scene.process_ownership == "local":
                self.local_scene_control.stop(scene_instance_id)
        finally:
            self._close_transport(scene_instance_id)
            self.scene_manager.remove(scene_instance_id)

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        scene = self.scene_manager.get(scene_instance_id)
        if scene.capabilities and not scene.capabilities.supports_soft_reset:
            return self.scene_manager.update_status(scene_instance_id, "running")
        return self.backends[scene_instance_id].reset(scene)

    def get_scene_status(self, scene_instance_id: str) -> str:
        return self.scene_manager.get(scene_instance_id).status

    def get_observation(
        self,
        scene_instance_id: str,
        agent_id: str,
        modalities: list[str],
        capture_sources: dict[str, str] | None = None,
    ) -> Observation:
        scene = self.scene_manager.get(scene_instance_id)
        return self.backends[scene_instance_id].get_observation(
            scene, agent_id, modalities, capture_sources
        )

    def execute_action(
        self,
        scene_instance_id: str,
        agent_id: str,
        action: str,
        args: dict,
    ) -> ActionResult:
        scene = self.scene_manager.get(scene_instance_id)
        if scene.capabilities and not scene.capabilities.supports(action):
            return ActionResult(
                False,
                action,
                (
                    f"{scene.capabilities.protocol}/"
                    f"{scene.capabilities.control_mode} does not support {action}"
                ),
            )
        return self.backends[scene_instance_id].execute_action(
            scene, agent_id, action, args
        )

    def _prepare_remote_scene(
        self,
        package: MapPackage,
        instance_id: str,
        scene_dir: Path,
        backend: SimulationBackend,
    ) -> tuple[SceneInstance, bool]:
        remote_host, remote_port = self._remote_endpoint(package)
        local_port = self.port_manager.allocate_control_port()
        self._reserved_ports[instance_id] = local_port
        self.remote_scene_control.preflight(package)
        settings_path, settings_sha256 = self._prepare_airsim_settings(
            package,
            instance_id,
            scene_dir,
            remote_port,
            remote=True,
        )

        tunnel = self.remote_scene_control.open_tunnel(
            package, local_port, remote_host, remote_port
        )
        self.tunnels[instance_id] = tunnel

        scene = self._new_scene(
            package,
            instance_id,
            host="127.0.0.1",
            control_port=local_port,
            settings_path=settings_path,
            settings_sha256=settings_sha256,
            ownership="external",
        )
        protocol_ready = backend.probe(
            package, "127.0.0.1", local_port
        ).ready
        if protocol_ready:
            if (
                package.launch.get("existing_process_policy", "attach")
                == "reject"
            ):
                raise ConnectionError(
                    f"Remote {package.control_protocol} is already ready at "
                    f"{remote_host}:{remote_port}, but map "
                    f"{package.map_id} requires a fresh managed process"
                )
            return scene, False

        remote_status = self.remote_scene_control.remote_port_status(
            package,
            instance_id,
            remote_host,
            remote_port,
            scene_dir,
        )
        if remote_status.get("status") == "reachable":
            detail = backend.probe(
                package, "127.0.0.1", local_port
            ).detail
            raise ConnectionError(
                f"Remote port {remote_host}:{remote_port} is open but does "
                f"not speak {package.control_protocol}: {detail}"
            )

        can_start = (
            self.execution_mode.may_start_process
            and bool(package.launch.get("linux_executable"))
            and not bool(package.launch.get("manual", False))
        )
        if not can_start:
            raise ConnectionError(
                f"Remote {package.control_protocol} is not ready at "
                f"{remote_host}:{remote_port}, and managed launch is disabled"
            )

        result = self.remote_scene_control.start(
            package, instance_id, settings_path, scene_dir
        )
        scene.process_ownership = "remote_managed"
        scene.remote_process_id = int(result["pid"])
        if result.get("remote_settings_path"):
            scene.settings_path = str(result["remote_settings_path"])
        self._wait_for_backend(backend, package, "127.0.0.1", local_port)
        self._post_launch_wait(package)
        return scene, True

    def _prepare_local_scene(
        self,
        package: MapPackage,
        instance_id: str,
        scene_dir: Path,
        backend: SimulationBackend,
    ) -> tuple[SceneInstance, bool]:
        configured_host, configured_port = self._configured_endpoint(package)
        managed = self.execution_mode.may_start_process and not bool(
            package.launch.get("manual", False)
        )
        if package.control_protocol == "airsim" and managed:
            control_port = self.port_manager.allocate_control_port()
            self._reserved_ports[instance_id] = control_port
        else:
            control_port = configured_port

        settings_path, settings_sha256 = self._prepare_airsim_settings(
            package,
            instance_id,
            scene_dir,
            control_port,
            remote=False,
        )
        scene = self._new_scene(
            package,
            instance_id,
            host=configured_host,
            control_port=control_port,
            settings_path=settings_path,
            settings_sha256=settings_sha256,
            ownership="external" if not managed else "local",
        )

        if managed:
            result = self.local_scene_control.start(
                package,
                instance_id,
                settings_path,
                scene_dir / "process.log",
            )
            scene.process_id = result.process_id
            self._wait_for_backend(
                backend, package, configured_host, control_port
            )
            self._post_launch_wait(package)
            return scene, True
        probe = backend.probe(package, configured_host, control_port)
        if not probe.ready:
            raise ConnectionError(
                f"{package.control_protocol} is not ready at "
                f"{configured_host}:{control_port}: {probe.detail}"
            )
        return scene, False

    def _prepare_airsim_settings(
        self,
        package: MapPackage,
        instance_id: str,
        scene_dir: Path,
        api_server_port: int,
        *,
        remote: bool,
    ) -> tuple[Path | None, str | None]:
        if package.control_protocol != "airsim":
            return None, None
        template = package.airsim.get("settings_template")
        if template:
            template_path: str | Path = str(template)
            source = str(
                package.airsim.get("settings_template_source", "local")
            )
            if remote and source == "remote":
                template_path = self.remote_scene_control.download_settings_template(
                    package, scene_dir / "source.settings.json"
                )
            rendered = self.settings_renderer.render(
                package,
                scene_dir / "settings.json",
                api_server_port,
                template_path=template_path,
            )
            return rendered.path, rendered.sha256

        # Existing local configs are migrated incrementally. Keep deterministic
        # generated settings for local packages that do not yet use a template.
        if remote:
            if package.launch.get("manual", False):
                return None, None
            raise ValueError(
                f"Remote AirSim map {package.map_id} requires settings_template"
            )
        path = self._write_airsim_settings(
            package, instance_id, api_server_port
        )
        return path, None

    def _new_scene(
        self,
        package: MapPackage,
        instance_id: str,
        *,
        host: str,
        control_port: int,
        settings_path: Path | None,
        settings_sha256: str | None,
        ownership: str,
    ) -> SceneInstance:
        return SceneInstance(
            scene_instance_id=instance_id,
            map_id=package.map_id,
            scene_id=package.scene_id,
            machine_id=str(
                package.connection.get(
                    "machine_id",
                    "remote" if package.connection_mode == "remote" else "local",
                )
            ),
            control_host=host,
            control_port=control_port,
            settings_path=str(settings_path) if settings_path else None,
            process_id=None,
            status="created",
            process_ownership=ownership,
            settings_sha256=settings_sha256,
        )

    def _close_transport(self, scene_instance_id: str) -> None:
        tunnel = self.tunnels.pop(scene_instance_id, None)
        if tunnel is not None:
            tunnel.close()
        port = self._reserved_ports.pop(scene_instance_id, None)
        if port is not None:
            self.port_manager.release(port)

    def _wait_for_backend(
        self,
        backend: SimulationBackend,
        package: MapPackage,
        host: str,
        port: int,
    ) -> None:
        timeout = float(package.launch.get("ready_timeout_sec", 90))
        deadline = time.monotonic() + timeout
        last_detail = ""
        while time.monotonic() < deadline:
            probe = backend.probe(package, host, port)
            if probe.ready:
                return
            last_detail = probe.detail
            time.sleep(1)
        raise TimeoutError(
            f"{package.control_protocol} did not become ready at {host}:{port} "
            f"within {timeout:.0f}s; last_error={last_detail}"
        )

    def _get_package(self, map_id: str) -> MapPackage:
        try:
            return self.map_packages[map_id]
        except KeyError as exc:
            raise KeyError(f"Unknown map_id: {map_id}") from exc

    @staticmethod
    def _configured_endpoint(package: MapPackage) -> tuple[str, int]:
        if package.control_protocol == "unrealcv":
            return (
                str(package.unrealcv.get("host", "127.0.0.1")),
                int(package.unrealcv.get("port", 9010)),
            )
        return (
            str(package.airsim.get("host", "127.0.0.1")),
            int(package.airsim.get("port", 41451)),
        )

    @staticmethod
    def _remote_endpoint(package: MapPackage) -> tuple[str, int]:
        config = (
            package.unrealcv
            if package.control_protocol == "unrealcv"
            else package.airsim
        )
        return (
            str(config.get("remote_host", "127.0.0.1")),
            int(
                config.get(
                    "port",
                    9010 if package.control_protocol == "unrealcv" else 41451,
                )
            ),
        )

    def _write_airsim_settings(
        self, package: MapPackage, instance_id: str, airsim_port: int
    ) -> Path:
        scene_dir = (self.run_dir / "scenes" / instance_id).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        settings_path = scene_dir / "settings.json"
        vehicle_name = package.airsim["vehicle_name"]
        camera_name = package.airsim["camera_name"]
        image_width = int(package.airsim.get("image_width", 224))
        image_height = int(package.airsim.get("image_height", 224))
        depth_width = int(package.airsim.get("depth_width", image_width))
        depth_height = int(package.airsim.get("depth_height", image_height))
        capture_settings = [
            {
                "ImageType": image_type,
                "Width": depth_width if image_type in {2, 3} else image_width,
                "Height": depth_height if image_type in {2, 3} else image_height,
                "FOV_Degrees": package.airsim.get("fov_degrees", 90),
            }
            for image_type in (0, 3, 2)
        ]
        settings = {
            "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
            "SettingsVersion": 1.2,
            "SimMode": package.airsim.get("mode", "Multirotor"),
            "ViewMode": self._view_mode(package),
            "ApiServerPort": airsim_port,
            "Vehicles": {
                vehicle_name: {
                    "VehicleType": (
                        "ComputerVision"
                        if str(package.airsim.get("mode", "")).lower()
                        == "computervision"
                        else "SimpleFlight"
                    ),
                    "AutoCreate": True,
                }
            },
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
        settings_path.write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )
        return settings_path

    def _launch_args(self, package: MapPackage) -> list[str]:
        return self.local_scene_control.launch_args(package)

    def _view_mode(self, package: MapPackage) -> str:
        if self.show_display is True:
            return "Fpv"
        if self.show_display is False:
            return "NoDisplay"
        return str(package.airsim.get("view_mode", "NoDisplay"))

    def _post_launch_wait(self, package: MapPackage) -> None:
        wait_sec = float(package.launch.get("post_launch_wait_sec", 5.0))
        if wait_sec > 0:
            time.sleep(wait_sec)

    def _wait_for_airsim_rpc(
        self,
        host: str,
        port: int,
        timeout: float,
        settings_path: str,
    ) -> None:
        """Legacy direct AirSim probe retained for the redundancy audit."""

        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                import airsim

                client = airsim.VehicleClient(
                    ip=host, port=port, timeout_value=2
                )
                if client.ping():
                    return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(
            f"AirSim RPC did not become ready: {host}:{port} within "
            f"{timeout:.0f}s; settings={settings_path}; last_error={last_error}"
        )

    @staticmethod
    def _wait_for_tcp_port(
        host: str,
        port: int,
        timeout: float,
        settings_path: str | None,
        label: str,
    ) -> None:
        """Legacy TCP readiness helper retained for the redundancy audit."""

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
            f"{label} port did not become ready: {host}:{port} within "
            f"{timeout:.0f}s; settings={settings_path}; last_error={last_error}"
        )
