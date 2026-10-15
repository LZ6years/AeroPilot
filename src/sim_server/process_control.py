"""Separate local process ownership from simulator protocol behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.env_client.schemas import MapPackage

from .process_manager import ProcessManager, windows_path_to_wsl


@dataclass(frozen=True)
class ProcessStartResult:
    """Return deployment ownership independently of a Backend connection."""

    ownership: str
    process_id: int | None = None
    remote_process_id: int | None = None
    settings_path: str | None = None


class SceneProcessControl(Protocol):
    """Define lifecycle operations without simulator SDK knowledge."""

    def stop(self, scene_instance_id: str) -> None: ...


class LocalSceneControl:
    """Launch and stop locally owned UE processes."""

    def __init__(
        self,
        manager: ProcessManager | None = None,
        show_display: bool | None = None,
    ) -> None:
        self.manager = manager or ProcessManager()
        self.show_display = show_display

    def start(
        self,
        package: MapPackage,
        scene_instance_id: str,
        settings_path: Path | None,
        log_path: Path,
    ) -> ProcessStartResult:
        executable = self.resolve_executable(package)
        method = str(package.launch.get("method", "direct"))
        settings_arg = str(settings_path.resolve()) if settings_path else None
        if settings_arg and method == "wsl_bash":
            settings_arg = windows_path_to_wsl(settings_arg)
        process_id = self.manager.start(
            scene_instance_id,
            executable,
            self.launch_args(package),
            method=method,
            settings_path=settings_arg,
            log_path=str(log_path),
            settings_arg_style=package.launch.get("settings_arg_style"),
        )
        return ProcessStartResult(
            ownership="local",
            process_id=process_id,
            settings_path=str(settings_path) if settings_path else None,
        )

    def stop(self, scene_instance_id: str) -> None:
        self.manager.stop(scene_instance_id)

    @staticmethod
    def resolve_executable(package: MapPackage) -> str:
        import platform

        system = platform.system().lower()
        if system not in {"windows", "linux"}:
            raise ValueError(f"Unsupported local simulator platform: {system}")
        if package.launch.get("method") == "wsl_bash" and system != "windows":
            raise ValueError("wsl_bash requires Windows; use linux_bash on Linux")
        key = system
        executable = package.launch.get(key) or package.launch.get(f"{key}_script")
        if executable is None and package.launch.get("method") == "wsl_bash":
            executable = package.launch.get("linux")
        if executable is None:
            raise ValueError(
                f"MapPackage {package.map_id} has no {key} executable"
            )
        root = Path(package.root or ".")
        path = Path(str(executable))
        return str(path if path.is_absolute() else root / path)

    def launch_args(self, package: MapPackage) -> list[str]:
        args = [str(value) for value in package.launch.get("args", [])]
        if self.show_display is True:
            return [arg for arg in args if arg != "-RenderOffscreen"]
        return args
