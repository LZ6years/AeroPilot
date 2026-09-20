"""Manage remote UE ownership, settings transfer, Worker calls, and tunnels."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.env_client.schemas import MapPackage

from .ssh import SshCommandRunner, SshTarget
from .tunnel import SshTunnel


@dataclass(frozen=True)
class RemotePaths:
    """Scene-scoped paths under the remote user's home directory."""

    directory: str
    worker: str
    settings: str
    spec: str
    registry: str
    log: str


class RemoteSceneControl:
    """Use one-shot SSH commands; no daemon or FlyAgent installation is needed."""

    def __init__(self, runner: SshCommandRunner | None = None) -> None:
        self.runner = runner or SshCommandRunner()

    def preflight(self, package: MapPackage) -> dict[str, Any]:
        """Validate SSH, Python, and optional executable before side effects."""

        target = self.target(package)
        args = [
            "python3",
            "-c",
            "import json,sys; print(json.dumps({'python':sys.version.split()[0]}))",
        ]
        output = self.runner.run(
            target,
            args,
            timeout_sec=float(package.ssh.get("connect_timeout_sec", 10)),
        )
        result = self._json_result(output, require_ok=False)
        executable = package.launch.get("linux_executable")
        if executable and not package.launch.get("manual", False):
            self.runner.run(
                target,
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        f"sys.exit(0 if Path({str(executable)!r}).is_file() else 2)"
                    ),
                ],
            )
        return result

    def open_tunnel(
        self,
        package: MapPackage,
        local_port: int,
        remote_host: str,
        remote_port: int,
        log_path: Path | None = None,
    ) -> SshTunnel:
        return SshTunnel.open(
            self.runner,
            self.target(package),
            local_port,
            remote_host,
            remote_port,
            timeout_sec=float(package.ssh.get("connect_timeout_sec", 10)),
            log_path=log_path,
        )

    def download_settings_template(
        self,
        package: MapPackage,
        destination: Path,
    ) -> Path:
        source = str(package.airsim["settings_template"])
        self.runner.download(
            self.target(package),
            source,
            destination,
            timeout_sec=float(package.launch.get("ssh_transfer_timeout_sec", 30)),
        )
        return destination

    def remote_port_status(
        self,
        package: MapPackage,
        scene_instance_id: str,
        host: str,
        port: int,
        local_scene_dir: Path,
    ) -> dict[str, Any]:
        paths = self._paths(package, scene_instance_id)
        self._prepare_worker(package, paths, local_scene_dir)
        return self.probe_prepared_port(package, scene_instance_id, host, port)

    def probe_prepared_port(self, package, scene_instance_id, host, port):
        """Read endpoint status without re-uploading the worker on each poll."""
        paths = self._paths(package, scene_instance_id)
        output = self.runner.run(
            self.target(package),
            [
                "python3",
                paths.worker,
                "probe",
                "--host",
                host,
                "--port",
                str(port),
            ],
        )
        return self._json_result(output)

    def start(
        self,
        package: MapPackage,
        scene_instance_id: str,
        settings_path: Path | None,
        local_scene_dir: Path,
    ) -> dict[str, Any]:
        target = self.target(package)
        paths = self._paths(package, scene_instance_id)
        self._prepare_worker(package, paths, local_scene_dir)
        if settings_path is not None:
            self.runner.upload(target, settings_path, paths.settings)

        spec = {
            "scene_instance_id": scene_instance_id,
            "executable": str(package.launch["linux_executable"]),
            "args": list(package.launch.get("args", [])),
            "env": dict(package.launch.get("env", {})),
            "settings_path": paths.settings if settings_path is not None else None,
            "settings_arg_style": str(
                package.launch.get(
                    "settings_arg_style",
                    "double_dash" if settings_path is not None else "none",
                )
            ),
            "launch_method": str(package.launch.get("method", "linux_bash")),
            "log_path": paths.log,
            "registry_path": paths.registry,
        }
        local_spec = local_scene_dir / "remote-scene-spec.json"
        local_spec.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.runner.upload(target, local_spec, paths.spec)
        output = self.runner.run(
            target,
            ["python3", paths.worker, "start", "--spec", paths.spec],
            timeout_sec=float(package.launch.get("ssh_start_timeout_sec", 30)),
        )
        result = self._json_result(output)
        result["remote_settings_path"] = (
            paths.settings if settings_path is not None else None
        )
        return result

    def stop(
        self, package: MapPackage, scene_instance_id: str
    ) -> dict[str, Any]:
        paths = self._paths(package, scene_instance_id)
        output = self.runner.run(
            self.target(package),
            [
                "python3",
                paths.worker,
                "stop",
                "--registry",
                paths.registry,
            ],
            timeout_sec=float(package.launch.get("ssh_stop_timeout_sec", 20)),
        )
        return self._json_result(output)

    def status(
        self, package: MapPackage, scene_instance_id: str
    ) -> dict[str, Any]:
        paths = self._paths(package, scene_instance_id)
        output = self.runner.run(
            self.target(package),
            [
                "python3",
                paths.worker,
                "status",
                "--registry",
                paths.registry,
            ],
        )
        return self._json_result(output)

    def _prepare_worker(
        self,
        package: MapPackage,
        paths: RemotePaths,
        local_scene_dir: Path,
    ) -> None:
        target = self.target(package)
        self.runner.run(target, ["mkdir", "-p", paths.directory])
        worker_source = Path(__file__).with_name("worker.py")
        if not worker_source.is_file():
            raise FileNotFoundError(f"Remote Worker source not found: {worker_source}")
        self.runner.upload(target, worker_source, paths.worker)

    @staticmethod
    def target(package: MapPackage) -> SshTarget:
        return SshTarget(
            host=str(package.ssh["host"]),
            user=str(package.ssh["user"]),
            key_path=str(package.ssh["key_path"]),
            port=int(package.ssh.get("port", 22)),
        )

    @staticmethod
    def _paths(package: MapPackage, scene_instance_id: str) -> RemotePaths:
        value = str(package.launch.get("remote_run_root", ".flyagent/scenes"))
        root = value[2:] if value.startswith("~/") else value
        directory = posixpath.join(root, scene_instance_id)
        return RemotePaths(
            directory=directory,
            worker=posixpath.join(directory, "flyagent_remote_worker.py"),
            settings=posixpath.join(directory, "settings.json"),
            spec=posixpath.join(directory, "scene-spec.json"),
            registry=posixpath.join(directory, "scene-registry.json"),
            log=posixpath.join(directory, "ue.log"),
        )

    @staticmethod
    def _json_result(
        output: str, *, require_ok: bool = True
    ) -> dict[str, Any]:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Remote Worker returned no JSON result")
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Remote Worker returned invalid JSON: {lines[-1]}"
            ) from exc
        if require_ok and not result.get("ok", False):
            raise RuntimeError(str(result.get("error", "remote Worker failed")))
        return result
