"""Start and stop external scene processes."""

from __future__ import annotations

import os
import json
import uuid
import platform
import shlex
import signal
import subprocess
import time
from pathlib import Path


class ProcessManager:
    def __init__(self, *, terminate_timeout_sec: float = 10, kill_timeout_sec: float = 5) -> None:
        # Track launched subprocesses by scene instance id.
        self._processes: dict[str, subprocess.Popen] = {}
        # Linux launchers may exit before their UE children. Retain group ownership
        # independently of the Popen leader, until every live member has exited.
        self._process_groups: dict[str, int] = {}
        self.terminate_timeout_sec = terminate_timeout_sec
        self.kill_timeout_sec = kill_timeout_sec

    def start(
        self,
        instance_id: str,
        executable: str,
        args: list[str],
        method: str = "direct",
        settings_path: str | None = None,
        log_path: str | None = None,
        settings_arg_style: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> int:
        # Launch a scene executable and remember its process handle.
        if instance_id in self._processes:
            raise RuntimeError(f"Scene {instance_id} already owns a process; stop it before restarting")
        command = self.build_command(executable, args, method, settings_path, settings_arg_style)
        stdout = subprocess.DEVNULL
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            stdout = open(log_path, "ab")
        cwd = None if method == "wsl_bash" else str(Path(executable).resolve().parent)
        try:
            linux = platform.system() == "Linux"
            launch_options = {}
            token = uuid.uuid4().hex
            if environment or linux:
                launch_options["env"] = {**os.environ, **(environment or {})}
                if linux:
                    launch_options["env"]["FLYAGENT_PROCESS_OWNER"] = token
            proc = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=cwd,
                                    start_new_session=linux, **launch_options)
        finally:
            if log_path is not None:
                stdout.close()
        self._processes[instance_id] = proc
        if linux:
            self._process_groups[instance_id] = proc.pid
            if log_path:
                # Persist independently of scene readiness, so a supervisor can
                # recover a launcher that died during startup as well.
                receipt = Path(log_path).with_suffix(".process.json")
                receipt.write_text(json.dumps({"scene": instance_id, "pgid": proc.pid,
                    "owner_token": token, "settings_path": settings_path}), encoding="utf-8")
        return proc.pid

    def stop(self, instance_id: str) -> None:
        # Terminate a remembered scene process if it is still alive.
        proc = self._processes.get(instance_id)
        if proc is None:
            return
        group = self._process_groups.get(instance_id)
        if group is not None:
            self._stop_group(proc, group)
            # A failed cleanup deliberately keeps both ownership records for retry.
            self._process_groups.pop(instance_id)
            self._processes.pop(instance_id)
            return
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._processes.pop(instance_id, None)

    def _stop_group(self, proc: subprocess.Popen, group: int) -> None:
        """Terminate only a session created by this manager, never a name match."""
        if group <= 1 or group == os.getpgrp():
            raise RuntimeError(f"Refusing to signal unsafe process group {group}")
        for sig, timeout in ((signal.SIGTERM, self.terminate_timeout_sec),
                             (signal.SIGKILL, self.kill_timeout_sec)):
            proc.poll()  # Reap our own leader even if the launcher exited early.
            if not self._group_alive(group):
                return
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + timeout
            while True:
                proc.poll()
                if not self._group_alive(group):
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        raise RuntimeError(f"Owned process group {group} still has live members after SIGKILL")

    @staticmethod
    def _group_alive(group: int) -> bool:
        """Inspect Linux members, excluding exited zombies awaiting their reaper.

        killpg(0) alone reports orphan zombies as alive indefinitely on some
        hosts. Unreadable member state is an error, not proof of clean shutdown.
        """
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / 'stat').read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) == group and fields[0] not in {'Z', 'X'}:
                return True
        return False

    def build_command(
        self, executable: str, args: list[str], method: str = "direct", settings_path: str | None = None,
        settings_arg_style: str | None = None,
    ) -> list[str]:
        # Build a platform-aware process command without launching it.
        system = platform.system()
        if system not in {'Windows', 'Linux'}:
            raise ValueError(f"Unsupported local simulator platform: {system}")
        if method not in {'direct', 'linux_bash', 'wsl_bash'}:
            raise ValueError(f"Unsupported local launch method: {method}")
        if method == 'wsl_bash' and system != 'Windows':
            raise ValueError('wsl_bash requires Windows; use linux_bash on Linux')
        if method == 'linux_bash' and system != 'Linux':
            raise ValueError('linux_bash requires Linux; use wsl_bash on Windows')
        style = settings_arg_style or ('double_dash' if method == 'wsl_bash' else 'single_dash_equals')
        if style not in {'double_dash', 'single_dash_equals'}:
            raise ValueError(f"Unsupported settings_arg_style: {style}")
        path = Path(executable).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Scene executable not found: {executable}"
                + (f" (WSL path: {windows_path_to_wsl(executable)})" if method == "wsl_bash" else "")
            )
        full_args = [*args]
        if settings_path:
            if style == 'double_dash':
                full_args.extend(['--settings', settings_path])
            else:
                full_args.append(f'-settings={settings_path}')
        if method == "wsl_bash":
            wsl_executable = windows_path_to_wsl(executable)
            wsl_args = [windows_path_to_wsl(arg) if _looks_like_windows_path(arg) else arg for arg in full_args]
            shell_command = " ".join(shlex.quote(part) for part in [wsl_executable, *wsl_args])
            return ["wsl.exe", "bash", "-lc", shell_command]
        if method == 'linux_bash':
            return ['bash', str(path), *full_args]
        return [str(path), *full_args]


def windows_path_to_wsl(path: str) -> str:
    # Convert a Windows absolute path into the matching /mnt/<drive> WSL path.
    normalized = path.replace("\\", "/")
    if len(normalized) >= 3 and normalized[1:3] == ":/":
        drive = normalized[0].lower()
        return f"/mnt/{drive}/{normalized[3:]}"
    return normalized


def _looks_like_windows_path(value: str) -> bool:
    # Detect Windows drive paths before passing arguments to WSL.
    return len(value) >= 3 and value[1:3] in {":\\", ":/"}
