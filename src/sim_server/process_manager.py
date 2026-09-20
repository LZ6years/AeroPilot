"""Start and stop external scene processes."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


class ProcessManager:
    def __init__(self) -> None:
        # Track launched subprocesses by scene instance id.
        self._processes: dict[str, subprocess.Popen] = {}

    def start(
        self,
        instance_id: str,
        executable: str,
        args: list[str],
        method: str = "direct",
        settings_path: str | None = None,
        log_path: str | None = None,
    ) -> int:
        # Launch a scene executable and remember its process handle.
        command = self.build_command(executable, args, method, settings_path)
        stdout = subprocess.DEVNULL
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            stdout = open(log_path, "ab")
        cwd = None if method == "wsl_bash" else str(Path(executable).resolve().parent)
        try:
            proc = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=cwd)
        finally:
            if log_path is not None:
                stdout.close()
        self._processes[instance_id] = proc
        return proc.pid

    def stop(self, instance_id: str) -> None:
        # Terminate a remembered scene process if it is still alive.
        proc = self._processes.get(instance_id)
        if proc is None:
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

    def build_command(
        self, executable: str, args: list[str], method: str = "direct", settings_path: str | None = None
    ) -> list[str]:
        # Build a platform-aware process command without launching it.
        path = Path(executable)
        if not path.exists():
            raise FileNotFoundError(
                f"Scene executable not found: {executable}"
                + (f" (WSL path: {windows_path_to_wsl(executable)})" if method == "wsl_bash" else "")
            )
        if method == "wsl_bash":
            full_args = [*args]
            if settings_path:
                full_args.extend(["--settings", settings_path])
            wsl_executable = windows_path_to_wsl(executable)
            wsl_args = [windows_path_to_wsl(arg) if _looks_like_windows_path(arg) else arg for arg in full_args]
            shell_command = " ".join(shlex.quote(part) for part in [wsl_executable, *wsl_args])
            return ["wsl.exe", "bash", "-lc", shell_command]
        full_args = [*args]
        if settings_path:
            full_args.append(f"-settings={settings_path}")
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
