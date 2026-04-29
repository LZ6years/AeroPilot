"""Start and stop external scene processes."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ProcessManager:
    def __init__(self) -> None:
        # Track launched subprocesses by scene instance id.
        self._processes: dict[str, subprocess.Popen] = {}

    def start(self, instance_id: str, executable: str, args: list[str]) -> int:
        # Launch a scene executable and remember its process handle.
        path = Path(executable)
        if not path.exists():
            raise FileNotFoundError(f"Scene executable not found: {executable}")
        proc = subprocess.Popen([str(path), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._processes[instance_id] = proc
        return proc.pid

    def stop(self, instance_id: str) -> None:
        # Terminate a remembered scene process if it is still alive.
        proc = self._processes.pop(instance_id, None)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
