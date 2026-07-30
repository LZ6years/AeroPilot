"""Own a local OpenSSH forwarding process for one remote scene."""

from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass

from .ssh import SshCommandRunner, SshTarget


@dataclass
class SshTunnel:
    """A process-scoped SSH local forward that can be closed deterministically."""

    process: subprocess.Popen
    local_host: str
    local_port: int

    @classmethod
    def open(
        cls,
        runner: SshCommandRunner,
        target: SshTarget,
        local_port: int,
        remote_host: str,
        remote_port: int,
        timeout_sec: float = 10,
    ) -> "SshTunnel":
        command = [
            "ssh",
            *runner._options(target),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:{remote_host}:{remote_port}",
            target.destination,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        tunnel = cls(process, "127.0.0.1", local_port)
        tunnel._wait_until_listening(timeout_sec)
        return tunnel

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def _wait_until_listening(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                stderr = self.process.stderr.read().strip() if self.process.stderr else ""
                raise RuntimeError(
                    f"SSH tunnel exited before ready ({return_code}): {stderr}"
                )
            try:
                with socket.create_connection(
                    (self.local_host, self.local_port), timeout=0.2
                ):
                    return
            except OSError:
                time.sleep(0.1)
        self.close()
        raise TimeoutError(
            f"SSH tunnel did not listen on {self.local_host}:{self.local_port} "
            f"within {timeout_sec:.0f}s"
        )
