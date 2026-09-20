"""Own a local OpenSSH forwarding process for one remote scene."""

from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import tempfile

from .ssh import SshCommandRunner, SshTarget


@dataclass
class SshTunnel:
    """A process-scoped SSH local forward that can be closed deterministically."""

    process: subprocess.Popen
    local_host: str
    local_port: int
    log_file: BinaryIO | None = None
    log_path: Path | None = None

    @classmethod
    def open(
        cls,
        runner: SshCommandRunner,
        target: SshTarget,
        local_port: int,
        remote_host: str,
        remote_port: int,
        timeout_sec: float = 10,
        log_path: Path | None = None,
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
        # A file cannot fill an unread PIPE and stall OpenSSH. Standalone callers
        # may use a temporary file; managed scenes always supply a durable path.
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a+b") if log_path else tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=log_file,
            )
        except Exception:
            log_file.close()
            raise
        tunnel = cls(process, "127.0.0.1", local_port, log_file, log_path)
        try:
            tunnel._wait_until_listening(timeout_sec)
        except Exception:
            tunnel.close()
            raise
        return tunnel

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self.log_file is not None:
                self.log_file.close()

    def check_alive(self) -> None:
        code = self.process.poll()
        if code is not None:
            tail = ""
            if self.log_file is not None and not self.log_file.closed:
                self.log_file.seek(0, 2)
                self.log_file.seek(max(0, self.log_file.tell() - 4096))
                tail = self.log_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SSH tunnel exited ({code}): {tail}")

    def _wait_until_listening(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.check_alive()
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
