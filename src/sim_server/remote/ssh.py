"""Run authenticated one-shot OpenSSH and SCP commands.

No password support is provided. Host-key checking remains enabled so a map
configuration cannot silently redirect scene control to an unknown machine.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SshTarget:
    host: str
    user: str
    key_path: str
    port: int = 22

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


class SshCommandRunner:
    """Execute remote commands through the system OpenSSH binaries."""

    def run(
        self,
        target: SshTarget,
        remote_args: list[str],
        timeout_sec: float = 30,
    ) -> str:
        command = [
            "ssh",
            *self._options(target),
            target.destination,
            shlex.join(remote_args),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SSH command failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.stdout

    def upload(
        self,
        target: SshTarget,
        local_path: str | Path,
        remote_path: str,
        timeout_sec: float = 30,
    ) -> None:
        command = [
            "scp",
            "-q",
            "-P",
            str(target.port),
            "-i",
            self._expanded_key(target),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            str(local_path),
            f"{target.destination}:{remote_path}",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SCP upload failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

    def download(
        self,
        target: SshTarget,
        remote_path: str,
        local_path: str | Path,
        timeout_sec: float = 30,
    ) -> None:
        """Download a remote template without changing the source file."""

        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "scp",
            "-q",
            "-P",
            str(target.port),
            "-i",
            self._expanded_key(target),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            f"{target.destination}:{remote_path}",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SCP download failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

    def _options(self, target: SshTarget) -> list[str]:
        return [
            "-p",
            str(target.port),
            "-i",
            self._expanded_key(target),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
        ]

    @staticmethod
    def _expanded_key(target: SshTarget) -> str:
        key_path = Path(os.path.expandvars(target.key_path)).expanduser()
        if not key_path.is_file():
            raise FileNotFoundError(f"SSH private key not found: {key_path}")
        return str(key_path)
