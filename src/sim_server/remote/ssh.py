"""Run authenticated one-shot OpenSSH and SCP commands.

No password support is provided. Host-key checking remains enabled so a map
configuration cannot silently redirect scene control to an unknown machine.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import hashlib
import re
import time
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_OBSERVER = ContextVar("ssh_operation_observer", default=None)
_DEADLINE = ContextVar("ssh_operation_deadline", default=None)


@contextmanager
def ssh_deadline(deadline):
    """Bound a sequence of lifecycle checks by the caller's cleanup deadline."""
    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


@contextmanager
def ssh_observer(callback):
    """Caller-owned persistence for sanitized operation timing, including success."""
    token = _OBSERVER.set(callback)
    try:
        yield
    finally:
        _OBSERVER.reset(token)


class SshOperationError(RuntimeError):
    """Structured command failure; credentials and command bodies stay private."""
    def __init__(self, operation, category, attempts, *, retryable=False, stderr=""):
        stderr = re.sub(r"(?i)(bearer\s+|(?:token|password|api[_-]?key)[=: ]+)[^\s]+", r"\1[redacted]", stderr)
        super().__init__(f"SSH {operation} failed: {category}: {stderr[:500]}")
        self.operation, self.stderr = operation, stderr[:500]
        self.category, self.retryable, self.attempts = category, retryable, attempts

    def to_dict(self):
        return {"category": self.category, "retryable": self.retryable, "operation": self.operation,
                "stderr": self.stderr, "attempt_details": self.attempts}


def _category(stderr, returncode):
    value = stderr.lower()
    if any(x in value for x in ("permission denied", "authentication failed")):
        return "authentication", False
    if any(x in value for x in ("host key verification failed", "remote host identification has changed")):
        return "host_key", False
    if returncode == 255:
        return "connection", True
    return "remote_exit", False


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

    def _execute(self, command, operation, timeout_sec, *, safe=False):
        started = time.monotonic()
        attempts = []
        category, retryable, stderr = "command_timeout", True, "operation deadline exceeded"
        for index in range(3 if safe else 1):
            remaining = 100 - (time.monotonic()-started)
            if _DEADLINE.get() is not None:
                remaining = min(remaining, _DEADLINE.get()-time.monotonic())
            if remaining <= 0:
                break
            before = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                        timeout=min(timeout_sec, remaining), check=False)
                category, retryable = _category(result.stderr or "", result.returncode)
                if result.returncode == 0:
                    attempts.append({"number": index+1, "category": "success",
                                     "elapsed_seconds": time.monotonic()-before})
                    if _OBSERVER.get():
                        _OBSERVER.get()({"operation": operation, "attempts": attempts, "status": "success"})
                    return result.stdout
                stderr = (result.stderr or result.stdout or "")[:1000]
            except subprocess.TimeoutExpired:
                category, retryable, stderr = "command_timeout", True, "command deadline exceeded"
            attempts.append({"number": index+1, "category": category,
                             "elapsed_seconds": time.monotonic()-before})
            if not retryable or not safe or index == 2:
                break
            remaining = max(0, 100-(time.monotonic()-started))
            if _DEADLINE.get() is not None:
                remaining = min(remaining, max(0, _DEADLINE.get()-time.monotonic()))
            time.sleep(min((2, 5)[index], remaining))
        error = SshOperationError(operation, category, attempts, retryable=retryable, stderr=stderr)
        if _OBSERVER.get():
            _OBSERVER.get()({**error.to_dict(), "status": "failed"})
        raise error

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
        safe = remote_args[:2] == ["mkdir", "-p"] or (
            len(remote_args) > 2 and remote_args[0] == "python3" and
            remote_args[1].endswith("flyagent_remote_worker.py") and remote_args[2] in {"probe", "status", "start", "stop"})
        # Worker start/stop are scene-locked, identity-verified operations.
        operation = remote_args[2] if safe and remote_args[0] == "python3" else remote_args[0]
        return self._execute(command, operation, timeout_sec, safe=safe)

    def upload(
        self,
        target: SshTarget,
        local_path: str | Path,
        remote_path: str,
        timeout_sec: float = 30,
    ) -> None:
        temporary = remote_path + ".upload-" + uuid.uuid4().hex
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
            f"{target.destination}:{temporary}",
        ]
        self._execute(command, "upload", timeout_sec, safe=True)
        digest = hashlib.sha256(Path(local_path).read_bytes()).hexdigest()
        # Exact content check makes an acknowledged or repeated commit safe.
        code = ("from pathlib import Path; import hashlib,os; "
                f"p=Path({temporary!r}).expanduser(); q=Path({remote_path!r}).expanduser(); "
                f"assert hashlib.sha256((p if p.exists() else q).read_bytes()).hexdigest()=={digest!r}; "
                "os.replace(p,q) if p.exists() else None")
        self._execute(["ssh", *self._options(target), target.destination,
                       shlex.join(["python3", "-c", code])], "upload_commit", timeout_sec, safe=True)

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
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
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
