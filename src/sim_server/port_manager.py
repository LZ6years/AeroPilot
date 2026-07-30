"""Allocate and release simulation service ports."""

from __future__ import annotations

import socket

from src.env_client.schemas import PortPolicy


class PortManager:
    def __init__(self, policy: PortPolicy) -> None:
        # Store the allocation policy and current reservations.
        self.policy = policy
        self._allocated: set[int] = set()

    def allocate_control_port(self) -> int:
        # Reserve the first OS-available local control or tunnel port.
        for offset in range(self.policy.max_scene_num):
            port = self.policy.control_port_start + offset
            if port not in self._allocated and self._is_available(port):
                self._allocated.add(port)
                return port
        raise RuntimeError("No simulator control ports available")

    def release(self, port: int) -> None:
        # Mark a port as available again.
        self._allocated.discard(port)

    @staticmethod
    def _is_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True
