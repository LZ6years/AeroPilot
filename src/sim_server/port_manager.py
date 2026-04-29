"""Allocate and release simulation service ports."""

from __future__ import annotations

from src.env_client.schemas import PortPolicy


class PortManager:
    def __init__(self, policy: PortPolicy) -> None:
        # Store the allocation policy and current reservations.
        self.policy = policy
        self._allocated: set[int] = set()

    def allocate_airsim_port(self) -> int:
        # Pick the first free AirSim port in the configured range.
        for offset in range(self.policy.max_scene_num):
            port = self.policy.airsim_port_start + offset
            if port not in self._allocated:
                self._allocated.add(port)
                return port
        raise RuntimeError("No AirSim ports available")

    def release(self, port: int) -> None:
        # Mark a port as available again.
        self._allocated.discard(port)
