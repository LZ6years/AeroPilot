"""Define FlyAgent's versioned public environment-client contract.

Benchmark and Runtime code may depend on the symbols in this module through
the stable ``src.env_client`` package entry point.  Simulator services,
Backends, adapters, SSH control, and process-management objects are deliberately
absent from this contract so their implementations can continue to evolve.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schemas import ActionResult, Observation, SceneInstance


ENV_CLIENT_API_VERSION = "1.0"
"""Current frozen EnvClient API version.

The compatibility policy is documented in
``docs/simulator_control/13_envclient_api_contract.md``.  In short, changing a
method signature or the meaning of a public response field requires a new
major version rather than an in-place edit.
"""

ENV_CLIENT_PUBLIC_METHODS = (
    "start_scene",
    "get_or_start_scene",
    "stop_scene",
    "reset_scene",
    "health_check",
    "get_observation",
    "execute_action",
)
"""Ordered method names that constitute EnvClient v1."""


@runtime_checkable
class EnvClient(Protocol):
    """Structural interface implemented by environment clients.

    The Protocol intentionally requires no shared base class.  A Benchmark may
    provide a compatible test or replay client, while production uses
    :class:`SimulatorEnvClient`.  Runtime checks confirm method presence;
    automated contract tests additionally lock the exact v1 signatures.
    """

    def start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        """Request one scene instance for a configured map."""

        ...

    def get_or_start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        """Reuse a running scene for the map, or start one when absent."""

        ...

    def stop_scene(self, scene_instance_id: str) -> None:
        """Release a scene according to its recorded process ownership."""

        ...

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        """Soft-reset a scene and return its current instance description."""

        ...

    def health_check(self) -> bool:
        """Return whether the environment service itself is available."""

        ...

    def get_observation(
        self, scene_instance_id: str, agent_id: str, modalities: list[str]
    ) -> Observation:
        """Fetch canonical raw modalities for one agent in one scene."""

        ...

    def execute_action(
        self,
        scene_instance_id: str,
        agent_id: str,
        action: str,
        args: dict[str, Any],
    ) -> ActionResult:
        """Execute one canonical primitive action."""

        ...
