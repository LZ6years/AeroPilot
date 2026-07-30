"""Provide explicit in-memory simulator Backends for automated tests.

Production ``SimulationServer`` instances always use real protocol adapters.
Tests that exercise Runtime or EnvClient orchestration inject this Registry so
they do not require AirSim, UnrealCV, SSH, or a packaged UE process.
"""

from __future__ import annotations

from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend


def in_memory_backend_registry() -> BackendRegistry:
    """Return fresh per-scene in-memory AirSim and UnrealCV factories."""

    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: AirSimBackend(
            use_real_uav=False,
            action_config=context.action_config,
            artifact_dir=context.artifact_dir,
            save_artifacts=context.save_artifacts,
        ),
    )
    registry.register(
        "unrealcv",
        lambda context: UnrealCvBackend(
            use_real_client=False,
            action_config=context.action_config,
        ),
    )
    return registry
