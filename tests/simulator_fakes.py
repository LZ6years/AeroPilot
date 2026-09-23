"""Provide explicit in-memory simulator Backends for automated tests.

Production ``SimulationServer`` instances always use real protocol adapters.
Tests that exercise Runtime or EnvClient orchestration inject this Registry so
they do not require AirSim, UnrealCV, SSH, or a packaged UE process.
"""

from __future__ import annotations

from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.base import BackendProbe
from src.sim_server.server import SimulationServer
from src.sim_server.execution import SimulationExecutionMode
from tests.fake_uav import FakeUavPrimitiveAdapter
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend


def in_memory_backend_registry() -> BackendRegistry:
    """Return fresh per-scene in-memory AirSim and UnrealCV factories."""

    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: FakeAirSimBackend(
            action_config=context.action_config,
            artifact_dir=context.artifact_dir,
            save_artifacts=context.save_artifacts,
        ),
    )
    registry.register(
        "unrealcv",
        lambda context: FakeUnrealCvBackend(
            action_config=context.action_config,
        ),
    )
    return registry


class FakeAirSimBackend(AirSimBackend):
    def __init__(self, **kwargs):
        super().__init__(uav=kwargs.pop("uav", None) or FakeUavPrimitiveAdapter(), **kwargs)

    def probe(self, package, host, port):
        return BackendProbe(True, "explicit test double")


class FakeUnrealCvBackend(UnrealCvBackend):
    def __init__(self, **kwargs):
        uav = kwargs.pop("uav", None) or FakeUavPrimitiveAdapter()
        uav.connect()
        uav.set_pose(0, 0, 3)
        super().__init__(uav=uav, **kwargs)

    def probe(self, package, host, port):
        return BackendProbe(True, "explicit test double")


def fake_simulation_server(*args, **kwargs):
    """Inject only in tests; never launch a process or probe a network endpoint."""
    kwargs["backend_registry"] = in_memory_backend_registry()
    kwargs["execution_mode"] = SimulationExecutionMode.ATTACH
    return SimulationServer(*args, **kwargs)


class FakeUavFlowScenario:
    def prepare_episode(self, scene, payload):
        pass

    def cleanup_episode(self, scene, payload):
        pass

    def label_map_for(self, case_id):
        return {}

    def diagnostics_for(self, case_id):
        return {}
