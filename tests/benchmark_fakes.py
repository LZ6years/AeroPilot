"""Explicit simulator/scenario injection for Runner integration tests only."""

from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner as _Aerial
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner as _Flow
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner as _Travel
from evaluation.benchmarks.core import NoopScenarioController
from tests.simulator_fakes import fake_simulation_server, FakeUavFlowScenario


class AerialVlnBenchmarkRunner(_Aerial):
    def __init__(self, config, **kwargs):
        kwargs.setdefault("simulation_server_factory", fake_simulation_server)
        super().__init__(config, **kwargs)


class UavFlowBenchmarkRunner(_Flow):
    def __init__(self, config, **kwargs):
        kwargs.setdefault("simulation_server_factory", fake_simulation_server)
        kwargs.setdefault("scenario_controller_factory", FakeUavFlowScenario)
        super().__init__(config, **kwargs)


class TravelUavBenchmarkRunner(_Travel):
    def __init__(self, config, **kwargs):
        kwargs.setdefault("simulation_server_factory", fake_simulation_server)
        kwargs.setdefault("scenario_controller_factory", NoopScenarioController)
        super().__init__(config, **kwargs)
