"""Public Runtime lifecycle regression migrated from the retired Phase harness.

The native loop/protocol/Safety/knowledge/debug/state-isolation coverage lives in
test_unified_tool_loop and test_builtin_control_tools. These tests retain the
initial-placement and preflight backend integration checks with real logging.
"""
import json
import pytest
from src.adapters.uav_primitives import PrimitiveResult
from tests.support.fake_uav import FakeUavPrimitiveAdapter
from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage, SceneCapabilities
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.task_runtime import TaskRuntime
from src.runtime.factory import RuntimeFactory
from src.plugins import PluginRegistry
from tests.support.simulator_fakes import FakeAirSimBackend as AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.server import SimulationServer
from tests.support.simulator_fakes import in_memory_backend_registry
from tests.support.benchmark import CompleteProbePlanner

LAMP_TASK = "Observe the lamp"
ENTRANCE_TASK = "Observe the entrance"

class TerminalPlanner(CompleteProbePlanner):
    def __init__(self):
        self.calls = 0

    def plan_turn(self, *args, **kwargs):
        self.calls += 1
        return super().plan_turn(*args, **kwargs)

@pytest.fixture(autouse=True)
def _use_explicit_in_memory_simulator_backends(monkeypatch):
    monkeypatch.setattr(BackendRegistry, "defaults",
                        classmethod(lambda cls: in_memory_backend_registry()))

def make_package() -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "connection": {"mode": "local"},
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )



def load_event_records(run_dir):
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    debug_path = run_dir / "debug" / "debug_events.jsonl"
    if debug_path.exists():
        events.extend(json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines())
    return sorted(events, key=lambda record: str(record.get("timestamp", "")))


def build_runtime(client, recorder, planner):
    """Use the production composition root with a deterministic test planner."""

    return RuntimeFactory(plugin_registry=PluginRegistry()).create_task_runtime(
        client,
        recorder,
        planner=planner,
        shutdown_policy="none",
    )



def test_task_runtime_applies_initial_pose_before_first_configured_observation(tmp_path):
    run_dir = tmp_path / "task_execution_options"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = build_runtime(SimulatorEnvClient(server), recorder, TerminalPlanner())
    options = TaskExecutionOptions(
        observation_modalities=("rgb", "pose"),
        initial_pose=InitialPose(4.0, 5.0, 12.0, yaw_deg=35.0),
    )

    state = runtime.run_task("m1", LAMP_TASK, execution_options=options)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    initial_observation = next(
        record["payload"]
        for record in events
        if record["event"] == "initial_observation_updated"
    )
    actions_text = (run_dir / "memory/actions.jsonl").read_text(encoding="utf-8")
    latest_checkpoint = json.loads(
        (run_dir / "contexts/latest_checkpoint.json").read_text(encoding="utf-8")
    )
    flight_memory = json.loads(
        (run_dir / "state/flight_memory.json").read_text(encoding="utf-8")
    )

    assert event_names.index("episode_initial_pose_completed") < event_names.index(
        "preflight_hover_completed"
    )
    assert event_names.index("episode_initial_pose_completed") < event_names.index(
        "initial_observation_updated"
    )
    assert initial_observation["pose"]["x"] == 4.0
    assert initial_observation["pose"]["z"] == 12.0
    assert initial_observation["modalities"] == ["rgb", "pose"]
    assert state.observation_modalities == ("rgb", "pose")
    assert state.latest_observation["modalities"] == ["rgb", "pose"]
    assert "depth" not in state.latest_observation.get("frames", {})
    motion = next(
        fragment for fragment in latest_checkpoint["fragments"]
        if fragment["name"] == "motion"
    )
    assert motion["content"]["observation_policy"]["modalities"] == ["rgb", "pose"]
    assert "uav.set_pose" not in actions_text
    assert flight_memory["reset_epoch"] == 1



def test_task_runtime_reapplies_initial_pose_when_scene_is_reused(tmp_path):
    run_dir = tmp_path / "reused_episode_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = build_runtime(SimulatorEnvClient(server), recorder, TerminalPlanner())

    first_state = runtime.run_task(
        "m1",
        LAMP_TASK,
        scene_policy="reuse_running",
        stop_scene_on_finish=False,
        execution_options=TaskExecutionOptions(
            initial_pose=InitialPose(1.0, 2.0, 3.0)
        ),
    )
    second_state = runtime.run_task(
        "m1",
        ENTRANCE_TASK,
        scene_policy="reuse_running",
        stop_scene_on_finish=False,
        execution_options=TaskExecutionOptions(
            initial_pose=InitialPose(10.0, 20.0, 30.0)
        ),
    )
    runtime.finish_reused_scene("m1", second_state)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    flight_memory = json.loads(
        (run_dir / "state/flight_memory.json").read_text(encoding="utf-8")
    )

    assert first_state.completed and second_state.completed
    assert event_names.count("control_connected") == 1
    assert event_names.count("episode_initial_pose_completed") == 2
    assert "episode_initialization_stabilization_completed" in event_names
    assert second_state.latest_observation["pose"]["x"] == 10.0
    assert second_state.latest_observation["pose"]["z"] == 30.0
    assert flight_memory["reset_epoch"] == 2



def test_task_runtime_rejects_unsupported_observation_modalities_before_planning(
    tmp_path,
):
    run_dir = tmp_path / "unsupported_modalities"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = TerminalPlanner()
    runtime = build_runtime(SimulatorEnvClient(server), recorder, planner)

    with pytest.raises(RuntimeError, match="does not support.*mask"):
        runtime.run_task(
            "m1",
            LAMP_TASK,
            execution_options=TaskExecutionOptions(
                observation_modalities=("mask", "pose")
            ),
        )
    recorder.close()

    assert planner.calls == 0
    assert not (run_dir / "memory/observations.jsonl").exists()



class _NoSetPoseBackend(AirSimBackend):
    def describe_capabilities(self, package):
        capabilities = super().describe_capabilities(package)
        return SceneCapabilities(
            protocol=capabilities.protocol,
            control_mode=capabilities.control_mode,
            supported_actions=tuple(
                action
                for action in capabilities.supported_actions
                if action != "set_pose"
            ),
            observation_modalities=capabilities.observation_modalities,
            requires_takeoff=capabilities.requires_takeoff,
            supports_soft_reset=capabilities.supports_soft_reset,
        )



class _FailingSetPoseAdapter(FakeUavPrimitiveAdapter):
    def set_pose(
        self,
        x,
        y,
        z,
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        ignore_collision=True,
    ):
        return PrimitiveResult(False, "injected set_pose failure")



def test_task_runtime_rejects_initial_pose_without_backend_capability(tmp_path):
    run_dir = tmp_path / "unsupported_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: _NoSetPoseBackend(
            action_config=context.action_config,
        ),
    )
    server = SimulationServer(
        [make_package()],
        run_dir=run_dir,
        backend_registry=registry,
    )
    planner = TerminalPlanner()
    runtime = build_runtime(SimulatorEnvClient(server), recorder, planner)

    with pytest.raises(RuntimeError, match="does not support set_pose"):
        runtime.run_task(
            "m1",
            LAMP_TASK,
            execution_options=TaskExecutionOptions(
                initial_pose=InitialPose(1.0, 2.0, 3.0)
            ),
        )
    recorder.close()

    assert planner.calls == 0
    assert "episode_initial_pose_started" not in [
        record["event"] for record in load_event_records(run_dir)
    ]



def test_task_runtime_reports_initial_pose_action_failure_before_planning(tmp_path):
    run_dir = tmp_path / "failed_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: AirSimBackend(
            uav=_FailingSetPoseAdapter(),
            action_config=context.action_config,
        ),
    )
    server = SimulationServer(
        [make_package()],
        run_dir=run_dir,
        backend_registry=registry,
    )
    planner = TerminalPlanner()
    runtime = build_runtime(SimulatorEnvClient(server), recorder, planner)

    with pytest.raises(RuntimeError, match="Episode initial pose failed"):
        runtime.run_task(
            "m1",
            LAMP_TASK,
            execution_options=TaskExecutionOptions(
                initial_pose=InitialPose(1.0, 2.0, 3.0)
            ),
        )
    recorder.close()

    event_names = [record["event"] for record in load_event_records(run_dir)]
    assert planner.calls == 0
    assert "episode_initial_pose_failed" in event_names
    assert "task_terminal" in event_names and "scene_stopped" in event_names
    actions = [json.loads(line) for line in (run_dir / "memory/actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(action["tool"] == "uav.hover" and action["result"]["ok"] for action in actions)
