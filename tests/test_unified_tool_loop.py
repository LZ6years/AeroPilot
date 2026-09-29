"""Phase-free loop integration with real dispatch/memory and an in-memory UAV.

The migration harness binds dependencies directly, without the legacy parser
constructor or a simulator. It exercises the production loop and physical batch
implementation; it does not emulate their ordering or ToolResult construction.
"""

import copy
from types import SimpleNamespace

import pytest

from src.agent.mission_state import MissionState
from src.agent.safety import SafetySupervisor
from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn
from src.env_client import ActionResult, Observation
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.registry import ContextModuleRegistry
from src.plugins import PluginRegistry
from src.plugins.contracts.skill import Skill
from src.plugins.skills.catalog import SkillCatalog
from src.plugins.skills.knowledge_service import KnowledgeService
from src.agent.config import ModelConfig
from src.runtime.context import ContextCheckpointService
from src.runtime.task import RuntimeJournal
from src.runtime.task.loop import InstructionLoop, TaskLoopGuard
from src.runtime.task.physical_window import PhysicalWindowExecutor
from src.runtime.task.session import InstructionSession
from src.runtime.task.turn_executor import TurnExecutor
from src.runtime.control.actions.task_plan import apply_plan_update
from src.runtime.middleware import ToolTelemetryMiddleware
from src.runtime.runners.task_runtime import TaskRuntime
from src.runtime.factory import RuntimeFactory
from src.tools.builtins import builtin_tools
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import ToolRegistry, default_registry
from src.uav import pose_measurements
from src.uav.config import UavSafetyConfig
from src.uav.motion import body_velocity_to_world_delta


class Recorder:
    def __init__(self, path):
        self.run_dir, self.events, self.artifacts = path, [], {}

    def event(self, name, payload):
        self.events.append((name, copy.deepcopy(payload)))

    debug_event = event

    def write_json(self, path, value):
        self.artifacts[path] = copy.deepcopy(value)


class Backend:
    def __init__(self):
        self.pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
        self.actions = []
        self.observations = 0
        self.fail_move = False

    def execute_action(self, scene, agent, action, args):
        self.actions.append((action, dict(args)))
        if action == "move_body" and self.fail_move:
            return ActionResult(False, action, "motion failed")
        if action == "turn_relative":
            self.pose["yaw_deg"] += args["yaw_deg"]
        if action == "move_body":
            delta = body_velocity_to_world_delta(args["vx"], args["vy"], args["vz"],
                                                  args["duration"], self.pose["yaw_deg"])
            for axis in ("x", "y", "z"):
                self.pose[axis] += delta["d" + axis]
        return ActionResult(True, action, "acknowledged", {"pose": dict(self.pose)})

    def get_observation(self, scene, agent, modalities):
        self.observations += 1
        return Observation(scene, agent, modalities, {"pose": dict(self.pose), "frames": {}})


class ScriptedPlanner:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.contexts = []
        self.histories = []
        self.decision_indices = []

    def plan_turn(self, transcript, registry, **kwargs):
        self.decision_indices.append(kwargs.get("decision_index"))
        checkpoint = transcript.latest_checkpoint()
        self.contexts.append(
            {
                "envelope": transcript.instruction_envelope(),
                "checkpoint": checkpoint.to_dict() if checkpoint else {},
            }
        )
        self.histories.append(transcript.snapshot())
        calls = next(self.batches)
        return AssistantTurn(f"turn_{len(self.contexts)}", "Use the current evidence", calls)


def call(identity, tool_name, **args):
    return AgentToolCall(identity, tool_name, args)


def move(identity):
    return call(identity, "uav.move_body", vx=1, vy=0, vz=0, duration=2)


def done(identity="done", observation="obs_0002"):
    return call(identity, "runtime.report_completed", reason="At requested endpoint", evidence_refs=[observation])


class ManagedFakeBackend(Backend):
    """Exercise the public lifecycle without sockets, processes, or a provider."""

    def __init__(self):
        super().__init__()
        from src.env_client import SceneInstance
        self.scene = SceneInstance("scene", "map", "default", "fake", "", 0, None, None, "running")
        self.starts = 0
        self.stops = []

    def start_scene(self, map_id, scene_id=None):
        self.starts += 1
        return self.scene

    def get_or_start_scene(self, map_id, scene_id=None):
        if not self.starts:
            return self.start_scene(map_id, scene_id)
        return self.scene

    def stop_scene(self, scene_id):
        self.stops.append(scene_id)


def public_runtime(tmp_path, batches, plugins=None):
    from src.runtime.configs.stabilization_config import UavStabilizationConfig
    plugins = plugins or PluginRegistry(skill_catalog=SkillCatalog([]))
    backend = ManagedFakeBackend()
    runtime = RuntimeFactory(plugin_registry=plugins).create_task_runtime(
        backend, Recorder(tmp_path), planner=ScriptedPlanner(batches),
        stabilization_config=UavStabilizationConfig(enabled=False),
    )
    return runtime, backend


def test_public_constructor_raw_instruction_and_terminal_lifecycle(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[move("move")], [done()]])
    state = runtime.run_task("map", "Move two meters", protocol_context="Do not land.")
    assert state.completed and state.task_plan is None
    assert state.physical_window_count == 1
    assert backend.stops == ["scene"]
    assert runtime.planner.contexts[0]["envelope"]["protocol_context"] == "Do not land."
    events = [name for name, _ in runtime.recorder.events]
    assert events.index("physical_window_recorded") < events.index("task_terminal") < events.index("scene_stopped")
    terminal = [e for e in runtime.agent_transcript.items if e.kind == "task_terminal"]
    assert len(terminal) == 1


def test_public_runtime_loads_knowledge_then_optional_plan_then_batch(tmp_path):
    skill = Skill("knowledge.test", "guide", "Guide", "summary", tmp_path,
                  tmp_path / "SKILL.md", body="Use current evidence.")
    runtime, backend = public_runtime(tmp_path, [
        [call("read", "knowledge.load", name="guide")],
        [call("plan", "runtime.update_plan", explanation="Reach endpoint", steps=[
            {"id": "reach", "description": "Reach requested endpoint", "status": "in_progress"}])],
        [move("m1"), move("m2")], [done()],
    ], plugins=PluginRegistry(skill_catalog=SkillCatalog([skill])))
    state = runtime.run_task("map", "Move four meters")
    assert state.completed and state.physical_window_count == 1
    assert state.loaded_knowledge[0]["name"] == "guide"
    assert state.task_plan.steps[0].status == "in_progress"
    assert backend.pose["x"] == 4
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [r.tool_call_id for r in results] == ["read", "plan", "m1", "m2", "done"]
    assert all(r.status == "ok" for r in results)


def test_public_runtime_guard_final_observation_and_cleanup(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[move("m1")]])
    runtime.runtime_loop_guard_windows = 1
    state = runtime.run_task("map", "Keep going")
    assert state.stopped and not state.completed
    assert state.stopped_reason == "instruction_physical_window_limit"
    assert state.physical_window_count == 1
    assert backend.observations == 3  # initial, automatic batch, terminal refresh
    assert backend.stops == ["scene"]


def test_public_constructor_does_not_accept_removed_parser(tmp_path):
    with pytest.raises(TypeError, match="task_parser"):
        TaskRuntime(ManagedFakeBackend(), Recorder(tmp_path), planner=ScriptedPlanner([]), task_parser=object())


def test_failure_audit_cannot_replace_provider_error_or_skip_cleanup(tmp_path):
    runtime, backend = public_runtime(tmp_path, [])
    failure = RuntimeError("primary provider failure")
    def fail_provider(*args, **kwargs):
        def fail_write(*args, **kwargs):
            raise OSError("audit disk failure")
        runtime._write_state = fail_write
        raise failure
    runtime.planner.plan_turn = fail_provider
    with pytest.raises(RuntimeError, match="primary provider failure") as captured:
        runtime.run_task("map", "Observe")
    assert captured.value is failure
    assert backend.stops == ["scene"]
    assert backend.observations == 2
    assert any(e["stage"] == "failure_audit" for e in failure.cleanup_errors)


def test_reused_scene_shutdown_error_survives_stop_error(tmp_path):
    runtime, backend = public_runtime(tmp_path, [[done(observation="obs_0001")]])
    state = runtime.run_task("map", "Observe", scene_policy="reuse_running", stop_scene_on_finish=False)
    failure = RuntimeError("shutdown failed")
    def fail_shutdown(*args):
        raise failure
    def fail_stop(scene):
        backend.stops.append(scene)
        raise OSError("stop failed")
    runtime._run_shutdown = fail_shutdown
    backend.stop_scene = fail_stop
    with pytest.raises(RuntimeError, match="shutdown failed") as captured:
        runtime.finish_reused_scene("map", state)
    assert captured.value is failure
    assert backend.stops == ["scene"]
    assert failure.cleanup_errors[0]["stage"] == "scene_stop"


def test_public_runtime_preserves_provider_error_when_scene_cleanup_also_fails(tmp_path):
    runtime, backend = public_runtime(tmp_path, [])
    failure = RuntimeError("provider unavailable")
    def fail_provider(*args, **kwargs):
        raise failure
    def fail_stop(scene):
        backend.stops.append(scene)
        raise OSError("scene cleanup failed")
    runtime.planner.plan_turn = fail_provider
    backend.stop_scene = fail_stop
    with pytest.raises(RuntimeError, match="provider unavailable") as captured:
        runtime.run_task("map", "Observe endpoint")
    assert captured.value is failure
    assert failure.cleanup_errors[0]["stage"] == "scene_stop"
    assert backend.stops == ["scene"]
    terminal = [e.payload for e in runtime.agent_transcript.items if e.kind == "task_terminal"]
    assert len(terminal) == 1 and not terminal[0]["completed"]
    assert terminal[0]["stop_source"] != "runtime.report_blocked"


def test_public_runtime_reused_scene_starts_fresh_instruction_state(tmp_path):
    runtime, backend = public_runtime(tmp_path, [
        [call("plan", "runtime.update_plan", explanation="First task only", steps=[])],
        [done(observation="obs_0001")],
        [done(observation="obs_0003")],
    ])
    first = runtime.run_task("map", "First instruction", scene_policy="reuse_running", stop_scene_on_finish=False)
    second = runtime.run_task("map", "Second instruction", scene_policy="reuse_running")
    assert runtime.planner.decision_indices == [1, 2, 1]
    assert first.task_plan is not None and second.task_plan is None
    assert first.completed and second.completed
    assert second.loaded_knowledge == [] and second.physical_window_count == 0
    assert backend.starts == 1 and backend.stops == ["scene"]
    assert runtime.agent_transcript.items[0].kind == "instruction_envelope"
    instruction_events = [
        event for event in runtime.agent_transcript.items if event.kind == "user_instruction"
    ]
    assert instruction_events[0].payload["instruction"] == "Second instruction"


def harness(tmp_path, batches):
    skill = Skill("knowledge.test", "guide", "Navigation guidance", "summary", tmp_path,
                  tmp_path / "SKILL.md", body="Use observable evidence, not plan completion.")
    plugins = PluginRegistry(skill_catalog=SkillCatalog([skill]))
    registry = ToolRegistry()
    for spec in default_registry(plugins)._tools.values():
        if spec.runtime.effect != "runtime_control":
            registry.register(spec)
    for spec in builtin_tools():
        registry.register(spec)
    backend = Backend()
    runtime = TaskRuntime.__new__(TaskRuntime)
    from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
    runtime.read_only_execution_config = ReadOnlyExecutionConfig()
    runtime.recorder = Recorder(tmp_path)
    from src.plugins.skills.dispatcher import SkillCapabilityDispatcher
    runtime.tools = ToolDispatcher(
        backend,
        registry,
        SkillCapabilityDispatcher(plugins),
        KnowledgeService(plugins.skill_catalog),
    )
    runtime.safety = SafetySupervisor()
    runtime.tool_telemetry = ToolTelemetryMiddleware(runtime.tools, runtime._debug_event)
    runtime.stabilizer = SimpleNamespace(wait_after_step=lambda *a, **k: {"status": "ok"})
    runtime.model_config = ModelConfig()
    runtime.loop_guard = TaskLoopGuard(
        runtime.model_config.max_consecutive_protocol_rejections
    )
    runtime.physical_windows = PhysicalWindowExecutor(
        runtime, runtime._pose_from_observation
    )
    runtime.turn_executor = TurnExecutor(runtime, runtime.physical_windows)
    runtime.instruction_loop = InstructionLoop(
        runtime, runtime.loop_guard, runtime.turn_executor
    )
    runtime.planner = ScriptedPlanner(batches)
    runtime.context_builder = ModelContextBuilder(
        registry=ContextModuleRegistry.default(plugin_warnings=plugins.warnings),
        knowledge_catalog=[
            {"name": item.name, "description": item.description}
            for item in plugins.skill_catalog.skills
        ],
    )
    runtime.memory_writer = MemoryEventWriter(tmp_path)
    runtime.agent_transcript = AgentTranscript(runtime.memory_writer)
    memory = FlightMemory("run", "map", "scene", "uav", active_instruction="Go to the endpoint",
                          active_instruction_id="inst_1", memory_reader=MemoryRecordReader(tmp_path))
    state = MissionState("run", "Go to the endpoint", protocol_context="Episode-specific guidance.")
    runtime.checkpoints = ContextCheckpointService(
        runtime.context_builder, runtime.agent_transcript, runtime.recorder
    )
    runtime.checkpoints.begin_instruction(state, memory)
    runtime._state_flight_memories = {"run": memory}
    runtime._decision_index = 0
    runtime.journal = RuntimeJournal(runtime.recorder)
    scene = SimpleNamespace(scene_instance_id="scene", capabilities=None)
    runtime._record_observation(scene, "uav", state, ["pose"], "instruction_start")
    runtime.checkpoints.publish(state, memory, "instruction_start")
    return runtime, state, scene, backend


def run_loop(runtime, state, scene):
    memory = runtime._memory_for_state(state)
    runtime.instruction_loop.run(
        InstructionSession(
            scene=scene,
            agent_id="uav",
            state=state,
            memory=memory,
            transcript=runtime.agent_transcript,
        )
    )


def test_knowledge_plan_physical_batch_and_terminal_without_phase(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [call("knowledge", "knowledge.load", name="guide")],
        [call("plan", "runtime.update_plan", explanation="Two short movements", steps=[
            {"id": "go", "description": "Reach the endpoint", "status": "in_progress"}])],
        [move("m1"), move("m2")], [done()],
    ])
    run_loop(runtime, state, scene)
    assert state.completed and not state.stopped
    assert state.physical_window_count == 1 and backend.observations == 2
    assert state.latest_pose["x"] == 4
    assert state.task_plan.steps[0].status == "in_progress"  # task terminal is independent
    assert state.loaded_knowledge[0]["name"] == "guide"
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [r.tool_call_id for r in results] == ["knowledge", "plan", "m1", "m2", "done"]
    assert all(r.status == "ok" for r in results)
    windows = [payload for name, payload in runtime.recorder.events if name == "physical_window_recorded"]
    assert len(windows) == 1 and windows[0]["phase_key"] is None
    assert windows[0]["observation_before_id"] == "obs_0001"
    assert windows[0]["observation_after_id"] == "obs_0002"
    assert [s["tool_call_id"] for s in windows[0]["ordered_steps"]] == ["m1", "m2"]
    for context in runtime.planner.contexts:
        checkpoint = context["checkpoint"]
        assert not {"current_phase", "phase_strategy", "semantic_plan"} & checkpoint.keys()
        assert context["envelope"]["protocol_context"] == "Episode-specific guidance."
    assert len([k for k in runtime.recorder.artifacts if k.startswith("debug/tool_loop/")]) == 4


def test_simple_task_can_move_and_finish_without_plan(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [[move("m1")], [done()]])
    run_loop(runtime, state, scene)
    assert state.completed and state.task_plan is None
    assert backend.observations == 2


@pytest.mark.parametrize("workers", [1, 4])
def test_mixed_turn_rejected_before_any_action_then_corrected(tmp_path, workers):
    runtime, state, scene, backend = harness(tmp_path, [
        [call("read", "knowledge.load", name="guide"), move("bad_move")],
        [move("good_move")], [done()],
    ])
    from src.runtime.configs.read_only_execution_config import ReadOnlyExecutionConfig
    runtime.read_only_execution_config = ReadOnlyExecutionConfig(workers)
    run_loop(runtime, state, scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [r.status for r in results] == ["rejected", "rejected", "ok", "ok"]
    assert len(backend.actions) == 1 and state.loaded_knowledge == []


def test_plan_changes_cannot_reset_nonphysical_guard(tmp_path):
    plans = [[call(f"plan{i}", "runtime.update_plan", explanation=f"revision {i}", steps=[])]
             for i in range(2)]
    runtime, state, scene, backend = harness(tmp_path, plans)
    state.max_nonphysical_rounds = 2
    run_loop(runtime, state, scene)
    assert state.stopped_reason == "nonphysical_round_limit"
    assert state.task_plan.revision == 2 and state.physical_window_count == 0
    assert backend.actions == [] and backend.observations == 1


def test_failed_first_physical_call_closes_remaining_as_skipped(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [move("failed"), move("skipped")],
        [call("blocked", "runtime.report_blocked", blocking_condition="unsafe_to_continue",
              reason="Motion unavailable", evidence_refs=["obs_0002"])],
    ])
    backend.fail_move = True
    run_loop(runtime, state, scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [r.status for r in results] == ["error", "skipped", "ok"]
    assert backend.observations == 2 and state.stopped and not state.completed
    assert [action for action, _ in backend.actions] == ["move_body", "hover"]


@pytest.mark.parametrize("reuse_scene", [False, True])
def test_scene_recovery_stops_public_runtime_and_pairs_calls(tmp_path, reuse_scene):
    from src.env_client.errors import SceneRecoveryRequired
    runtime, backend = public_runtime(tmp_path, [[move("fatal"), move("cancelled")], [done()]])
    original = backend.execute_action
    def execute(scene, agent, action, args):
        if action == "move_body" and args.get("vx"):
            backend.actions.append((action, args))
            return ActionResult(False, action, "recreate scene", {},
                                {"requires_scene_reset": True, "action_diagnostics": {"private": True}})
        return original(scene, agent, action, args)
    backend.execute_action = execute
    with pytest.raises(SceneRecoveryRequired):
        runtime.run_task("map", "Move", scene_policy="reuse_running" if reuse_scene else "isolated",
                         stop_scene_on_finish=not reuse_scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [(r.tool_call_id, r.status) for r in results] == [("fatal", "error"), ("cancelled", "skipped")]
    assert "action_diagnostics" not in results[0].data
    assert len(runtime.planner.contexts) == 1
    assert backend.actions[-1][0] == "move_body"
    if reuse_scene:
        count = len(backend.actions)
        runtime.safe_hover_reused_scene("map")
        with pytest.raises(SceneRecoveryRequired):
            runtime.run_task("map", "Retry without reset", scene_policy="reuse_running", stop_scene_on_finish=False)
        state = MissionState("cleanup", "Move")
        state.connected = True
        runtime.finish_reused_scene("map", state)
        assert len(backend.actions) == count and len(runtime.planner.contexts) == 1
    assert backend.stops == ["scene"]
    assert any(e.kind == "task_terminal" for e in runtime.agent_transcript.items)


def test_instruction_motion_and_plan_updates_preserve_origins_and_last_window():
    state = MissionState("run", "Move")
    assert state.max_physical_windows == 300
    state.record_observation({"observation_id": "obs1", "pose": {"x": 0, "y": 0, "z": 0, "yaw": 0}})
    before = copy.deepcopy(state.latest_observation)
    state.begin_physical_window()
    after = {"observation_id": "obs2", "pose": {"x": 2, "y": 0, "z": 0, "yaw": 0}}
    state.finish_physical_window(after, pose_measurements(before["pose"], after["pose"]))
    state.record_observation(after)
    apply_plan_update(state, {"explanation": "new route", "steps": []})
    assert state.motion_summary["last_window"]["forward_m"] == 2
    assert state.instruction_origin_observation_id == "obs1"
    assert state.before_last_action_window_id == "obs1"
    assert state.physical_window_count == 1
    snapshot = state.to_dict()
    snapshot["latest_observation"]["pose"]["x"] = 999
    assert state.latest_pose["x"] == 2


def test_missing_pose_is_unknown_not_zero():
    state = MissionState("run", "Look")
    state.record_observation({"observation_id": "obs1"})
    assert state.motion_summary["measurements"]["forward_m"] is None


def test_context_never_selects_knowledge_or_exposes_unloaded_body(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [[done(observation="obs_0001")]])
    module = runtime.context_builder.registry.modules["skills"]
    module.skill_selector = SimpleNamespace(select=lambda *args: pytest.fail("context must not select knowledge"))
    run_loop(runtime, state, scene)
    context = runtime.planner.contexts[0]
    assert context["envelope"]["knowledge_catalog"] == [
        {"name": "guide", "description": "Navigation guidance"}
    ]
    assert context["checkpoint"]["loaded_knowledge"] == []
    assert "phase_start" not in runtime._memory_for_state(state).observation_anchors


def test_reused_call_id_is_provider_rejection_not_duplicate_canonical_result(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [move("one")], [move("one")], [done()],
    ])
    run_loop(runtime, state, scene)
    results = [e.payload.tool_call_id for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert results == ["one", "done"]
    assert len(backend.actions) == 1
    assert any(e.kind == "provider_turn_rejected" for e in runtime.agent_transcript.items)


def test_instruction_guard_keeps_final_after_checkpoint(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [[move("one")], [move("two")]])
    state.max_physical_windows = 2
    run_loop(runtime, state, scene)
    assert state.physical_window_count == 2 and state.stopped_reason == "instruction_physical_window_limit"
    assert backend.observations == 3
    assert runtime.agent_transcript.latest_observation_id() == "obs_0003"


def test_safety_modification_is_associated_with_its_own_batch_step(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [move("clipped"), call("unchanged", "uav.move_body", vx=0.2, vy=0, vz=0, duration=2)],
        [done()],
    ])
    runtime.safety = SafetySupervisor(UavSafetyConfig(max_linear_velocity_mps=0.5))
    run_loop(runtime, state, scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert results[0].requested_args["vx"] == 1 and results[0].executed_args["vx"] == 0.5
    assert results[0].safety_interventions[0]["step_index"] == 0
    assert results[1].safety_interventions == []
    assert state.latest_pose["x"] == pytest.approx(1.4)


def test_unknown_knowledge_returns_error_and_can_be_corrected(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [
        [call("unknown", "knowledge.load", name="not_enabled")],
        [call("known", "knowledge.load", name="guide")], [done(observation="obs_0001")],
    ])
    run_loop(runtime, state, scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [result.status for result in results] == ["error", "ok", "ok"]
    assert state.physical_window_count == 0 and backend.observations == 1


@pytest.mark.parametrize("camera_fails", [False, True])
def test_stabilization_failure_closes_batch_and_preserves_original_error(tmp_path, camera_fails):
    runtime, state, scene, backend = harness(tmp_path, [[move("ack"), move("not_run")]])
    original = TimeoutError("stabilization connection lost")

    def fail_stabilization(*args, **kwargs):
        raise original

    runtime.stabilizer.wait_after_step = fail_stabilization
    if camera_fails:
        def fail_camera(*args, **kwargs):
            raise OSError("camera unavailable")
        backend.get_observation = fail_camera
    with pytest.raises(TimeoutError) as failure:
        run_loop(runtime, state, scene)
    assert failure.value is original
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [(r.tool_call_id, r.status) for r in results] == [("ack", "ok"), ("not_run", "skipped")]
    assert len([a for a, _ in backend.actions if a == "move_body"]) == 1
    windows = [p for name, p in runtime.recorder.events if name == "physical_window_recorded"]
    assert len(windows) == 1 and windows[0]["status"] == "error"
    assert windows[0]["error_type"] == "TimeoutError"
    assert windows[0]["observation_after_id"] == (None if camera_fails else "obs_0002")
    assert bool(windows[0]["observation_error"]) == camera_fails
    assert not state.completed


def test_after_camera_failure_preserves_action_results_without_fake_observation(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [[move("ack1"), move("ack2")]])

    def fail_camera(*args, **kwargs):
        raise OSError("camera unavailable")

    backend.get_observation = fail_camera
    with pytest.raises(OSError, match="camera unavailable"):
        run_loop(runtime, state, scene)
    results = [e.payload for e in runtime.agent_transcript.items if e.kind == "tool_result"]
    assert [(r.tool_call_id, r.status) for r in results] == [("ack1", "ok"), ("ack2", "ok")]
    windows = [p for name, p in runtime.recorder.events if name == "physical_window_recorded"]
    assert len(windows) == 1
    assert windows[0]["observation_before_id"] == "obs_0001"
    assert windows[0]["observation_after_id"] is None
    assert not state.completed


def test_capture_context_does_not_duplicate_image_binary(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [])
    runtime._capture_instruction_exchange("decision_0001")
    ref = runtime.recorder.events[-1][1]["debug_ref"]
    artifact = runtime.recorder.artifacts[ref]
    checkpoint = runtime.agent_transcript.latest_checkpoint()
    assert checkpoint is not None
    assert "data:image/" not in str(artifact["model_context"])
    assert artifact["model_context"] == checkpoint.to_dict()
    assert runtime.recorder.events[-1][0] == "model_turn_recorded"


def test_instruction_captures_do_not_overwrite_previous_decisions(tmp_path):
    runtime, state, scene, backend = harness(tmp_path, [])
    runtime._capture_instruction_exchange("decision_0001")
    first_ref = runtime.recorder.events[-1][1]["debug_ref"]
    first_snapshot = copy.deepcopy(runtime.recorder.artifacts[first_ref])
    from src.memory.context.contracts import InstructionEnvelope

    runtime.agent_transcript.start_instruction(
        "Another task",
        "../different/instruction",
        InstructionEnvelope(
            instruction_id="../different/instruction",
            protocol_context="",
            identity={},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )
    runtime._capture_instruction_exchange("decision_0001")
    second_ref = runtime.recorder.events[-1][1]["debug_ref"]
    assert first_ref != second_ref and ".." not in second_ref
    assert runtime.recorder.artifacts[first_ref] == first_snapshot


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_instruction_guard_rejects_invalid_configuration(value):
    with pytest.raises(ValueError):
        MissionState("run", "task", max_physical_windows=value)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_acquisition_audit_failure_rolls_back_and_preserves_original(tmp_path, cleanup_fails):
    runtime, backend = public_runtime(tmp_path, [])
    original = OSError("scene contract cannot be persisted")

    def fail_record(scene):
        raise original

    runtime._record_scene_contract = fail_record
    if cleanup_fails:
        def fail_stop(scene_id):
            backend.stops.append(scene_id)
            raise RuntimeError("stop failed")
        backend.stop_scene = fail_stop
    with pytest.raises(OSError) as failure:
        runtime.run_task("map", "Observe")
    assert failure.value is original
    assert backend.stops == ["scene"]
    assert backend.actions == []
    if cleanup_fails:
        assert original.cleanup_errors[0]["stage"] == "scene_acquisition_rollback"


def test_existing_reused_scene_is_not_stopped_by_audit_failure(tmp_path):
    runtime, backend = public_runtime(tmp_path, [])
    runtime.ensure_reused_scene("map")
    def fail_record(scene):
        raise OSError("storage offline")
    runtime._record_scene_contract = fail_record
    with pytest.raises(OSError):
        runtime.ensure_reused_scene("map")
    assert backend.stops == []


def test_public_turn_move_batch_uses_canonical_yaw_and_one_after(tmp_path):
    runtime, backend = public_runtime(tmp_path, [
        [call("turn", "uav.turn_relative", yaw_deg=30), move("move")], [done()],
    ])
    state = runtime.run_task("map", "Turn and move")
    assert state.completed
    windows = [p for name, p in runtime.recorder.events if name == "physical_window_recorded"]
    assert len(windows) == 1
    expected = body_velocity_to_world_delta(1, 0, 0, 2, 30)
    assert state.latest_pose["x"] == pytest.approx(expected["dx"])
    assert state.latest_pose["y"] == pytest.approx(expected["dy"])
    assert state.latest_pose["yaw_deg"] == 30
    assert backend.observations == 3  # initial, one batch after, terminal refresh
