"""Test M4 task runtime dry-run behavior."""

from __future__ import annotations

import base64
import inspect
import json

import pytest

from evaluation.benchmarks.core import EpisodeTraceReader, TraceCursor
from src.agent.mission_state import MissionState
from src.agent.messages import AgentMessage
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan, PlannedStep, ShortHorizonPlan
from src.agent.safety import SafetyDecision, SafetySupervisor
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal, TaskParseResult
from src.agent.semantic.task_parser import TaskParser
from src.adapters.uav_primitives import NullUavPrimitiveAdapter, PrimitiveResult
from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage, SceneCapabilities
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.runtime_control_config import RuntimeControlActionConfig, RuntimeControlConfig
from src.runtime.configs.safety_config import UavSafetyConfig
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.context import PhaseWindowContextBuilder
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.plugins.contracts.skill import SkillCapabilityResult
from src.plugins.skills import dispatcher as skill_dispatcher_module
from src.plugins.skills.selection import ModelSkillSelector
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.configs.model_config import ModelConfig
from src.uav import pose_measurements
from src.runtime.runners.task_runtime import TaskRuntime
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.server import SimulationServer
from tests.simulator_fakes import in_memory_backend_registry


LAMP_TASK = "前面右转会有一个黑色的路灯，右转面向路灯后停下"
ENTRANCE_TASK = "左转寻找建筑入口，并停在入口前"


@pytest.fixture(autouse=True)
def _use_explicit_in_memory_simulator_backends(monkeypatch):
    """Keep Runtime tests independent from real AirSim and UE processes."""

    monkeypatch.setattr(
        BackendRegistry,
        "defaults",
        classmethod(lambda cls: in_memory_backend_registry()),
    )


def test_task_runtime_default_loop_guard_is_40():
    parameter = inspect.signature(TaskRuntime.__init__).parameters["runtime_loop_guard_windows"]

    assert parameter.default == 40


def test_native_completion_requires_fresh_success_after_evidence_tool_use(
    tmp_path,
):
    run_dir = tmp_path / "completion_evidence_freshness"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )
    runtime.agent_transcript.start_instruction("hold a measured distance", "inst")
    runtime.agent_transcript.append_state(
        {"current_phase": {"name": "final_phase"}},
        "obs_0002",
    )
    runtime.agent_transcript.append_tool_result(
        ToolResult(
            "call_failed",
            "skill.visual_grounding.locate",
            "error",
            data={"observation_id": "obs_0002"},
            requested_args={
                "observation_id": "obs_0002",
                "query": "the reference",
                "include_distance": True,
            },
        )
    )

    with pytest.raises(ValueError, match="successful result for the current observation"):
        runtime._validate_current_observation_completion_evidence("final_phase")

    runtime.agent_transcript.append_tool_result(
        ToolResult(
            "call_ok",
            "skill.visual_grounding.locate",
            "ok",
            data={
                "observation_id": "obs_0002",
                "candidates": [
                    {"candidate_id": "candidate_01", "distance_m": 3.5}
                ],
            },
            requested_args={
                "observation_id": "obs_0002",
                "query": "the reference",
                "include_distance": True,
            },
        )
    )
    runtime._validate_current_observation_completion_evidence("final_phase")
    recorder.close()


def test_detection_only_grounding_does_not_create_metric_completion_requirement(
    tmp_path,
):
    run_dir = tmp_path / "detection_only_completion"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )
    runtime.agent_transcript.start_instruction("find the reference", "inst")
    runtime.agent_transcript.append_state(
        {"current_phase": {"name": "final_phase"}},
        "obs_0002",
    )
    runtime.agent_transcript.append_tool_result(
        ToolResult(
            "call_detect",
            "skill.visual_grounding.locate",
            "ok",
            data={"observation_id": "obs_0001", "candidates": []},
            requested_args={
                "observation_id": "obs_0001",
                "query": "the reference",
                "include_distance": False,
            },
        )
    )

    runtime._validate_current_observation_completion_evidence("final_phase")
    recorder.close()


def test_task_runtime_injects_one_model_config_into_planner_dependencies(tmp_path):
    config = ModelConfig(
        task_parser_mode="model",
        model_name="stub",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    run_dir = tmp_path / "model_dependency_wiring"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=ModelActionPlanner(config),
        model_config=config,
        shutdown_policy="none",
        runtime_control_config=RuntimeControlConfig(
            enabled=True,
            actions={
                "semantic_plan_update": RuntimeControlActionConfig(
                    enabled=True,
                    exposed=True,
                )
            },
        ),
    )

    status = runtime._model_dependency_status()
    recorder.close()

    assert status["model_config_source"] == "task_runtime_injected"
    assert status["model_config_has_api_key"] is True
    assert status["planner_skill_selection_enabled"] is True
    assert status["planner_skill_selector_uses_runtime_config"] is True
    assert status["semantic_updater_configured"] is True
    assert status["semantic_updater_uses_runtime_config"] is True
    assert "test-key" not in json.dumps(status)


def test_task_runtime_rejects_model_planner_without_explicit_model_config(tmp_path):
    config = ModelConfig(
        task_parser_mode="model",
        model_name="stub",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    recorder = AsyncRunLogger(tmp_path / "missing_model_config")
    server = SimulationServer([make_package()], run_dir=tmp_path)

    with pytest.raises(ValueError, match="explicit model_config"):
        TaskRuntime(
            SimulatorEnvClient(server),
            recorder,
            TaskParser(mode="rule_based"),
            planner=ModelActionPlanner(config),
            shutdown_policy="none",
        )

    recorder.close()


def test_task_runtime_planner_skill_trace_is_compact_and_auditable():
    trace = TaskRuntime._planner_skill_selection_trace(
        {
            "fragments": [
                {
                    "name": "skills",
                    "content": {
                        "selected_skills": [
                            {"name": "approach_and_standoff", "skill_text": "hidden body"}
                        ]
                    },
                    "metadata": {
                        "selection_source": "phase_cache",
                        "selection": {
                            "status": "selected",
                            "requested_skill_names": ["approach_and_standoff"],
                            "accepted_skill_names": ["approach_and_standoff"],
                        },
                    },
                }
            ]
        }
    )

    assert trace == {
        "selection_status": "selected",
        "requested_skills": ["approach_and_standoff"],
        "accepted_skills": ["approach_and_standoff"],
        "loaded_skills": ["approach_and_standoff"],
        "selection_source": "phase_cache",
    }
    assert "skill_text" not in json.dumps(trace)


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


def make_single_phase_parse_result(
    instruction: str = "single phase task",
    phase_name: str = "final_phase",
) -> TaskParseResult:
    return TaskParseResult(
        instruction=instruction,
        parser_mode="test",
        plan=SemanticPlan(
            intent="test_single_phase",
            task_goal=TaskGoal(
                target="single phase target",
                final_state="done",
            ),
            phases=[
                SemanticPhase(
                    index=1,
                    total=1,
                    name=phase_name,
                    goal="complete the single test phase",
                    done_when=["single test phase is complete"],
                    continue_when=["single test phase is not complete"],
                )
            ],
        ),
    )


def make_visual_target_parse_result(
    instruction: str = "turn to face the visible target",
    phase_name: str = "orient_visual_target",
) -> TaskParseResult:
    return TaskParseResult(
        instruction=instruction,
        parser_mode="test",
        plan=SemanticPlan(
            intent="visual_target_orientation",
            task_goal=TaskGoal(
                target="visible target",
                final_state="target is visually aligned",
            ),
            phases=[
                SemanticPhase(
                    index=1,
                    total=1,
                    name=phase_name,
                    goal="align the UAV view with the target",
                    done_when=["visible target is identifiable", "view is roughly aligned with the target"],
                    continue_when=["target is not visually confirmed"],
                )
            ],
        ),
    )


def make_approach_parse_result() -> TaskParseResult:
    return TaskParseResult(
        instruction="approach the sunshade",
        parser_mode="test",
        plan=SemanticPlan(
            intent="navigate_by_visual_landmark",
            task_goal=TaskGoal(target="sunshade", final_state="near_sunshade"),
            phases=[
                SemanticPhase(
                    index=1,
                    total=1,
                    name="approach_sunshade",
                    goal="approach the sunshade",
                    done_when=[
                        "sunshade is clearly visible",
                        "UAV has moved meaningfully closer to sunshade than at phase start",
                    ],
                    continue_when=["UAV has not moved closer yet"],
                )
            ],
        ),
    )


def make_two_phase_parse_result() -> TaskParseResult:
    return TaskParseResult(
        instruction="two phase task",
        parser_mode="test",
        plan=SemanticPlan(
            intent="test_two_phase",
            task_goal=TaskGoal(target="test target", final_state="done"),
            phases=[
                SemanticPhase(
                    index=1,
                    total=2,
                    name="phase_one",
                    goal="finish phase one",
                    done_when=["phase one is complete"],
                    continue_when=["phase one is not complete"],
                ),
                SemanticPhase(
                    index=2,
                    total=2,
                    name="phase_two",
                    goal="finish phase two",
                    done_when=["phase two is complete"],
                    continue_when=["phase two is not complete"],
                ),
            ],
        ),
    )


def load_event_records(run_dir):
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    debug_path = run_dir / "debug" / "debug_events.jsonl"
    if debug_path.exists():
        events.extend(json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines())
    return sorted(events, key=lambda record: str(record.get("timestamp", "")))


def write_tiny_png(path):
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )


def coupled_continue_from_short_plan(plan: ShortHorizonPlan, reason: str = "test coupled planner continues") -> CoupledActionPhasePlan:
    return CoupledActionPhasePlan(
        phase=plan.phase,
        phase_decision=PhaseTransition(CONTINUE_CURRENT_PHASE, reason),
        steps=list(plan.steps),
        action_window=plan.action_window,
        agent_messages=list(plan.agent_messages),
    )


class HoverOnlyPlanner:
    planner_mode = "test"

    def __init__(self) -> None:
        self._phase_calls: dict[str, int] = {}

    def plan_next_with_phase_decision(self, state, context=None):
        phase_key = state.current_phase_key
        calls = self._phase_calls.get(phase_key, 0)
        self._phase_calls[phase_key] = calls + 1
        if calls == 0:
            return CoupledActionPhasePlan(
                phase=state.current_phase_name,
                phase_decision=PhaseTransition(CONTINUE_CURRENT_PHASE, "test planner records one hover window"),
                steps=[PlannedStep.tool_step("uav.hover", {}, "test planner holds position")],
                agent_messages=[],
            )
        decision = COMPLETE_TASK
        reason = "test planner completed final phase"
        if state.current_phase_index < len(state.semantic_plan.phases) - 1:
            decision = ADVANCE_TO_NEXT_PHASE
            reason = "test planner completed current phase"
        return CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=PhaseTransition(decision, reason),
            steps=[],
            agent_messages=[],
        )


class CoupledPlannerStub:
    planner_mode = "model"

    def __init__(self, decisions: list[str]) -> None:
        self.decisions = list(decisions)
        self.contexts = []
        self.judge_calls = 0

    def plan_next_with_phase_decision(self, state, context=None):
        self.contexts.append(context)
        decision = self.decisions.pop(0) if self.decisions else COMPLETE_TASK
        if decision == CONTINUE_CURRENT_PHASE:
            return CoupledActionPhasePlan.from_dict(
                {
                    "phase": state.current_phase_name,
                    "phase_decision": {"decision": CONTINUE_CURRENT_PHASE, "reason": "need one more tool window"},
                    "action_window": {
                        "intent": "continue current phase with one hover",
                        "scope": "current_phase_only",
                        "requires_intermediate_observation": False,
                        "expected_effect": "collect progress after a stable hover",
                    },
                    "agent_messages": [
                        {
                            "type": "progress",
                            "content": "当前阶段还需要继续，我先保持悬停确认。",
                            "evidence_refs": [],
                            "confidence": "medium",
                            "requires_user_reply": False,
                        }
                    ],
                    "phase_strategy": {
                        "stages": ["Hold until the current phase evidence is refreshed"],
                        "active_stage": 0,
                        "reason": "The current phase still requires evidence",
                    },
                    "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
                },
                require_agent_messages=True,
            )
        return CoupledActionPhasePlan.from_dict(
            {
                "phase": state.current_phase_name,
                "phase_decision": {"decision": decision, "reason": f"coupled planner decided {decision}"},
                "agent_messages": [
                    {
                        "type": "final" if decision == COMPLETE_TASK else "progress",
                        "content": "阶段证据已经足够，我将切换阶段。",
                        "evidence_refs": [],
                        "confidence": "high",
                        "requires_user_reply": False,
                    }
                ],
                "steps": [],
            },
            require_agent_messages=True,
        )


class RuntimeControlRequestPlanner(CoupledPlannerStub):
    def __init__(self) -> None:
        super().__init__([CONTINUE_CURRENT_PHASE, COMPLETE_TASK])

    def plan_next_with_phase_decision(self, state, context=None):
        self.contexts.append(context)
        if len(self.contexts) == 1:
            return CoupledActionPhasePlan.from_dict(
                {
                    "phase": state.current_phase_name,
                    "phase_decision": {
                        "decision": CONTINUE_CURRENT_PHASE,
                        "reason": "continue while requesting runtime control validation",
                    },
                    "action_window": {
                        "intent": "continue current phase with one hover",
                        "scope": "current_phase_only",
                        "requires_intermediate_observation": False,
                        "expected_effect": "keep normal flight behavior unchanged",
                    },
                    "agent_messages": [],
                    "control_actions": [
                        {
                            "name": "semantic_plan_update",
                            "reason": "phase contract may be too strict",
                            "payload": {"reason": "phase contract may be too strict"},
                        }
                    ],
                    "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
                },
                require_agent_messages=True,
            )
        return CoupledActionPhasePlan.from_dict(
            {
                "phase": state.current_phase_name,
                "phase_decision": {"decision": COMPLETE_TASK, "reason": "done"},
                "agent_messages": [],
                "steps": [],
            },
            require_agent_messages=True,
        )


class AlwaysHoverPlanner:
    planner_mode = "test"

    def __init__(self) -> None:
        self.contexts = []

    def plan_next_with_phase_decision(self, state, context=None):
        self.contexts.append(context)
        plan = ShortHorizonPlan(
            state.current_phase_name,
            [PlannedStep.tool_step("uav.hover", {}, "keep checking current phase")],
        )
        return coupled_continue_from_short_plan(plan, "not enough evidence")


class FakeSemanticPlanUpdater:
    def __init__(self, patch: dict) -> None:
        self.patch = dict(patch)
        self.contexts = []

    def update(self, context):
        self.contexts.append(context)
        return dict(self.patch)


class CompleteAfterSemanticUpdatePlanner:
    planner_mode = "model"

    def __init__(self) -> None:
        self.contexts = []
        self.calls = 0

    def plan_next_with_phase_decision(self, state, context=None):
        self.calls += 1
        self.contexts.append(context)
        done_when = context.get("current_phase", {}).get("done_when", []) if isinstance(context, dict) else []
        if any("updated completion" in item for item in done_when):
            return CoupledActionPhasePlan.from_dict(
                {
                    "phase": state.current_phase_name,
                    "phase_decision": {"decision": COMPLETE_TASK, "reason": "semantic contract update is visible"},
                    "agent_messages": [],
                    "steps": [],
                },
                require_agent_messages=True,
            )
        control_actions = []
        if self.calls == 1:
            control_actions = [
                {
                    "name": "semantic_plan_update",
                    "reason": "current phase contract is too strict",
                    "payload": {"reason": "current phase contract is too strict"},
                }
            ]
        return CoupledActionPhasePlan.from_dict(
            {
                "phase": state.current_phase_name,
                "phase_decision": {"decision": CONTINUE_CURRENT_PHASE, "reason": "wait for semantic update"},
                "action_window": {
                    "intent": "continue while semantic update runs",
                    "scope": "current_phase_only",
                    "requires_intermediate_observation": False,
                    "expected_effect": "keep flight behavior safe while updater works",
                },
                "agent_messages": [],
                "control_actions": control_actions,
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
            },
            require_agent_messages=True,
        )


def test_task_runtime_completes_dry_run_and_writes_records(tmp_path):
    run_dir = tmp_path / "task_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(SimulatorEnvClient(server), recorder, TaskParser(mode="rule_based"), planner=HoverOnlyPlanner())

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    assert state.completed
    assert state.connected
    assert state.airborne
    assert state.latest_observation
    assert (run_dir / "state/semantic_plan.json").exists()
    assert (run_dir / "state/scene_capabilities.json").exists()
    assert not (run_dir / "mission_state.json").exists()
    assert not (run_dir / "latest_state.json").exists()
    assert not (run_dir / "flight_memory.json").exists()
    assert not (run_dir / "semantic_plan.json").exists()
    assert not (run_dir / "observations").exists()
    assert not (run_dir / "memory/artifacts").exists()
    assert (run_dir / "debug/latest_observation.json").exists()
    assert (run_dir / "debug/debug_events.jsonl").exists()
    assert (run_dir / "contexts/latest_context.json").exists()
    assert (run_dir / "contexts/latest_action_planner_model_context.json").exists()
    assert (run_dir / "state/flight_memory.json").exists()
    assert (run_dir / "memory/instructions.jsonl").exists()
    assert (run_dir / "memory/actions.jsonl").exists()
    assert (run_dir / "memory/poses.jsonl").exists()
    assert (run_dir / "memory/observations.jsonl").exists()
    assert (run_dir / "memory/frames.jsonl").exists()

    event_records = load_event_records(run_dir)
    event_names = [record["event"] for record in event_records]
    short_horizon_tools = [
        step.get("tool")
        for record in event_records
        if record["event"] == "short_horizon_planned"
        for step in record["payload"]["steps"]
        if step["kind"] == "tool"
    ]
    mission_state = json.loads((run_dir / "state/latest_state.json").read_text(encoding="utf-8"))
    latest_observation = json.loads((run_dir / "debug/latest_observation.json").read_text(encoding="utf-8"))
    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))
    latest_model_context = json.loads(
        (run_dir / "contexts/latest_action_planner_model_context.json").read_text(
            encoding="utf-8"
        )
    )
    scene_capabilities = json.loads(
        (run_dir / "state/scene_capabilities.json").read_text(
            encoding="utf-8"
        )
    )
    flight_memory = json.loads((run_dir / "state/flight_memory.json").read_text(encoding="utf-8"))

    assert "control_connected" in event_names
    assert "preflight_takeoff_started" in event_names
    assert "preflight_takeoff_completed" in event_names
    assert "preflight_hover_completed" in event_names
    assert "initial_observation_updated" in event_names
    assert "short_horizon_planned" in event_names
    assert "tool_call_completed" in event_names
    assert "automatic_observation_policy_applied" in event_names
    assert "automatic_observation_updated" in event_names
    assert "shutdown_started" in event_names
    assert "shutdown_hover_completed" in event_names
    assert "scene_stopped" in event_names
    assert "uav.connect" not in short_horizon_tools
    assert "uav.takeoff" not in short_horizon_tools
    assert mission_state["connected"] is True
    assert mission_state["airborne"] is True
    assert "recent_observations" not in mission_state
    assert mission_state["latest_observation"]["source"] == "automatic_phase_window"
    assert "pose" in mission_state["latest_observation"]
    assert "frames" in mission_state["latest_observation"]
    assert (run_dir / "state/latest_state.json").exists()
    assert latest_observation["source"] == "automatic_phase_window"
    assert latest_context["active_instruction"]
    assert scene_capabilities["protocol"] == "airsim"
    assert scene_capabilities["control_mode"] == "Multirotor"
    assert "simulator_contract" not in latest_context
    assert "simulator_contract" not in latest_model_context
    assert latest_context["current_phase"] is not None
    assert "context_schema" in latest_context
    assert "image" in latest_context["context_schema"]["module_content_fields"]
    fragments = {fragment["name"]: fragment for fragment in latest_context["fragments"]}
    selected_roles = [item["role"] for item in fragments["image"]["content"]["selected_frames"]]
    assert selected_roles == ["current_observation"]
    assert "lifecycle_tools_excluded" not in latest_context["allowed_action_constraints"]
    assert "recent_actions" not in flight_memory
    assert "recent_observations" not in flight_memory
    assert flight_memory["observation_anchors"]["task_initial"]["ref"].startswith("memory/observations.jsonl#")
    assert flight_memory["recent_action_refs"]
    assert flight_memory["recent_observation_refs"]
    assert "runtime_control" not in latest_context


def test_task_runtime_validates_runtime_control_request_without_executing_it(tmp_path):
    run_dir = tmp_path / "runtime_control_request_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = RuntimeControlRequestPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        runtime_control_config=RuntimeControlConfig(
            enabled=True,
            actions={
                "semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True),
            },
        ),
        semantic_plan_updater=FakeSemanticPlanUpdater({}),
    )
    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    request = next(record for record in events if record["event"] == "runtime_control_action_requested")
    completion = next(record for record in events if record["event"] == "runtime_control_action_completed")

    assert state.completed
    assert planner.contexts[0]["runtime_control"]["available"] == ["semantic_plan_update"]
    assert planner.contexts[0]["runtime_control"]["contracts"] == {
        "semantic_plan_update": (
            "After multiple action-observation cycles, request when confidence is low that the current goal, "
            "done_when, or continue_when still fit observed progress and user intent. Name the doubtful "
            "fields and evidence; keep normal UAV steps if continuing."
        )
    }
    assert "tool_step_completed" in event_names
    assert request["payload"] == {
        "name": "semantic_plan_update",
        "reason": "phase contract may be too strict",
        "status": "accepted",
    }
    assert any(
        record["event"] == "runtime_control_action_started"
        and record["payload"]["name"] == "semantic_plan_update"
        and record["payload"]["status"] == "pending"
        for record in events
    )
    assert "ask_user" not in planner.contexts[0]["runtime_control"]["available"]


def test_task_runtime_uses_injected_context_builder(tmp_path):
    class InjectedContextBuilder:
        def build(self, state, memory):
            return {"active_instruction": state.user_instruction, "fragments": []}

    recorder = AsyncRunLogger(tmp_path / "injected_context")
    server = SimulationServer([make_package()], run_dir=tmp_path / "injected_context")
    injected = InjectedContextBuilder()

    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        context_builder=injected,
    )
    recorder.close()

    assert runtime.context_builder is injected


def test_task_runtime_loads_runtime_control_config_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_ENABLED", "true")
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_SEMANTIC_PLAN_UPDATE_ENABLED", "true")
    monkeypatch.setenv("FLYAGENT_RUNTIME_CONTROL_SEMANTIC_PLAN_UPDATE_EXPOSED", "true")
    run_dir = tmp_path / "runtime_control_env_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerStub([CONTINUE_CURRENT_PHASE, COMPLETE_TASK])
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    assert state.completed
    assert planner.contexts[0]["runtime_control"]["available"] == ["semantic_plan_update"]
    assert planner.contexts[0]["runtime_control"]["contracts"] == {
        "semantic_plan_update": (
            "After multiple action-observation cycles, request when confidence is low that the current goal, "
            "done_when, or continue_when still fit observed progress and user intent. Name the doubtful "
            "fields and evidence; keep normal UAV steps if continuing."
        )
    }


def test_task_runtime_recommends_semantic_contract_review_after_multiple_action_windows(tmp_path):
    run_dir = tmp_path / "runtime_control_recommendation_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = AlwaysHoverPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        runtime_loop_guard_windows=8,
        runtime_control_config=RuntimeControlConfig(
            enabled=True,
            actions={"semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True)},
        ),
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result("blocked visual phase", "final_phase"))
    recorder.close()

    assert state.stopped
    recommendation_contexts = [
        context
        for context in planner.contexts
        if context.get("runtime_control", {}).get("recommendations", {}).get("semantic_plan_update")
    ]
    assert recommendation_contexts
    assert recommendation_contexts[0]["runtime_control"]["recommendations"] == {
            "semantic_plan_update": (
                "review_now: multiple action-observation cycles completed; request only if confidence in "
                "goal/done_when/continue_when is now low"
            )
        }


def test_task_runtime_applies_semantic_plan_update_on_later_boundary(tmp_path):
    class SkillContextBuilder:
        def __init__(self):
            self.capability_calls = 0

        def semantic_evidence_capabilities(self):
            self.capability_calls += 1
            return [
                "NOW RGB: current visible appearance and image-space relations.",
                "Action history: phase-relative UAV self-motion; not target distance.",
            ]

        def build(self, state, memory):
            return {
                "active_instruction": state.user_instruction,
                "current_phase": state.semantic_plan.phases[state.current_phase_index].to_dict(),
                "fragments": [
                    {
                        "name": "skills",
                        "content": {
                            "selected_skills": [
                                {
                                    "name": "route_traversal_and_passage",
                                    "skill_text": "Use route progress evidence.",
                                    "requested_references": [
                                        {
                                            "path": "references/route_progress_examples.md",
                                            "text": "must not reach updater",
                                        }
                                    ],
                                    "source_path": "must/not/reach/updater",
                                },
                                {
                                    "name": "route_traversal_and_passage",
                                    "skill_text": "duplicate must be ignored",
                                },
                            ]
                        },
                    }
                ],
            }

    run_dir = tmp_path / "semantic_plan_update_apply_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CompleteAfterSemanticUpdatePlanner()
    updater = FakeSemanticPlanUpdater(
        {
            "phase": {
                "done_when": ["updated completion evidence is available"],
                "continue_when": ["updated completion evidence is not available"],
            }
        }
    )
    context_builder = SkillContextBuilder()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        runtime_loop_guard_windows=20,
        runtime_control_config=RuntimeControlConfig(
            enabled=True,
            actions={"semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True)},
        ),
        semantic_plan_updater=updater,
        context_builder=context_builder,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]

    assert state.completed
    assert state.semantic_plan_version == 1
    assert state.semantic_plan.phases[0].done_when == ["updated completion evidence is available"]
    assert context_builder.capability_calls == 1
    assert runtime.task_parser.configured_evidence == [
        "NOW RGB: current visible appearance and image-space relations.",
        "Action history: phase-relative UAV self-motion; not target distance.",
    ]
    assert updater.contexts
    assert updater.contexts[0]["phase_execution_summary"]["phase_key"] == "1:final_phase"
    assert updater.contexts[0]["selected_skill_guidance"] == [
        {
            "name": "route_traversal_and_passage",
            "skill_text": "Use route progress evidence.",
        }
    ]
    assert updater.contexts[0]["configured_evidence"] == [
        "NOW RGB: current visible appearance and image-space relations.",
        "Action history: phase-relative UAV self-motion; not target distance.",
    ]
    assert "requested_references" not in str(updater.contexts[0]["selected_skill_guidance"])
    assert "source_path" not in str(updater.contexts[0]["selected_skill_guidance"])
    assert any(
        context.get("runtime_control", {}).get("status", {}).get("semantic_plan_update") == "pending"
        for context in planner.contexts
    )
    assert "tool_step_completed" in event_names
    assert "semantic_plan_update_applied" in event_names
    assert any(
        record["event"] == "runtime_control_action_completed"
        and record["payload"].get("name") == "semantic_plan_update"
        and record["payload"].get("status") == "applied"
        for record in events
    )


def test_run_task_refreshes_semantic_evidence_for_each_instruction(tmp_path):
    class MutableEvidenceBuilder:
        def __init__(self):
            self.capabilities = ["visual evidence", "", "visual evidence"]

        def semantic_evidence_capabilities(self):
            return list(self.capabilities)

    class RecordingParser:
        def __init__(self):
            self.configured_evidence = []
            self.evidence_seen_at_parse = []
            self.last_parse_trace = {}

        def configure_evidence_capabilities(self, capabilities):
            self.configured_evidence = list(capabilities)

        def parse(self, instruction):
            self.evidence_seen_at_parse.append(list(self.configured_evidence))
            return make_single_phase_parse_result(instruction)

    run_dir = tmp_path / "dynamic_evidence_refresh"
    recorder = AsyncRunLogger(run_dir)
    parser = RecordingParser()
    context_builder = MutableEvidenceBuilder()
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        parser,
        planner=CoupledPlannerStub([COMPLETE_TASK]),
        context_builder=context_builder,
        shutdown_policy="none",
    )
    # This test isolates the parse boundary: scene execution is covered by the
    # surrounding Runtime tests and does not affect capability discovery.
    runtime.run_parsed_task = lambda *args, **kwargs: args[1]

    runtime.run_task("m1", "first instruction")
    context_builder.capabilities = [
        "action evidence",
        "  optional   map evidence  ",
        "action evidence",
        None,
    ]
    runtime.run_task("m1", "second instruction")
    recorder.close()

    assert parser.evidence_seen_at_parse == [
        ["visual evidence"],
        ["action evidence", "optional map evidence"],
    ]


def test_task_runtime_discards_semantic_update_with_invented_numeric_threshold(tmp_path):
    run_dir = tmp_path / "semantic_plan_update_numeric_rejection_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CompleteAfterSemanticUpdatePlanner()
    updater = FakeSemanticPlanUpdater(
        {
            "phase": {
                "done_when": ["updated completion evidence persists for 3 observations"],
            }
        }
    )
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        runtime_loop_guard_windows=4,
        runtime_control_config=RuntimeControlConfig(
            enabled=True,
            actions={"semantic_plan_update": RuntimeControlActionConfig(enabled=True, exposed=True)},
        ),
        semantic_plan_updater=updater,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    discarded = [
        record
        for record in load_event_records(run_dir)
        if record["event"] == "semantic_plan_update_discarded"
    ]
    assert state.stopped
    assert state.semantic_plan_version == 0
    assert discarded
    assert discarded[0]["payload"]["discard_reason"] == "invalid_patch"
    assert "unsupported numeric" in discarded[0]["payload"]["error"]


def test_task_runtime_can_land_on_shutdown(tmp_path):
    run_dir = tmp_path / "task_run_land"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="land",
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    mission_state = json.loads((run_dir / "state/latest_state.json").read_text(encoding="utf-8"))

    assert state.completed
    assert state.connected
    assert state.airborne is False
    assert mission_state["airborne"] is False
    assert "recent_actions" not in mission_state
    assert any(record["event"] == "shutdown_land_completed" for record in events)
    assert "shutdown_land_started" in event_names
    assert "shutdown_land_completed" in event_names


def test_task_runtime_can_reuse_existing_parse_result(tmp_path):
    class ExplodingParser:
        def parse(self, instruction: str):
            raise AssertionError("parser should not be called")

    parse_result = TaskParser(mode="rule_based").parse(ENTRANCE_TASK)
    run_dir = tmp_path / "parsed_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(SimulatorEnvClient(server), recorder, ExplodingParser(), planner=HoverOnlyPlanner())

    state = runtime.run_parsed_task("m1", parse_result)
    recorder.close()

    semantic_plan = json.loads((run_dir / "state/semantic_plan.json").read_text(encoding="utf-8"))
    assert state.completed
    assert semantic_plan["instruction"] == ENTRANCE_TASK
    assert semantic_plan["plan"] == parse_result.plan.to_dict()


def test_task_runtime_coupled_advance_skips_tool_and_phase_judge(tmp_path):
    run_dir = tmp_path / "coupled_advance_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerStub([ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK])
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        enable_action_planner_debug=True,
    )

    state = runtime.run_parsed_task("m1", make_two_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    debug_payload = json.loads(sorted((run_dir / "debug" / "action_planner").glob("*.json"))[0].read_text(encoding="utf-8"))

    assert state.completed
    assert planner.judge_calls == 0
    assert "coupled_planner_phase_decided" in event_names
    assert "model_phase_transition_judged" not in event_names
    assert "tool_step_completed" not in event_names
    assert debug_payload["coupled_phase_decision"]["decision"] == ADVANCE_TO_NEXT_PHASE
    applied = [record for record in events if record["event"] == "phase_transition_applied"]
    assert applied[0]["payload"]["transition_source"] == "coupled_planner"


def test_task_runtime_coupled_complete_final_phase(tmp_path):
    run_dir = tmp_path / "coupled_complete_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerStub([COMPLETE_TASK])
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    applied = [record for record in events if record["event"] == "phase_transition_applied"]

    assert state.completed
    assert planner.judge_calls == 0
    assert applied[-1]["payload"]["decision"] == COMPLETE_TASK
    assert applied[-1]["payload"]["transition_source"] == "coupled_planner"


def test_task_runtime_coupled_complete_before_final_downgrades_to_advance(tmp_path):
    run_dir = tmp_path / "coupled_downgrade_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerStub([COMPLETE_TASK, COMPLETE_TASK])
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_two_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    applied = [record for record in events if record["event"] == "phase_transition_applied"]

    assert state.completed
    assert applied[0]["payload"]["original_decision"] == COMPLETE_TASK
    assert applied[0]["payload"]["applied_decision"] == ADVANCE_TO_NEXT_PHASE
    assert applied[0]["payload"]["transition_source"] == "coupled_planner"


def test_task_runtime_coupled_continue_executes_tool_and_observes(tmp_path):
    run_dir = tmp_path / "coupled_continue_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerStub([CONTINUE_CURRENT_PHASE, COMPLETE_TASK])
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    continued = [record for record in events if record["event"] == "phase_transition_continued"]

    assert state.completed
    assert planner.judge_calls == 0
    assert "tool_step_completed" in event_names
    assert "automatic_observation_updated" in event_names
    assert continued[0]["payload"]["transition_source"] == "coupled_planner"
    assert "phase_progress_evidence" in planner.contexts[0]
    assert "motion_progress" in planner.contexts[0]["phase_progress_evidence"]
    first_action = next(fragment for fragment in planner.contexts[0]["fragments"] if fragment["name"] == "action")
    second_action = next(fragment for fragment in planner.contexts[1]["fragments"] if fragment["name"] == "action")
    assert "phase_strategy" not in first_action["content"]
    assert "phase_strategy" not in second_action["content"]
    assert "phase_strategy" not in planner.contexts[0]
    assert planner.contexts[1]["phase_strategy"] == {
        "stages": ["Hold until the current phase evidence is refreshed"],
        "active_stage": 0,
        "previous_reason": "The current phase still requires evidence",
    }
    assert state.current_phase_strategy_revision == 1
    assert state.current_phase_strategy["stages"] == [
        "Hold until the current phase evidence is refreshed"
    ]
    assert "phase_strategy_updated" in event_names
    strategy_event = next(record for record in events if record["event"] == "phase_strategy_updated")
    assert strategy_event["payload"]["update_kind"] == "initialize"
    phase_window = next(record for record in events if record["event"] == "phase_window_recorded")
    assert phase_window["payload"]["phase_strategy_revision"] == 1
    phase_window_record = json.loads(
        (run_dir / phase_window["payload"]["debug_ref"]).read_text(encoding="utf-8")
    )
    assert phase_window_record["phase_strategy"]["active_stage"] == 0
    assert "model_phase_transition_judged" not in event_names


def test_task_runtime_next_planner_context_receives_previous_window_motion(tmp_path):
    class MoveThenCompletePlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.contexts = []

        def plan_next_with_phase_decision(self, state, context=None):
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return CoupledActionPhasePlan.from_dict(
                    {
                        "phase": state.current_phase_name,
                        "phase_decision": {
                            "decision": CONTINUE_CURRENT_PHASE,
                            "reason": "verify measured lateral motion in the next window",
                        },
                        "action_window": {
                            "intent": "move laterally once",
                            "scope": "current_phase_only",
                            "requires_intermediate_observation": True,
                            "expected_effect": "the next observation records leftward UAV motion",
                        },
                        "agent_messages": [],
                        "phase_strategy": {
                            "stages": ["Move laterally until the new viewpoint is established"],
                            "active_stage": 0,
                            "reason": "The new viewpoint has not yet been established",
                        },
                        "steps": [
                            {
                                "kind": "tool",
                                "tool": "uav.move_body",
                                "args": {"vx": 0.0, "vy": -1.0, "vz": 0.0, "duration": 1.0},
                                "reason": "create a measurable lateral displacement",
                            }
                        ],
                    },
                    require_agent_messages=True,
                )
            return CoupledActionPhasePlan.from_dict(
                {
                    "phase": state.current_phase_name,
                    "phase_decision": {"decision": COMPLETE_TASK, "reason": "motion context captured"},
                    "agent_messages": [],
                    "steps": [],
                },
                require_agent_messages=True,
            )

    run_dir = tmp_path / "last_window_motion_handoff"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = MoveThenCompletePlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    # Build against the live compact semantic schema; this regression concerns
    # Runtime motion handoff, not optional semantic metric fixtures.
    parsed = TaskParseResult(
        instruction="measure previous action window",
        parser_mode="test",
        plan=SemanticPlan(
            intent="measure_window_motion",
            task_goal=TaskGoal(target="new viewpoint", final_state="motion recorded"),
            phases=[
                SemanticPhase(
                    index=1,
                    total=1,
                    name="measure_motion",
                    goal="record one lateral action window",
                    done_when=["the lateral action window is recorded"],
                    continue_when=["the lateral action window has not been recorded"],
                )
            ],
        ),
    )
    state = runtime.run_parsed_task("m1", parsed)
    recorder.close()

    assert state.completed
    assert len(planner.contexts) == 2
    first_action = next(
        fragment for fragment in planner.contexts[0]["fragments"] if fragment["name"] == "action"
    )
    second_action = next(
        fragment for fragment in planner.contexts[1]["fragments"] if fragment["name"] == "action"
    )
    assert first_action["content"]["motion_history_summary"]["last_window_uav_motion"] == {
        "forward_m": 0.0,
        "right_m": 0.0,
        "up_m": 0.0,
        "planar_m": 0.0,
        "yaw_delta_deg": 0.0,
    }
    last_window = second_action["content"]["motion_history_summary"]["last_window_uav_motion"]
    assert last_window == {
        "forward_m": 0.0,
        "right_m": -1.0,
        "up_m": 0.0,
        "planar_m": 1.0,
        "yaw_delta_deg": 0.0,
    }
    assert "recent_execution" not in second_action["content"]["motion_history_summary"]

    events = load_event_records(run_dir)
    first_window = next(record for record in events if record["event"] == "phase_window_recorded")
    first_window_record = json.loads(
        (run_dir / first_window["payload"]["debug_ref"]).read_text(encoding="utf-8")
    )
    assert last_window["right_m"] == first_window_record["measurements"]["right_displacement_m"]
    assert last_window["planar_m"] == first_window_record["measurements"]["planar_displacement_m"]


def test_task_runtime_phase_progress_evidence_marks_yaw_only_approach_progress():
    state = MissionState(
        run_id="run_yaw_only",
        user_instruction="approach the sunshade",
        semantic_plan=make_approach_parse_result().plan,
    )
    state.current_phase_progress = {
        "phase_key": "1:approach_sunshade",
        "phase": "approach_sunshade",
        "window_count": 3,
        "measurements": {
            "forward_m": 0.0,
            "right_m": 0.0,
            "up_m": 0.0,
            "planar_m": 0.0,
            "yaw_delta_deg": 18.0,
        },
        "tool_facts": {
            "turn_yaw_success_sum_deg": 18.0,
            "move_duration_success_sum_sec": 0.0,
            "tools_executed": {"uav.turn_relative": 3},
        },
    }
    evidence = PhaseWindowContextBuilder().phase_progress_evidence(state)
    progress = evidence["motion_progress"]

    assert progress["measurement_scope"] == "phase_start_relative_uav_motion"
    assert "not distance-to-target" in progress["interpretation_note"]
    assert "not distance-to-landmark" in progress["interpretation_note"]
    assert progress["has_translation"] is False
    assert progress["yaw_only_so_far"] is True
    assert progress["tools_executed"] == {"uav.turn_relative": 3}
    assert "done_when_support" not in evidence
    assert "model_guidance" not in evidence


def test_task_runtime_phase_progress_evidence_marks_translation_progress():
    state = MissionState(
        run_id="run_move",
        user_instruction="approach the sunshade",
        semantic_plan=make_approach_parse_result().plan,
    )
    state.current_phase_progress = {
        "phase_key": "1:approach_sunshade",
        "phase": "approach_sunshade",
        "window_count": 1,
        "measurements": {
            "forward_m": 2.0,
            "right_m": 0.0,
            "up_m": 0.0,
            "planar_m": 2.0,
            "yaw_delta_deg": 0.0,
        },
        "tool_facts": {
            "turn_yaw_success_sum_deg": 0.0,
            "move_duration_success_sum_sec": 1.0,
            "tools_executed": {"uav.move_body": 1},
        },
    }
    evidence = PhaseWindowContextBuilder().phase_progress_evidence(state)
    progress = evidence["motion_progress"]

    assert progress["measurement_scope"] == "phase_start_relative_uav_motion"
    assert "UAV motion progress from the current phase start" in progress["interpretation_note"]
    assert progress["has_translation"] is True
    assert progress["yaw_only_so_far"] is False
    assert progress["forward_m"] == 2.0
    assert progress["planar_m"] == 2.0
    assert progress["tools_executed"] == {"uav.move_body": 1}
    assert "done_when_support" not in evidence
    assert "model_guidance" not in evidence


def test_task_runtime_default_mode_uses_coupled_planner_decision(tmp_path):
    run_dir = tmp_path / "default_judge_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = HoverOnlyPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]

    assert state.completed
    assert "coupled_planner_phase_decided" in event_names
    assert "model_phase_transition_judged" not in event_names
    applied = [record for record in events if record["event"] == "phase_transition_applied"]
    assert applied[-1]["payload"]["transition_source"] == "coupled_planner"


def test_task_runtime_model_planner_path_records_model_action_and_safety(tmp_path):
    class FakeModelPlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.contexts = []

        def plan_next_with_phase_decision(self, state, context=None):
            self.contexts.append(context)
            plan = ShortHorizonPlan(
                state.current_phase_name,
                [
                    PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 999.0}, "oversized model suggestion"),
                ],
            )
            return coupled_continue_from_short_plan(plan, "keep testing model plan safety review")

    run_dir = tmp_path / "model_planner_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = FakeModelPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        runtime_loop_guard_windows=2,
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    planned_records = [record for record in events if record["event"] == "short_horizon_planned"]

    assert not state.completed
    assert state.stopped
    assert state.stopped_reason == "runtime_loop_guard_reached"
    assert planner.contexts
    assert planner.contexts[0]["current_phase"]["name"] == "orient_right"
    assert planner.contexts[0]["phase_neighborhood"]["previous"] is None
    assert planner.contexts[0]["phase_neighborhood"]["next"] == {
        "index": 2,
        "name": "search_landmark",
        "goal": "identify the black street lamp candidate",
        "status": "upcoming",
    }
    assert "fragments" in planner.contexts[0]
    assert "context_schema" in planner.contexts[0]
    assert "module_content_fields" in planner.contexts[0]["context_schema"]
    assert "image" in planner.contexts[0]["context_schema"]["module_content_fields"]
    assert "selected_frames" in planner.contexts[0]["context_schema"]["module_content_fields"]["image"]
    action_constraints = planner.contexts[0]["allowed_action_constraints"]
    assert action_constraints["planner_tools"] == [
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
        "skill.visual_grounding.locate",
    ]
    assert "observe_step_schema" not in action_constraints
    assert planner.contexts[0]["observation_policy"]["mode"] == "automatic"
    assert set(action_constraints["action_schemas"]) == {
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
        "skill.visual_grounding.locate",
    }
    assert action_constraints["action_schemas"]["uav.turn_relative"]["limits"] == {
        "yaw_deg_min": -30.0,
        "yaw_deg_max": 30.0,
    }
    assert action_constraints["action_schemas"]["uav.move_body"]["limits"] == {
        "abs_vx_vy_max": 6.0,
        "abs_vz_max": 3.0,
        "duration_max": 8.0,
    }
    assert "forbidden_tools" not in action_constraints
    assert "flight_memory" not in planner.contexts[0]
    assert "instruction_history" not in planner.contexts[0]
    assert "recent_observations" not in planner.contexts[0]
    assert "model_action_planned" in event_names
    assert "safety_intervened" in event_names
    assert "automatic_observation_updated" in event_names
    assert "phase_transition_suppressed_by_safety" in event_names
    assert "runtime_loop_guard_reached" in event_names
    assert "phase_transition_window_limit_reached" in event_names
    assert planned_records[0]["payload"]["steps"][0]["args"]["yaw_deg"] == 30.0
    latest_agent_message = json.loads((run_dir / "state/latest_agent_message.json").read_text(encoding="utf-8"))
    agent_message_lines = [
        json.loads(line)
        for line in (run_dir / "memory/agent_messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert latest_agent_message["message"]["type"] == "warning"
    assert latest_agent_message["message_ref"]["ref"].startswith("memory/agent_messages.jsonl#")
    assert agent_message_lines
    assert {record["message"]["type"] for record in agent_message_lines} == {"warning"}
    assert not (run_dir / "state/final_answer.json").exists()


def test_task_runtime_action_planner_debug_redacts_non_provider_model_input_images(tmp_path):
    class OneTurnPlanner:
        planner_mode = "model"

        def plan_next_with_phase_decision(self, state, context=None):
            assert isinstance(context, dict)
            context.setdefault("fragments", []).append(
                {
                    "name": "test_image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": (
                                "data:image/png;base64,"
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4n"
                                "GNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
                            ),
                        }
                    ],
                }
            )
            plan = ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 5.0}, "debug turn")],
            )
            return coupled_continue_from_short_plan(plan, "write planner debug for current phase")

    run_dir = tmp_path / "planner_debug_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=OneTurnPlanner(),
        runtime_loop_guard_windows=1,
        enable_action_planner_debug=True,
    )

    state = runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    debug_files = sorted((run_dir / "debug" / "action_planner").glob("*.json"))
    assert debug_files
    payload = json.loads(debug_files[0].read_text(encoding="utf-8"))
    assert payload["planner_context"]["active_instruction"] == LAMP_TASK
    assert payload["model_context"]["active_instruction"] == LAMP_TASK
    assert payload["raw_plan"]["steps"][0]["tool"] == "uav.turn_relative"
    assert payload["safe_plan"]["steps"][0]["tool"] == "uav.turn_relative"
    assert "safety_decision" in payload
    assert "model_request_recovery" not in payload
    debug_text = debug_files[0].read_text(encoding="utf-8")
    assert "data:image" not in debug_text
    assert "[inline image omitted from planner context]" in debug_text
    assert state.completed or state.stopped


def test_task_runtime_records_agent_messages_before_and_after_execution(tmp_path):
    class MessagePlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.calls = 0

        def plan_next_with_phase_decision(self, state, context=None):
            self.calls += 1
            if self.calls == 1:
                return CoupledActionPhasePlan(
                    state.current_phase_name,
                    PhaseTransition(CONTINUE_CURRENT_PHASE, "continue once"),
                    [PlannedStep.tool_step("uav.hover", {}, "hold")],
                    agent_messages=[
                        AgentMessage("progress", "我会先悬停并采集新的观察结果。", confidence="medium"),
                    ],
                )
            return CoupledActionPhasePlan(
                state.current_phase_name,
                PhaseTransition(COMPLETE_TASK, "final state reached"),
                [],
                agent_messages=[
                    AgentMessage("final", "任务完成，我已保持悬停。", confidence="high"),
                ],
            )

    run_dir = tmp_path / "agent_message_events"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    live_events = []
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=MessagePlanner(),
        shutdown_policy="none",
        event_sink=live_events.append,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result("agent message task", "hold_position"))
    turn_messages = runtime.latest_turn_agent_messages()
    recorder.close()

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_names = [record["event"] for record in events]
    message_records = [record for record in events if record["event"] == "agent_message"]
    agent_message_lines = [
        json.loads(line)
        for line in (run_dir / "memory/agent_messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    latest_agent_message = json.loads((run_dir / "state/latest_agent_message.json").read_text(encoding="utf-8"))
    final_answer = json.loads((run_dir / "state/final_answer.json").read_text(encoding="utf-8"))
    flight_memory = json.loads((run_dir / "state/flight_memory.json").read_text(encoding="utf-8"))

    assert state.completed
    assert [message["type"] for message in turn_messages] == ["progress", "final"]
    assert turn_messages[-1]["content"] == "任务完成，我已保持悬停。"
    assert [record["payload"]["message"]["type"] for record in message_records] == ["progress", "final"]
    assert [record["message"]["type"] for record in agent_message_lines] == ["progress", "final"]
    assert [record["id"] for record in agent_message_lines] == ["msg_0001", "msg_0002"]
    assert all(record["payload"]["message_ref"]["ref"].startswith("memory/agent_messages.jsonl#") for record in message_records)
    assert latest_agent_message["message"]["type"] == "final"
    assert latest_agent_message["message"]["content"] == "任务完成，我已保持悬停。"
    assert latest_agent_message["message_ref"]["ref"] == "memory/agent_messages.jsonl#2"
    assert final_answer == latest_agent_message
    assert [ref["ref"] for ref in flight_memory["recent_agent_message_refs"]] == [
        "memory/agent_messages.jsonl#1",
        "memory/agent_messages.jsonl#2",
    ]
    assert "任务完成" not in json.dumps(flight_memory, ensure_ascii=False)
    assert [event["kind"] for event in live_events if event["kind"] == "agent_message"] == [
        "agent_message",
        "agent_message",
    ]
    assert any(event["kind"] == "instruction_started" for event in live_events)
    assert any(event["kind"] == "phase_plan_ready" for event in live_events)
    assert any(event["kind"] == "tool_step_completed" for event in live_events)
    assert any(event["kind"] == "observation_updated" for event in live_events)
    assert any(event["kind"] == "phase_transition_applied" for event in live_events)
    transition_payload = next(event["payload"] for event in live_events if event["kind"] == "phase_transition_applied")
    transition_payload_text = json.dumps(transition_payload, ensure_ascii=False)
    assert "phase_completion_evaluation" not in transition_payload
    assert "phase_metrics" not in transition_payload
    assert "planner_context" not in transition_payload
    assert "data:image" not in transition_payload_text
    assert "base64" not in transition_payload_text
    assert event_names.index("agent_message") < event_names.index("phase_plan_ready")
    automatic_observation_indexes = [
        index
        for index, record in enumerate(events)
        if record["event"] == "observation_updated"
        and record.get("payload", {}).get("source") == "automatic_phase_window"
    ]
    assert automatic_observation_indexes[-1] < max(
        index for index, name in enumerate(event_names) if name == "agent_message"
    )


def test_task_runtime_agent_message_cache_resets_between_instructions(tmp_path):
    class InstructionMessagePlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.calls_by_instruction = {}

        def plan_next_with_phase_decision(self, state, context=None):
            calls = self.calls_by_instruction.get(state.user_instruction, 0)
            self.calls_by_instruction[state.user_instruction] = calls + 1
            if calls == 0:
                return CoupledActionPhasePlan(
                    state.current_phase_name,
                    PhaseTransition(CONTINUE_CURRENT_PHASE, "continue once"),
                    [PlannedStep.tool_step("uav.hover", {}, "hold")],
                    agent_messages=[
                        AgentMessage("progress", f"开始执行：{state.user_instruction}", confidence="medium"),
                    ],
                )
            return CoupledActionPhasePlan(
                state.current_phase_name,
                PhaseTransition(COMPLETE_TASK, "done"),
                [],
                agent_messages=[AgentMessage("final", "本轮完成。", confidence="high")],
            )

    run_dir = tmp_path / "agent_message_cache_reset"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=InstructionMessagePlanner(),
        shutdown_policy="none",
    )

    runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    first_messages = runtime.latest_turn_agent_messages()
    runtime.run_task("m1", ENTRANCE_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    second_messages = runtime.latest_turn_agent_messages()
    recorder.close()

    assert any(LAMP_TASK in message["content"] for message in first_messages)
    assert not any(LAMP_TASK in message["content"] for message in second_messages)
    assert any(ENTRANCE_TASK in message["content"] for message in second_messages)


def test_task_runtime_passes_action_config_to_planner_context(tmp_path):
    class ContextCapturePlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.contexts = []

        def plan_next_with_phase_decision(self, state, context=None):
            self.contexts.append(context)
            return CoupledActionPhasePlan(
                state.current_phase_name,
                PhaseTransition(COMPLETE_TASK, "context captured"),
                [],
            )

    run_dir = tmp_path / "action_config_context"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = ContextCapturePlanner()
    action_config = UavActionConfig(linear_velocity_mps=3.5, vertical_velocity_mps=1.25, move_duration_sec=1.75)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        action_config=action_config,
    )

    runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    nominal_scale = planner.contexts[0]["allowed_action_constraints"]["nominal_action_scale"]
    assert nominal_scale["linear_velocity_mps"] == 3.5
    assert nominal_scale["vertical_velocity_mps"] == 1.25
    assert "move_duration_sec" not in nominal_scale


def test_task_runtime_reuse_running_scene_does_not_repeat_preflight(tmp_path):
    run_dir = tmp_path / "continuous_task_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

    first_state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    second_state = runtime.run_task("m1", ENTRANCE_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    runtime.finish_reused_scene("m1", second_state)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    scene_started = [record for record in events if record["event"] == "scene_started"]
    scene_reused = [record for record in events if record["event"] == "scene_reused"]

    assert first_state.completed
    assert second_state.completed
    assert second_state.connected
    assert second_state.airborne
    assert second_state.latest_observation["source"] == "automatic_phase_window"
    flight_memory = json.loads((run_dir / "state/flight_memory.json").read_text(encoding="utf-8"))
    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))
    assert len(flight_memory["instruction_history"]) >= 2
    assert flight_memory["active_instruction"] == ENTRANCE_TASK
    assert flight_memory["recent_action_refs"]
    assert flight_memory["recent_observation_refs"]
    assert (run_dir / "memory/instructions.jsonl").read_text(encoding="utf-8").count("\n") >= 2
    assert (run_dir / "memory/actions.jsonl").exists()
    assert (run_dir / "memory/observations.jsonl").exists()
    assert (run_dir / "memory/poses.jsonl").exists()
    assert (run_dir / "memory/frames.jsonl").exists()
    assert latest_context["active_instruction"] == ENTRANCE_TASK
    assert latest_context["metadata"]["run_id"] == flight_memory["run_id"]
    assert "fragments" in latest_context
    assert "instruction_history" not in latest_context
    assert "recent_actions" not in latest_context
    assert event_names.count("control_connected") == 1
    assert event_names.count("preflight_takeoff_completed") == 1
    assert "instruction_start_observation_updated" in event_names
    assert len(scene_started) == 1
    assert len(scene_reused) == 1
    assert scene_started[0]["payload"]["scene_instance_id"] == scene_reused[0]["payload"]["scene_instance_id"]
    assert event_names.count("scene_stopped") == 1


def test_task_runtime_can_leave_reused_scene_running(tmp_path):
    run_dir = tmp_path / "leave_scene_running"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    running_scene = server.get_or_start_scene("m1")

    assert state.completed
    assert "scene_stopped" not in event_names
    assert running_scene.status == "running"


def test_task_runtime_model_planner_keeps_memory_module_windows_across_many_instructions(tmp_path):
    class ContextEchoModelPlanner(ModelActionPlanner):
        def __init__(self) -> None:
            self.contexts = []
            super().__init__(
                ModelConfig(
                    task_parser_mode="model",
                    model_name="stub",
                    api_key="test-key",
                    base_url="https://example.test/v1",
                )
            )

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            context = transcript.latest_context()
            self.contexts.append(context)
            call_index = len(self.contexts)
            return AssistantTurn(
                turn_id=f"assistant_{call_index}",
                public_intent="The context verification phase is complete.",
                tool_calls=[AgentToolCall(
                    f"commit_{call_index}",
                    "runtime.commit_phase_turn",
                    {
                        "phase": context["current_phase"]["name"],
                        "completion_assessment": {
                    "done_when": [
                        {
                            "index": index,
                            "status": "supported",
                            "evidence": "the context verification condition is satisfied",
                        }
                        for index, _ in enumerate(context["current_phase"]["done_when"])
                    ]
                        },
                    },
                )],
            )

    instructions = [
        "右转寻找黑色路灯",
        "左转寻找建筑入口",
        "寻找红色车辆",
        "继续寻找黑色路灯",
        "确认建筑入口位置",
        "观察红色车辆附近区域",
        "面向黑色路灯后悬停",
        "停在建筑入口前",
        "靠近红色车辆并保持安全距离",
        "右转看有没有黑色路灯",
        "左转寻找建筑入口并停下",
        "寻找红色车辆然后悬停",
    ]
    run_dir = tmp_path / "many_instruction_model_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = ContextEchoModelPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        model_config=planner.model_config,
        context_builder=ObservationContextBuilder(
            skill_selector=ModelSkillSelector(enabled=False)
        ),
    )

    state = None
    for instruction in instructions:
        state = runtime.run_task("m1", instruction, scene_policy="reuse_running", stop_scene_on_finish=False)
    runtime.finish_reused_scene("m1", state)
    recorder.close()

    latest_context = json.loads((run_dir / "contexts/latest_context.json").read_text(encoding="utf-8"))
    flight_memory = json.loads((run_dir / "state/flight_memory.json").read_text(encoding="utf-8"))
    events = load_event_records(run_dir)
    fragments = {fragment["name"]: fragment for fragment in latest_context["fragments"]}

    assert state is not None and state.completed
    assert len(planner.contexts) >= len(instructions)
    assert latest_context["active_instruction"] == instructions[-1]
    assert "context_schema" in latest_context
    assert "action" in latest_context["context_schema"]["module_content_fields"]
    assert len(flight_memory["instruction_history"]) >= len(instructions)
    assert (run_dir / "memory/instructions.jsonl").read_text(encoding="utf-8").count("\n") >= len(instructions)
    assert fragments["instruction"]["content"]["current_instruction"]["text"] == instructions[-1]
    instruction_history = fragments["instruction"]["content"]["instruction_history"]
    assert len(instruction_history) == 5
    assert [item["instruction"] for item in instruction_history] == instructions[-6:-1]
    assert all(item["id"] != flight_memory["active_instruction_id"] for item in instruction_history)
    motion_history = fragments["action"]["content"]["motion_history_summary"]
    assert "recent_execution" not in motion_history
    assert "recent_execution_order" not in motion_history
    assert fragments["pose"]["metadata"]["pose_history_count"] <= 3
    assert fragments["image"]["metadata"]["observation_count"] <= 3
    assert fragments["image"]["artifact_refs"]
    assert fragments["map"]["content"]["recalled_map_memories"] == []
    assert "flight_memory" not in latest_context
    assert "recent_actions" not in latest_context
    assert "instruction_history" not in latest_context
    event_names = [record["event"] for record in events]
    assert "phase_completion_assessed" in event_names
    assert "model_action_planned" in event_names


def test_native_tool_loop_reasons_with_grounding_result_and_batches_physical_calls(
    tmp_path, monkeypatch
):
    class NativeSequencePlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.calls = 0
            self.allowed_names = []
            self.observation_ids = []
            self.provider_roles = []
            self.checkpoint_selected_roles = []

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            self.calls += 1
            self.last_exchange = {
                "transport": {
                    "attempt_count": 1,
                    "retry_count": 0,
                    "final_status": "success",
                    "retries": [],
                }
            }
            self.allowed_names.append(set(allowed_tool_names or set()))
            self.observation_ids.append(transcript.latest_observation_id())
            provider_messages = transcript.provider_messages(registry)
            self.provider_roles.append([message["role"] for message in provider_messages])
            latest_context = transcript.latest_context()
            image_fragment = next(
                (
                    fragment
                    for fragment in latest_context.get("fragments", [])
                    if fragment.get("name") == "image"
                ),
                {},
            )
            self.checkpoint_selected_roles.append(
                [
                    item.get("role")
                    for item in image_fragment.get("content", {}).get(
                        "selected_frames", []
                    )
                ]
            )
            observation_id = transcript.latest_observation_id()
            if self.calls == 1:
                return AssistantTurn(
                    "assistant_detect",
                    "I will confirm the person in the current observation.",
                    [
                        AgentToolCall(
                            "call_detect",
                            "skill.visual_grounding.locate",
                            {
                                "observation_id": observation_id,
                                "query": "the person to approach",
                                "include_distance": False,
                            },
                        )
                    ],
                )
            if self.calls == 2:
                tool_results = [
                    item.payload
                    for item in transcript.items
                    if item.kind == "tool_result"
                ]
                grounding_result = next(
                    result for result in tool_results if result.tool_call_id == "call_detect"
                )
                assert grounding_result.status == "ok"
                assert grounding_result.data["observation_id"] == observation_id
                assert grounding_result.data["candidates"][0]["candidate_id"] == "candidate_01"
                assert grounding_result.artifact_refs
                return AssistantTurn(
                    "assistant_move",
                    "The detector confirmed a person, so I will turn and approach in one window.",
                    [
                        AgentToolCall(
                            "call_commit_move",
                            "runtime.commit_phase_turn",
                            {
                                "phase": "final_phase",
                                "completion_assessment": {
                                    "done_when": [
                                        {
                                            "index": 0,
                                            "status": "unknown",
                                            "evidence": "The person is detected but the movement is not yet observed.",
                                        }
                                    ]
                                },
                                "phase_strategy": {
                                    "stages": ["confirm person", "turn and approach"],
                                    "active_stage": 1,
                                    "reason": "Detection supports a bounded approach.",
                                },
                                "action_window": {
                                    "intent": "turn and approach the detected person",
                                    "expected_effect": "the UAV changes bearing and translates forward",
                                },
                            },
                        ),
                        AgentToolCall(
                            "call_turn",
                            "uav.turn_relative",
                            {"yaw_deg": 5.0},
                        ),
                        AgentToolCall(
                            "call_move",
                            "uav.move_body",
                            {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                        ),
                    ],
                )
            return AssistantTurn(
                "assistant_complete",
                "The observed action window completed the requested phase.",
                [
                    AgentToolCall(
                        "call_commit_done",
                        "runtime.commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "supported",
                                        "evidence": "The post-action checkpoint confirms the completed window.",
                                    }
                                ]
                            },
                        },
                    )
                ],
            )

    plugin_registry = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.visual_grounding"],
            }
        }
    )
    def fake_grounding_handler(payload):
        assert payload["settings"]["grounding_backend"] == "grounding_dino"
        observation = payload["runtime_context"]["latest_observation"]
        rgb = observation["frames"]["rgb"]
        artifact = str(rgb.get("path") or rgb.get("artifact_path"))
        return SkillCapabilityResult(
            message="Located 1 candidate(s).",
            data={
                "observation_id": observation["observation_id"],
                "query": payload["args"]["query"],
                "include_distance": False,
                "candidates": [
                    {
                        "candidate_id": "candidate_01",
                        "label": "person",
                        "confidence": 0.91,
                        "box_normalized": [0.4, 0.2, 0.6, 0.8],
                        "distance_m": None,
                        "distance_status": "not_requested",
                    }
                ],
            },
            artifact_refs=(artifact,),
            model_inputs=(
                {
                    "type": "image",
                    "role": "visual_grounding_candidates",
                    "observation_id": observation["observation_id"],
                    "path": artifact,
                    "artifact_ref": artifact,
                    "mime_type": "image/png",
                },
            ),
        )

    monkeypatch.setattr(
        skill_dispatcher_module,
        "_load_handler_module",
        lambda handler_path: type(
            "HandlerModule", (), {"handle": staticmethod(fake_grounding_handler)}
        )(),
    )
    run_dir = tmp_path / "native_tool_loop"
    recorder = AsyncRunLogger(run_dir)
    planner = NativeSequencePlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        plugin_registry=plugin_registry,
        enable_action_planner_debug=True,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    transcript_records = [
        json.loads(line)
        for line in (run_dir / "memory" / "agent_transcript.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    action_records = [
        json.loads(line)
        for line in (run_dir / "memory" / "actions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    observation_records = [
        json.loads(line)
        for line in (run_dir / "memory" / "observations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    events = load_event_records(run_dir)

    assert state.completed
    assert state.current_phase_window_count == 1
    assert planner.calls == 3
    assert all(
        "skill.visual_grounding.locate" in names
        and "runtime.commit_phase_turn" in names
        for names in planner.allowed_names
    )
    assert planner.observation_ids[0] == planner.observation_ids[1]
    assert planner.observation_ids[2] != planner.observation_ids[1]
    assert planner.provider_roles[1][-2:] == ["assistant", "tool"]
    assert planner.checkpoint_selected_roles == [
        ["current_observation"],
        ["current_observation"],
        ["current_observation"],
    ]
    assert [record["tool"] for record in action_records].count(
        "skill.visual_grounding.locate"
    ) == 1
    assert [record["tool"] for record in action_records].count("uav.turn_relative") == 1
    assert [record["tool"] for record in action_records].count("uav.move_body") == 1
    assert len(observation_records) == 2
    instruction_event = next(
        record for record in events if record["event"] == "instruction_started"
    )
    assert instruction_event["payload"]["physical_trace_schema"] == (
        "physical_window_v2"
    )
    window_event = next(
        record for record in events if record["event"] == "phase_window_recorded"
    )
    window_payload = window_event["payload"]
    assert window_payload["schema_version"] == "physical_window_v2"
    assert window_payload["observation_before_id"] == observation_records[0]["id"]
    assert window_payload["observation_after_id"] == observation_records[1]["id"]
    assert [item["tool_call_id"] for item in window_payload["ordered_steps"]] == [
        "call_turn",
        "call_move",
    ]
    assert all(item["effect"] == "physical" for item in window_payload["ordered_steps"])
    assert all(item["action_ref"] for item in window_payload["ordered_steps"])
    trace = EpisodeTraceReader().read(TraceCursor.freeze(run_dir))
    assert trace.trace_integrity == "ok"
    assert trace.physical_call_count == 2
    assert trace.physical_window_count == 1
    assert [sample.tool for sample in trace.samples[1:]] == [
        "uav.turn_relative",
        "uav.move_body",
    ]
    flight_memory = json.loads(
        (run_dir / "state" / "flight_memory.json").read_text(encoding="utf-8")
    )
    assert flight_memory["observation_anchors"]["before_last_action_window"]["id"] == (
        observation_records[0]["id"]
    )
    assert [record["kind"] for record in transcript_records].count(
        "state_checkpoint"
    ) == 2
    result_ids = {
        record["payload"]["tool_call_id"]
        for record in transcript_records
        if record["kind"] == "tool_result"
    }
    assert {
        "call_detect",
        "call_commit_move",
        "call_turn",
        "call_move",
        "call_commit_done",
    } <= result_ids
    decisions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "debug" / "action_planner").glob("*.json"))
    ]
    assert [len(item["model_rounds"]) for item in decisions] == [2, 1]
    assert all(
        model_round["transport"]
        == {
            "attempt_count": 1,
            "retry_count": 0,
            "final_status": "success",
            "retries": [],
        }
        for decision in decisions
        for model_round in decision["model_rounds"]
    )
    assert [
        item["runtime_outcome"]["status"]
        for item in decisions[0]["model_rounds"]
    ] == ["read_only_executed", "commit_accepted"]
    assert decisions[0]["model_rounds"][0]["runtime_outcome"]["tool_results"][0][
        "tool_call_id"
    ] == "call_detect"


def test_native_read_only_quota_rejects_duplicates_and_resets_after_success(
    tmp_path, monkeypatch
):
    class QuotaPlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.calls = 0

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            self.calls += 1
            observation_id = transcript.latest_observation_id()
            queries = {
                1: "candidate wording 1",
                2: "candidate wording 1",
                3: "candidate wording 2",
                4: "candidate wording 3",
            }
            if self.calls <= 4:
                return AssistantTurn(
                    f"assistant_ground_{self.calls}",
                    f"Try grounding query {self.calls}.",
                    [
                        AgentToolCall(
                            f"call_ground_{self.calls}",
                            "skill.visual_grounding.locate",
                            {
                                "observation_id": observation_id,
                                "query": queries[self.calls],
                                "include_distance": False,
                            },
                        )
                    ],
                )
            return AssistantTurn(
                "assistant_complete",
                "Use the existing qualitative evidence and finish.",
                [
                    AgentToolCall(
                        "call_commit",
                        "runtime.commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "supported",
                                        "evidence": "The qualitative result is sufficient for this test.",
                                    }
                                ]
                            },
                        },
                    )
                ],
            )

    handler_calls: list[dict] = []

    def fake_handler(payload):
        handler_calls.append(payload)
        return SkillCapabilityResult(
            message="no candidate",
            data={
                "observation_id": payload["args"]["observation_id"],
                "query": payload["args"]["query"],
                "include_distance": False,
                "candidates": [],
            },
        )

    monkeypatch.setattr(
        skill_dispatcher_module,
        "_load_handler_module",
        lambda handler_path: type(
            "HandlerModule", (), {"handle": staticmethod(fake_handler)}
        )(),
    )
    plugin_registry = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.visual_grounding"],
            }
        }
    )
    run_dir = tmp_path / "read_only_quota"
    recorder = AsyncRunLogger(run_dir)
    planner = QuotaPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        model_config=ModelConfig(
            task_parser_mode="model",
            max_tool_rounds=10,
            max_consecutive_protocol_rejections=2,
        ),
        shutdown_policy="none",
        plugin_registry=plugin_registry,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()
    records = [
        json.loads(line)
        for line in (run_dir / "memory" / "agent_transcript.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rejected = [
        item["payload"]
        for item in records
        if item["kind"] == "tool_result"
        and item["payload"].get("tool_call_id") in {
            "call_ground_2",
            "call_ground_4",
        }
    ]

    assert state.completed is True
    assert planner.calls == 5
    assert len(handler_calls) == 2
    assert [item["status"] for item in rejected] == ["rejected", "rejected"]
    assert "identical read-only call" in rejected[0]["message"]
    assert "reached its limit of 2" in rejected[1]["message"]


def test_phase_transition_clears_before_action_anchor_for_new_phase(tmp_path):
    class FirstPhaseOnlyActionPlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.phase_calls: dict[str, int] = {}

        def plan_next_with_phase_decision(self, state, context=None):
            calls = self.phase_calls.get(state.current_phase_key, 0)
            self.phase_calls[state.current_phase_key] = calls + 1
            if state.current_phase_index == 0 and calls == 0:
                return CoupledActionPhasePlan(
                    phase=state.current_phase_name,
                    phase_decision=PhaseTransition.continue_current_phase(),
                    steps=[PlannedStep.tool_step("uav.hover", {}, "observe once")],
                )
            decision = (
                ADVANCE_TO_NEXT_PHASE
                if state.current_phase_index == 0
                else COMPLETE_TASK
            )
            return CoupledActionPhasePlan(
                phase=state.current_phase_name,
                phase_decision=PhaseTransition(decision, "phase evidence accepted"),
                steps=[],
            )

    run_dir = tmp_path / "phase_anchor_reset"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=FirstPhaseOnlyActionPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_two_phase_parse_result())
    recorder.close()

    memory = json.loads(
        (run_dir / "state" / "flight_memory.json").read_text(encoding="utf-8")
    )
    observations = [
        json.loads(line)
        for line in (run_dir / "memory" / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert state.completed
    assert "before_last_action_window" not in memory["observation_anchors"]
    assert memory["observation_anchors"]["phase_start"]["id"] == observations[-1]["id"]


def test_native_runtime_audits_provider_rejection_and_executes_null_content_turns(
    tmp_path,
):
    """A missing explanation cannot block otherwise valid physical ToolCalls."""

    def response(*calls):
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "reasoning_content": "must never enter the transcript",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": wire_name,
                                    "arguments": json.dumps(args),
                                },
                            }
                            for call_id, wire_name, args in calls
                        ],
                    },
                }
            ]
        }

    class ProviderSequencePlanner(ModelActionPlanner):
        def __init__(self) -> None:
            super().__init__(
                ModelConfig(
                    model_name="stub",
                    api_key="test-key",
                    base_url="https://example.test/v1",
                )
            )
            self.responses = [
                response(("bad_call", "unknown_tool", {"value": 1})),
                response(
                    (
                        "commit_move",
                        "runtime_commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "unknown",
                                        "evidence": "No action has executed yet.",
                                    }
                                ]
                            },
                            "phase_strategy": {
                                "stages": ["hover once"],
                                "active_stage": 0,
                                "reason": "A bounded action is still required.",
                            },
                            "action_window": {
                                "intent": "hover once before reassessing",
                                "expected_effect": "the UAV remains stable",
                            },
                        },
                    ),
                    ("hover_once", "uav_hover", {}),
                ),
                response(
                    (
                        "commit_done",
                        "runtime_commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "supported",
                                        "evidence": "The post-action observation is available.",
                                    }
                                ]
                            },
                        },
                    )
                ),
            ]
            self.payloads = []

        def _request(self, payload):
            self.payloads.append(payload)
            return self.responses.pop(0)

    run_dir = tmp_path / "native_null_content"
    recorder = AsyncRunLogger(run_dir)
    planner = ProviderSequencePlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
        model_config=planner.model_config,
        enable_action_planner_debug=True,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    transcript = [
        json.loads(line)
        for line in (run_dir / "memory" / "agent_transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    kinds = [item["kind"] for item in transcript]
    rejected_index = kinds.index("provider_turn_rejected")
    assert state.completed
    assert kinds[rejected_index + 1] == "protocol_feedback"
    assert transcript[rejected_index]["payload"]["tool_calls"][0]["wire_name"] == (
        "unknown_tool"
    )
    assert "must never enter" not in json.dumps(transcript)
    accepted = [item for item in transcript if item["kind"] == "assistant_turn"]
    assert [item["payload"]["public_intent_source"] for item in accepted] == [
        "tool_arguments_fallback",
        "tool_arguments_fallback",
    ]
    assert accepted[0]["payload"]["public_intent"] == (
        "hover once before reassessing"
    )
    actions = [
        json.loads(line)
        for line in (run_dir / "memory" / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    hover_result = next(
        item
        for item in transcript
        if item["kind"] == "tool_result"
        and item["payload"].get("tool_call_id") == "hover_once"
    )
    assert hover_result["payload"]["status"] == "ok"
    action_index = int(hover_result["payload"]["result_ref"].rsplit("#", 1)[1]) - 1
    assert actions[action_index]["tool"] == "uav.hover"
    replayed_assistant = [
        item
        for item in planner.payloads[-1]["messages"]
        if item.get("role") == "assistant"
    ]
    assert replayed_assistant[-1]["content"] is None
    decision_files = sorted((run_dir / "debug" / "action_planner").glob("*.json"))
    decisions = [
        json.loads(path.read_text(encoding="utf-8")) for path in decision_files
    ]
    assert [item["decision_id"] for item in decisions] == [
        "decision_0001",
        "decision_0002",
    ]
    assert [len(item["model_rounds"]) for item in decisions] == [2, 1]
    assert decisions[0]["physical_window_index"] == 1
    assert decisions[1]["physical_window_index"] is None
    assert decisions[1]["completed_window_count"] == 1
    assert [
        item["runtime_outcome"]["status"]
        for item in decisions[0]["model_rounds"]
    ] == ["protocol_rejected", "commit_accepted"]
    accepted_results = decisions[0]["model_rounds"][-1]["runtime_outcome"][
        "tool_results"
    ]
    assert {item["tool_call_id"] for item in accepted_results} == {
        "commit_move",
        "hover_once",
    }
    assert decisions[1]["model_rounds"][0]["runtime_outcome"]["tool_results"][
        0
    ]["tool_call_id"] == "commit_done"
    debug_text = json.dumps(decisions)
    assert "must never enter" not in debug_text
    assert "data:image" not in debug_text
    phase_window = next(
        record["payload"]
        for record in load_event_records(run_dir)
        if record["event"] == "phase_window_recorded"
    )
    assert phase_window["decision_id"] == "decision_0001"


def test_native_decision_ids_keep_two_physical_windows_and_terminal_separate(
    tmp_path,
):
    class TwoWindowsThenComplete:
        planner_mode = "model"

        def __init__(self) -> None:
            self.calls = 0

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            self.calls += 1
            supported = self.calls == 3
            commit = AgentToolCall(
                f"commit_{self.calls}",
                "runtime.commit_phase_turn",
                {
                    "phase": "final_phase",
                    "completion_assessment": {
                        "done_when": [
                            {
                                "index": 0,
                                "status": "supported" if supported else "unknown",
                                "evidence": "complete" if supported else "observe again",
                            }
                        ]
                    },
                    **(
                        {}
                        if supported
                        else {
                            "phase_strategy": {
                                "stages": ["hover and observe"],
                                "active_stage": 0,
                                "reason": "another checkpoint is needed",
                            },
                            "action_window": {
                                "intent": "hover once",
                                "expected_effect": "a new checkpoint is captured",
                            },
                        }
                    ),
                },
            )
            calls = [commit]
            if not supported:
                calls.append(AgentToolCall(f"hover_{self.calls}", "uav.hover", {}))
            return AssistantTurn(
                f"assistant_{self.calls}",
                "complete" if supported else "hover once",
                calls,
            )

    run_dir = tmp_path / "three_decisions"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=TwoWindowsThenComplete(),
        shutdown_policy="none",
        enable_action_planner_debug=True,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result())
    recorder.close()

    decisions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "debug" / "action_planner").glob("*.json"))
    ]
    assert state.completed
    assert len(decisions) == 3
    assert [item["decision_id"] for item in decisions] == [
        "decision_0001",
        "decision_0002",
        "decision_0003",
    ]
    assert [item["physical_window_index"] for item in decisions] == [1, 2, None]
    assert [item["completed_window_count"] for item in decisions] == [0, 1, 2]
    assert decisions[-1]["coupled_phase_decision"]["decision"] == COMPLETE_TASK


def test_task_runtime_normalizes_yaw_wraparound_delta(tmp_path):
    recorder = AsyncRunLogger(tmp_path / "yaw_wrap")
    server = SimulationServer([make_package()], run_dir=tmp_path / "yaw_wrap")
    runtime = TaskRuntime(SimulatorEnvClient(server), recorder, TaskParser(mode="rule_based"), planner=HoverOnlyPlanner())

    measurements = pose_measurements(
        {"euler_rad": {"yaw": 159.0 * 3.141592653589793 / 180.0}},
        {"euler_rad": {"yaw": -172.0 * 3.141592653589793 / 180.0}},
    )
    recorder.close()

    assert measurements["yaw_delta_deg"] == 29.0


def test_task_runtime_recharges_yaw_scan_only_after_verified_translation(
    tmp_path,
):
    run_dir = tmp_path / "yaw_scan_recharge"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        safety=SafetySupervisor(
            UavSafetyConfig(yaw_scan_recharge_distance_m=0.5)
        ),
    )
    parsed = make_single_phase_parse_result()
    state = MissionState("recharge", parsed.instruction, parsed.plan)
    state.record_phase_window_started()
    state.record_action(
        "uav.turn_relative", {"yaw_deg": -120.0}, {"ok": True}
    )

    below_threshold = runtime._maybe_recharge_visual_search_budget(
        state,
        {
            "planar_displacement_m": 0.49,
            "vertical_delta_m": 0.0,
        },
        phase=state.current_phase_name,
    )
    assert below_threshold is None
    assert state.yaw_scan_extent_deg == 120.0
    assert state.yaw_scan_turn_count == 1

    vertical_recharge = runtime._maybe_recharge_visual_search_budget(
        state,
        {
            "planar_displacement_m": 0.0,
            "vertical_delta_m": -0.5,
        },
        phase=state.current_phase_name,
    )
    assert vertical_recharge is not None
    assert vertical_recharge["translation_3d_m"] == 0.5
    assert vertical_recharge["previous_segment"]["extent_deg"] == 120.0
    assert state.yaw_scan_extent_deg == 0.0
    assert state.yaw_scan_turn_count == 0
    assert state.yaw_scan_recharge_count == 1

    state.record_action(
        "uav.turn_relative", {"yaw_deg": 60.0}, {"ok": True}
    )
    planar_recharge = runtime._maybe_recharge_visual_search_budget(
        state,
        {
            "planar_displacement_m": 0.5,
            "vertical_delta_m": 0.0,
        },
        phase=state.current_phase_name,
    )
    assert planar_recharge is not None
    assert planar_recharge["recharge_count"] == 2
    assert state.yaw_scan_offset_deg == 0.0
    recorder.close()

    recharge_events = [
        record
        for record in load_event_records(run_dir)
        if record["event"] == "visual_search_budget_recharged"
    ]
    assert len(recharge_events) == 2
    assert recharge_events[0]["payload"]["vertical_displacement_m"] == 0.5


def test_task_runtime_continue_transition_window_limit_stops_without_advancing_phase(tmp_path):
    class AlwaysContinuePlanner:
        planner_mode = "test"

        def plan_next_with_phase_decision(self, state, context=None):
            plan = ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.hover", {}, "keep checking current phase")],
            )
            return coupled_continue_from_short_plan(plan, "not enough evidence")

    run_dir = tmp_path / "continue_loop_guard"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=AlwaysContinuePlanner(),
        shutdown_policy="none",
        runtime_loop_guard_windows=2,
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result("multi motion final phase", "final_phase"))
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    assert not state.completed
    assert state.stopped
    assert state.current_phase_index == 0
    assert state.stopped_reason == "runtime_loop_guard_reached"
    assert "runtime_loop_guard_reached" in event_names
    assert "phase_transition_window_limit_reached" in event_names
    mission_state = json.loads((run_dir / "state/latest_state.json").read_text(encoding="utf-8"))
    assert mission_state["max_phase_window_count"] == 2
    assert mission_state["stopped"] is True
    assert mission_state["stopped_reason"] == "runtime_loop_guard_reached"
    assert "phase_window_counts" not in mission_state
    assert "current_phase_window_count" in mission_state


def test_native_runtime_loop_guard_closes_last_physical_batch_with_checkpoint(tmp_path):
    class NativeAlwaysContinuePlanner:
        planner_mode = "model"

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            observation_id = transcript.latest_observation_id()
            return AssistantTurn(
                "assistant_continue",
                "Turn once, then check the resulting observation.",
                [
                    AgentToolCall(
                        "call_commit",
                        "runtime.commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "unknown",
                                        "evidence": (
                                            "The requested movement has not yet been "
                                            f"confirmed in observation {observation_id}."
                                        ),
                                    }
                                ]
                            },
                            "phase_strategy": {
                                "stages": ["turn", "verify"],
                                "active_stage": 0,
                                "reason": "A post-action observation is required.",
                            },
                            "action_window": {
                                "intent": "turn once",
                                "expected_effect": "the camera bearing changes",
                            },
                        },
                    ),
                    AgentToolCall(
                        "call_turn",
                        "uav.turn_relative",
                        {"yaw_deg": 5.0},
                    ),
                ],
            )

    run_dir = tmp_path / "native_loop_guard_checkpoint"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=NativeAlwaysContinuePlanner(),
        shutdown_policy="none",
        runtime_loop_guard_windows=1,
    )

    state = runtime.run_parsed_task(
        "m1",
        make_single_phase_parse_result("turn once", "final_phase"),
    )
    recorder.close()

    transcript_records = [
        json.loads(line)
        for line in (run_dir / "memory" / "agent_transcript.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    checkpoints = [
        record
        for record in transcript_records
        if record["kind"] == "state_checkpoint"
    ]
    assert state.stopped_reason == "runtime_loop_guard_reached"
    assert len(checkpoints) == 2
    assert (
        checkpoints[0]["payload"]["observation_id"]
        != checkpoints[1]["payload"]["observation_id"]
    )
    assert transcript_records[-1]["kind"] == "state_checkpoint"


def test_native_runtime_report_blocked_stops_current_instruction_and_closes_calls(tmp_path):
    class NativeBlockedPlanner:
        planner_mode = "model"

        def plan_turn(self, transcript, registry, *, allowed_tool_names=None):
            observation_id = transcript.latest_observation_id()
            assert observation_id
            assert allowed_tool_names is not None
            assert "runtime.report_blocked" in allowed_tool_names
            return AssistantTurn(
                "assistant_blocked",
                "The available evidence cannot support another safe action.",
                [
                    AgentToolCall(
                        "call_commit_blocked",
                        "runtime.commit_phase_turn",
                        {
                            "phase": "final_phase",
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "unknown",
                                        "evidence": f"Observation {observation_id} is insufficient.",
                                    }
                                ]
                            },
                        },
                    ),
                    AgentToolCall(
                        "call_report_blocked",
                        "runtime.report_blocked",
                        {
                            "blocking_condition": "insufficient_evidence",
                            "reason": "No safe evidence-producing maneuver remains.",
                            "evidence_refs": [observation_id],
                            "suggested_user_action": "Provide a new landmark.",
                        },
                    ),
                ],
            )

    run_dir = tmp_path / "native_report_blocked"
    recorder = AsyncRunLogger(run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(SimulationServer([make_package()], run_dir=run_dir)),
        recorder,
        TaskParser(mode="rule_based"),
        planner=NativeBlockedPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task(
        "m1",
        make_single_phase_parse_result("inspect safely", "final_phase"),
    )
    recorder.close()

    assert state.completed is False
    assert state.stopped is True
    assert state.stopped_reason == "agent_reported_blocked:insufficient_evidence"
    assert state.stop_source == "runtime.report_blocked"
    transcript_records = [
        json.loads(line)
        for line in (run_dir / "memory" / "agent_transcript.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assistant = next(
        item for item in transcript_records if item["kind"] == "assistant_turn"
    )
    assert [
        call["tool_name"] for call in assistant["payload"]["tool_calls"]
    ] == ["runtime.commit_phase_turn", "runtime.report_blocked"]
    results = {
        item["payload"]["tool_call_id"]: item["payload"]
        for item in transcript_records
        if item["kind"] == "tool_result"
    }
    assert results["call_report_blocked"]["status"] == "ok"
    assert results["call_commit_blocked"]["status"] == "ok"
    assert transcript_records[-1]["kind"] == "task_terminal"
    assert transcript_records[-1]["payload"]["completed"] is False
    checkpoints = [
        item for item in transcript_records if item["kind"] == "state_checkpoint"
    ]
    assert len(checkpoints) == 2
    assert checkpoints[0]["payload"]["observation_id"] != checkpoints[1]["payload"]["observation_id"]
    assert not any(
        call["tool_name"].startswith("uav.")
        for call in assistant["payload"]["tool_calls"]
    )


def test_task_runtime_stabilizes_after_each_motion_step_before_observation(tmp_path):
    class MultiMotionPlanner:
        planner_mode = "test"

        def __init__(self) -> None:
            self.calls = 0

        def plan_next_with_phase_decision(self, state, context=None):
            self.calls += 1
            if self.calls > 1:
                return CoupledActionPhasePlan(
                    state.current_phase_name,
                    PhaseTransition(COMPLETE_TASK, "stabilization verified"),
                    [],
                )
            plan = ShortHorizonPlan(
                state.current_phase_name,
                [
                    PlannedStep.tool_step("uav.move_body", {"direction": "forward", "duration": 1.0}, "move"),
                    PlannedStep.tool_step("uav.turn_relative", {"yaw_deg": 10.0}, "turn"),
                ],
            )
            return coupled_continue_from_short_plan(plan, "execute both motion steps")

    run_dir = tmp_path / "stabilizes_between_steps"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=MultiMotionPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result("multi motion final phase", "final_phase"))
    recorder.close()

    events = load_event_records(run_dir)
    event_names = [record["event"] for record in events]
    phase_summary = next(record["payload"] for record in events if record["event"] == "phase_window_recorded")
    phase_record = json.loads((run_dir / phase_summary["debug_ref"]).read_text(encoding="utf-8"))
    stabilization_events = [
        index for index, name in enumerate(event_names) if name in {"uav_stabilization_completed", "uav_stabilization_skipped", "uav_stabilization_timeout"}
    ]
    automatic_observation_index = event_names.index("automatic_observation_policy_applied")

    assert state.completed
    assert "preflight_stabilization_completed" in event_names
    assert phase_summary["action_window"]["scope"] == "current_phase_only"
    assert phase_record["action_window"]["expected_effect"]
    assert len(phase_record["executed_steps"]) == 2
    assert all("stabilization" in step for step in phase_record["executed_steps"])
    assert phase_record["executed_steps"][0]["stabilization"]["tool"] == "uav.move_body"
    assert phase_record["executed_steps"][1]["stabilization"]["tool"] == "uav.turn_relative"
    assert max(stabilization_events) < automatic_observation_index


def test_task_runtime_applies_initial_pose_before_first_configured_observation(tmp_path):
    run_dir = tmp_path / "task_execution_options"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )
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
    latest_context = json.loads(
        (run_dir / "contexts/latest_context.json").read_text(encoding="utf-8")
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
    assert latest_context["observation_policy"]["modalities"] == ["rgb", "pose"]
    assert "uav.set_pose" not in actions_text
    assert flight_memory["reset_epoch"] == 1


def test_task_runtime_reapplies_initial_pose_when_scene_is_reused(tmp_path):
    run_dir = tmp_path / "reused_episode_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

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
    planner = HoverOnlyPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    with pytest.raises(RuntimeError, match="does not support.*mask"):
        runtime.run_task(
            "m1",
            LAMP_TASK,
            execution_options=TaskExecutionOptions(
                observation_modalities=("mask", "pose")
            ),
        )
    recorder.close()

    assert planner._phase_calls == {}
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


class _NoHoverBackend(AirSimBackend):
    def describe_capabilities(self, package):
        capabilities = super().describe_capabilities(package)
        return SceneCapabilities(
            protocol=capabilities.protocol,
            control_mode=capabilities.control_mode,
            supported_actions=tuple(
                action
                for action in capabilities.supported_actions
                if action != "hover"
            ),
            observation_modalities=capabilities.observation_modalities,
            requires_takeoff=capabilities.requires_takeoff,
            supports_soft_reset=capabilities.supports_soft_reset,
        )


def test_task_runtime_rejects_custom_planner_unsupported_action(tmp_path):
    run_dir = tmp_path / "unsupported_custom_planner_action"
    recorder = AsyncRunLogger(run_dir)
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: _NoHoverBackend(
            use_real_uav=False,
            action_config=context.action_config,
        ),
    )
    server = SimulationServer(
        [make_package()],
        run_dir=run_dir,
        backend_registry=registry,
    )
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=HoverOnlyPlanner(),
        shutdown_policy="none",
    )

    with pytest.raises(RuntimeError, match="does not support hover"):
        runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    debug_events = [
        json.loads(line)
        for line in (run_dir / "debug/debug_events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rejected = [
        event
        for event in debug_events
        if event["event"] == "runtime_action_capability_rejected"
    ]
    assert rejected
    assert rejected[-1]["payload"]["tool"] == "uav.hover"
    action_records = [
        json.loads(line)
        for line in (run_dir / "memory/actions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(record["tool"] != "uav.hover" for record in action_records)


def test_task_runtime_rejects_initial_pose_without_backend_capability(tmp_path):
    run_dir = tmp_path / "unsupported_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: _NoSetPoseBackend(
            use_real_uav=False,
            action_config=context.action_config,
        ),
    )
    server = SimulationServer(
        [make_package()],
        run_dir=run_dir,
        backend_registry=registry,
    )
    planner = HoverOnlyPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

    with pytest.raises(RuntimeError, match="does not support set_pose"):
        runtime.run_task(
            "m1",
            LAMP_TASK,
            execution_options=TaskExecutionOptions(
                initial_pose=InitialPose(1.0, 2.0, 3.0)
            ),
        )
    recorder.close()

    assert planner._phase_calls == {}
    assert "episode_initial_pose_started" not in [
        record["event"] for record in load_event_records(run_dir)
    ]


class _FailingSetPoseAdapter(NullUavPrimitiveAdapter):
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


def test_task_runtime_reports_initial_pose_action_failure_before_planning(tmp_path):
    run_dir = tmp_path / "failed_initial_pose"
    recorder = AsyncRunLogger(run_dir)
    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: AirSimBackend(
            uav=_FailingSetPoseAdapter(),
            use_real_uav=False,
            action_config=context.action_config,
        ),
    )
    server = SimulationServer(
        [make_package()],
        run_dir=run_dir,
        backend_registry=registry,
    )
    planner = HoverOnlyPlanner()
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=planner,
        shutdown_policy="none",
    )

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
    assert planner._phase_calls == {}
    assert "episode_initial_pose_failed" in event_names
    assert "safety_hover_completed" in event_names
