"""Test M4 task runtime dry-run behavior."""

from __future__ import annotations

import base64
import inspect
import json

import pytest

from src.agent.mission_state import MissionState
from src.agent.messages import AgentMessage
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.model_repair_planner import ModelRepairPlannerError
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan, PlannedStep, ShortHorizonPlan
from src.agent.safety import SafetyDecision, SafetySupervisor
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
                    metrics=[],
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
                    metrics=[],
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
                    metrics=[],
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
                    metrics=[],
                ),
                SemanticPhase(
                    index=2,
                    total=2,
                    name="phase_two",
                    goal="finish phase two",
                    done_when=["phase two is complete"],
                    continue_when=["phase two is not complete"],
                    metrics=[],
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


class CoupledPlannerWithRepairEvent(CoupledPlannerStub):
    def __init__(self) -> None:
        super().__init__([CONTINUE_CURRENT_PHASE, COMPLETE_TASK])
        self._repair_event_available = True
        self.last_repair_attempts = [
            {
                "attempt": 1,
                "status": "repaired",
                "repair_context": {
                    "phase": "final_phase",
                    "violation": {"code": "action_window_intermediate_observation_requires_single_step"},
                },
                "repaired_plan": {
                    "phase": "final_phase",
                    "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "repaired"}],
                },
            }
        ]

    def plan_next_with_phase_decision(self, state, context=None):
        result = super().plan_next_with_phase_decision(state, context)
        if self._repair_event_available:
            self._repair_event_available = False
        else:
            self.last_repair_attempts = []
        return result


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
    assert selected_roles == ["phase_start_reference", "current_observation"]
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
            "done_when, continue_when, or metrics still fit observed progress and user intent. Name the doubtful "
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
            "done_when, continue_when, or metrics still fit observed progress and user intent. Name the doubtful "
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
            "goal/done_when/continue_when/metrics is now low"
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
    assert "model_phase_transition_judged" not in event_names


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


def test_task_runtime_coupled_records_contract_repair_events(tmp_path):
    run_dir = tmp_path / "coupled_repair_event_run"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    planner = CoupledPlannerWithRepairEvent()
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
    assert "model_action_contract_repair_requested" in event_names
    assert "model_action_contract_repaired" in event_names
    assert "model_phase_transition_judged" not in event_names


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
    assert action_constraints["planner_tools"] == ["uav.hover", "uav.turn_relative", "uav.move_body"]
    assert "observe_step_schema" not in action_constraints
    assert planner.contexts[0]["observation_policy"]["mode"] == "automatic"
    assert set(action_constraints["action_schemas"]) == {"uav.hover", "uav.turn_relative", "uav.move_body"}
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


def test_task_runtime_action_planner_debug_writes_redacted_context(tmp_path):
    class OneTurnPlanner:
        planner_mode = "model"
        last_model_request_recovery = {
            "status": "recovered",
            "attempt_count": 3,
            "degraded_to_now_only": True,
            "attempts": [
                {"attempt": 1, "status": "failed", "image_mode": "all", "image_count": 2},
                {"attempt": 2, "status": "failed", "image_mode": "all", "image_count": 2},
                {
                    "attempt": 3,
                    "status": "ok",
                    "image_mode": "current_only",
                    "image_count": 1,
                },
            ],
        }

        def plan_next_with_phase_decision(self, state, context=None):
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
    assert payload["model_request_recovery"]["status"] == "recovered"
    assert payload["model_request_recovery"]["degraded_to_now_only"] is True
    assert "data:image" not in debug_files[0].read_text(encoding="utf-8")
    events = load_event_records(run_dir)
    recovered = [
        record
        for record in events
        if record["event"] == "model_action_request_recovered"
    ]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["attempt_count"] == 3
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
    assert transition_payload["phase_completion_evaluation"]["metrics_count"] == 0
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

        def _call_coupled_action_phase_openai_compatible(self, context: dict):
            self.contexts.append(context)
            return {
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
                "agent_messages": [
                    {
                        "type": "progress",
                        "content": "The context verification phase is complete.",
                        "evidence_refs": [],
                        "confidence": "medium",
                        "requires_user_reply": False,
                    }
                ],
                "steps": [],
            }

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
    assert 1 <= len(motion_history["recent_execution"]) <= 5
    assert motion_history["recent_execution_order"] == "oldest_to_newest"
    assert all("ok" not in item for item in motion_history["recent_execution"])
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


def test_task_runtime_logs_contract_repair_success_and_continues_to_safety(tmp_path):
    class RepairSuccessPlanner:
        planner_mode = "model"

        def __init__(self) -> None:
            self.last_repair_attempts = []
            self.calls = 0

        def plan_next_with_phase_decision(self, state, context=None):
            self.calls += 1
            if self.calls > 1:
                self.last_repair_attempts = []
                return CoupledActionPhasePlan(
                    state.current_phase_name,
                    PhaseTransition(COMPLETE_TASK, "repair path verified"),
                    [],
                )
            self.last_repair_attempts = [
                {
                    "attempt": 1,
                    "status": "failed",
                    "repair_context": {"attempt": 1, "violation": {"code": "forbidden_tool"}},
                    "repaired_plan": {"phase": state.current_phase_name, "steps": [{"kind": "observe"}]},
                    "violation": {"code": "observe_managed_by_runtime"},
                },
                {
                    "attempt": 2,
                    "status": "repaired",
                    "repair_context": {"attempt": 2, "violation": {"code": "observe_managed_by_runtime"}},
                    "repaired_plan": {
                        "phase": state.current_phase_name,
                        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}}],
                    },
                },
            ]
            plan = ShortHorizonPlan(
                state.current_phase_name,
                [PlannedStep.tool_step("uav.hover", {}, "repaired safe hover")],
            )
            return coupled_continue_from_short_plan(plan, "execute repaired action once")

    run_dir = tmp_path / "repair_success"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=RepairSuccessPlanner(),
        shutdown_policy="none",
    )

    state = runtime.run_parsed_task("m1", make_single_phase_parse_result("repair succeeds", "final_phase"))
    recorder.close()

    event_names = [record["event"] for record in load_event_records(run_dir)]
    assert state.completed
    assert event_names.count("model_action_contract_repair_requested") == 2
    assert "model_action_contract_repair_failed" in event_names
    assert "model_action_contract_repaired" in event_names
    assert "short_horizon_planned" in event_names
    assert "automatic_observation_updated" in event_names


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


def test_task_runtime_logs_contract_repair_failure_then_best_effort_hover(tmp_path):
    class RepairFailPlanner:
        planner_mode = "model"

        def plan_next_with_phase_decision(self, state, context=None):
            attempts = [
                {
                    "attempt": 1,
                    "status": "failed",
                    "repair_context": {"attempt": 1, "violation": {"code": "forbidden_tool"}},
                    "violation": {"code": "observe_managed_by_runtime"},
                }
            ]
            raise ModelRepairPlannerError(
                "repair failed",
                initial_plan={"phase": state.current_phase_name, "steps": []},
                final_plan={"phase": state.current_phase_name, "steps": [{"kind": "observe"}]},
                repair_attempts=attempts,
                violations=[{"code": "forbidden_tool"}, {"code": "observe_managed_by_runtime"}],
                max_repair_attempts=3,
                reason="max_repair_attempts_exhausted",
            )

    run_dir = tmp_path / "repair_failure"
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=RepairFailPlanner(),
        shutdown_policy="none",
    )

    with pytest.raises(ModelRepairPlannerError):
        runtime.run_task("m1", LAMP_TASK)
    recorder.close()

    event_names = [record["event"] for record in load_event_records(run_dir)]
    assert "model_action_contract_repair_requested" in event_names
    assert "model_action_contract_repair_failed" in event_names
    assert "task_failed" in event_names
    assert "safety_hover_completed" in event_names


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
