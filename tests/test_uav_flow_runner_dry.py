"""Dry-run tests for UAV-Flow EnvClient and default image context."""

from __future__ import annotations

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from src.agent.phase import COMPLETE_TASK, PhaseTransition
from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.perception.raw_observation import RawObservation


def make_case() -> UavFlowCase:
    return UavFlowCase(
        case_id="dry",
        file_name="dry.json",
        source_path="dry.json",
        class_name="Turn",
        instruction="Turn to the person",
        instruction_unified="Turn to the direction of the person",
        selected_instruction="Turn to the direction of the person",
        initial_pos=[0, 0, 100, 0, 0, 0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0]],
    )


def test_dry_env_client_generates_real_rgb_artifact_for_default_image_module(tmp_path):
    case = make_case()
    client = UavFlowBenchmarkEnvClient(case, run_dir=tmp_path, env_id="dry", real_unrealcv=False)
    scene = client.start_scene("uav_flow_downtownwest", case.case_id)
    observation = client.get_observation(scene.scene_instance_id, "uav_flow_drone", ["rgb", "pose"])
    raw = RawObservation.from_env_observation(observation, source="uav_flow_benchmark").to_dict()

    writer = MemoryEventWriter(tmp_path)
    write_result = writer.write_observation(raw)
    reader = MemoryRecordReader(tmp_path)
    plan = semantic_plan_from_dict(
        {
            "intent": "benchmark",
            "goal": {
                "target": {"description": "person", "attributes": {}},
                "final_state": "facing_target",
            },
            "phases": [{"name": "turn_to_target", "meaning": "face the visible person"}],
            "uncertainties": [],
        }
    )
    state = MissionState(run_id=tmp_path.name, user_instruction=case.selected_instruction, semantic_plan=plan)
    state.record_observation(raw)
    memory = FlightMemory(
        run_id=tmp_path.name,
        map_id="uav_flow_downtownwest",
        scene_instance_id=case.case_id,
        agent_id="uav_flow_drone",
        active_instruction=case.selected_instruction,
        memory_reader=reader,
    )
    memory.append_observation_ref(write_result["observation_ref"])
    memory.append_pose_ref(write_result["pose_ref"])
    memory.append_frame_refs(write_result["frame_refs"])

    context = ObservationContextBuilder().build(state, memory)
    image_fragment = next(fragment for fragment in context["fragments"] if fragment["name"] == "image")

    assert image_fragment["model_inputs"][0]["type"] == "image"
    assert image_fragment["model_inputs"][0]["encoding"] == "data_url"
    assert image_fragment["model_inputs"][0]["data"].startswith("data:image/png;base64,")
    assert "target_pos" not in str(context)
    assert "reference_path_preprocessed" not in str(context)


class FakeTaskParser:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def parse(self, instruction: str) -> TaskParseResult:
        plan = semantic_plan_from_dict(
            {
                "intent": "benchmark",
                "goal": {
                    "target": {"description": "person", "attributes": {}},
                    "final_state": "facing_target",
                },
                "phases": [{"name": "benchmark_phase", "meaning": "complete benchmark smoke"}],
                "uncertainties": [],
            }
        )
        return TaskParseResult(instruction=instruction, parser_mode="fake", plan=plan)


class FakePlanner:
    planner_mode = "fake"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def plan_next(self, state, context) -> ShortHorizonPlan:
        return ShortHorizonPlan(
            phase="benchmark_phase",
            steps=[PlannedStep.tool_step("uav.hover", {})],
            phase_transition=PhaseTransition(COMPLETE_TASK, "dry runner complete"),
        )


def test_runner_dry_writes_review_leak_check_and_rgb_summary(tmp_path, monkeypatch):
    import evaluation.benchmarks.uav_flow.runner as runner_module

    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    config = UavFlowEvalConfig.from_dict(
        {
            "output_dir": str(tmp_path / "eval"),
            "run_root": str(tmp_path / "runs"),
            "real_unrealcv": False,
            "parser_mode": "fake",
            "safety_overrides": {"max_yaw_step_deg": 10.0},
            "runtime_loop_guard_windows": 7,
        }
    )
    runner = UavFlowBenchmarkRunner(config)
    case = make_case()

    result = runner.run_case(case, tmp_path / "eval" / "manual")

    assert result["status"] == "ok"
    assert result["evaluation_truth_leak_check"]["status"] == "passed"
    assert result["run_artifacts"]["first_rgb"]
    assert result["visual_diagnostics"]["truth_leak_check"]["status"] == "passed"
    assert result["review_path"].endswith("review.md")
    assert runner.safety_config.max_yaw_step_deg == 10.0
    assert runner.config.runtime_loop_guard_windows == 7
    assert result["completed"] is True
    assert result["stopped"] is False
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "review.md").exists()
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "plot_result.json").exists()
    assert (tmp_path / "eval" / "manual" / "cases" / case.case_id / "visual_diagnostics.json").exists()
