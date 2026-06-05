"""Dry-run tests for UAV-Flow EnvClient and default image context."""

from __future__ import annotations

from dataclasses import replace

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


def test_config_roundtrip_includes_reuse_real_unrealcv_env():
    config = UavFlowEvalConfig.from_dict({"real_unrealcv": True, "reuse_real_unrealcv_env": False})

    assert config.real_unrealcv is True
    assert config.reuse_real_unrealcv_env is False
    assert UavFlowEvalConfig.from_dict({}).reuse_real_unrealcv_env is True
    assert config.to_dict()["reuse_real_unrealcv_env"] is False


def test_runner_reuses_single_real_env_client_across_cases(tmp_path, monkeypatch):
    import evaluation.benchmarks.uav_flow.runner as runner_module

    CountingReusableClient.instances = []
    monkeypatch.setattr(runner_module, "TaskParser", FakeTaskParser)
    monkeypatch.setattr(runner_module, "ModelActionPlanner", FakePlanner)
    monkeypatch.setattr(runner_module, "UavFlowBenchmarkEnvClient", CountingReusableClient)
    cases = [make_case(), replace(make_case(), case_id="dry_2", file_name="dry_2.json")]
    monkeypatch.setattr(UavFlowBenchmarkRunner, "load_cases", lambda self: cases)
    config = UavFlowEvalConfig.from_dict(
        {
            "output_dir": str(tmp_path / "eval"),
            "run_root": str(tmp_path / "runs"),
            "real_unrealcv": True,
            "reuse_real_unrealcv_env": True,
            "parser_mode": "fake",
            "enable_sr_judge": False,
            "enable_visual_diagnostics": False,
        }
    )

    result = UavFlowBenchmarkRunner(config).run(eval_id="reuse")

    assert len(result["results"]) == 2
    assert len(CountingReusableClient.instances) == 1
    client = CountingReusableClient.instances[0]
    assert client.start_real_count == 1
    assert client.prepare_count == 1
    assert client.stop_count == 0
    assert client.close_count == 1
    assert (tmp_path / "eval" / "reuse" / "cases" / "dry" / "review.md").exists()
    assert (tmp_path / "eval" / "reuse" / "cases" / "dry_2" / "review.md").exists()
    assert result["summary_paths"]["html_review"].endswith("review.html")
    assert (tmp_path / "eval" / "reuse" / "review.html").exists()
    assert (tmp_path / "eval" / "reuse" / "cases" / "dry" / "review.html").exists()


class CountingReusableClient(UavFlowBenchmarkEnvClient):
    instances: list["CountingReusableClient"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_real_count = 0
        self.prepare_count = 0
        self.stop_count = 0
        self.close_count = 0
        CountingReusableClient.instances.append(self)

    def _start_real_env(self) -> None:
        self.start_real_count += 1
        self.env = object()

    def prepare_case(self, case: UavFlowCase, run_dir, scene_id: str | None = None):
        self.prepare_count += 1
        return super().prepare_case(case, run_dir, scene_id=scene_id)

    def stop_scene(self, scene_instance_id: str) -> None:
        self.stop_count += 1
        super().stop_scene(scene_instance_id)

    def close_real_env(self) -> None:
        if self.env is not None:
            self.close_count += 1
            self.env = None
        if self.scene is not None:
            self.scene.status = "stopped"

    def _init_case_objects_in_real_env(self) -> None:
        self._case_objects_initialized = True

    def _place_case_objects(self) -> None:
        return

    def _apply_pose_to_real_env(self) -> None:
        return

    def _rgb_frame(self) -> dict:
        self._frame_index += 1
        path = self.run_dir / "artifacts" / "rgb" / f"frame_{self._frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_dry_rgb(path)
        return {
            "path": str(path),
            "width": 224,
            "height": 224,
            "mock": False,
            "source": "uav_flow_benchmark",
            "case_id": self.case.case_id,
        }
