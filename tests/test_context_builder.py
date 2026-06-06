"""Test model-ready observation context building."""

from __future__ import annotations

import json

import pytest

from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticPhase
from src.agent.semantic.task_parser import TaskParser
from plugins.research.multimodal_image.context_modules import MultimodalImageContextModule
from src.memory.context.registry import ContextModuleRegistry
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.safety_config import UavSafetyConfig


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        meaning="向左调整视角，寻找入口可能区域",
        completion_criteria=[
            {
                "kind": "rotation",
                "axis": "yaw",
                "target": -30.0,
                "unit": "deg",
                "tolerance": 8.0,
                "correction_allowed": True,
            }
        ],
    )
    return MissionState("run_test", result.instruction, result.plan)


def make_memory(tmp_path, state: MissionState, actions=None, observations=None) -> FlightMemory:
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_test", "m1", "scene_1", "uav_1", active_instruction=state.user_instruction, memory_reader=reader)
    memory.update_instruction_ref(writer.write_instruction(state.user_instruction, state.semantic_plan.to_dict()), state.user_instruction)
    for action in actions or []:
        result = writer.write_action(action.get("tool", ""), action.get("args", {}), action.get("result", {}))
        memory.append_action_ref(result["action_ref"])
        memory.append_pose_ref(result.get("pose_ref"))
    default_observations = [state.latest_observation] if state.latest_observation else []
    for observation in observations if observations is not None else default_observations:
        result = writer.write_observation(observation)
        memory.append_observation_ref(result["observation_ref"])
        memory.append_pose_ref(result.get("pose_ref"))
        memory.append_frame_refs(result.get("frame_refs"))
    return memory


def test_context_builder_includes_mission_phase_observation_actions_and_safety(tmp_path):
    state = make_state()
    state.record_action("uav.hover", {}, {"ok": True})
    state.record_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 0.0},
            "frames": {"rgb": {"mock": True}},
            "timestamp": "t0",
            "source": "phase_step",
        }
    )

    context = ObservationContextBuilder().build(
        state,
        make_memory(tmp_path, state, actions=[{"tool": "uav.hover", "args": {}, "result": {"ok": True}}]),
    )

    assert context["active_instruction"] == state.user_instruction
    assert context["semantic_goal"] == state.semantic_plan.goal.to_dict()
    assert context["current_phase"]["name"] == "orient_leftward"
    assert context["current_phase"]["index"] == 1
    assert context["safety_constraints"] == {
        "hover_required": False,
        "yaw_scan_accum_deg": 0.0,
        "yaw_scan_remaining_deg": 480.0,
        "max_yaw_scan_accum_deg": 480.0,
        "yaw_scan_budget_exceeded": False,
    }
    action_constraints = context["allowed_action_constraints"]
    assert action_constraints["planner_tools"] == ["uav.hover", "uav.turn_relative", "uav.move_body"]
    assert "observe_step_schema" not in action_constraints
    assert context["observation_policy"] == {
        "mode": "automatic",
        "modalities": ["rgb", "depth", "pose"],
        "sample_timing": "after_each_phase_window",
        "note": "Do not output observe steps. The runtime samples observations automatically.",
    }
    assert context["phase_transition_policy"] == {
        "mode": "post_action_model_judged_runtime_applied",
        "decisions": ["continue_current_phase", "advance_to_next_phase", "complete_task"],
        "current_phase_window_count": 0,
        "max_phase_window_count": 5,
        "note": "Phase transition is judged after tool execution and automatic observation. Do not output phase_transition in the action plan. Window counts are runtime loop guard diagnostics, not phase-completion evidence.",
    }
    assert context["phase_completion_criteria"] == [
        {
            "kind": "rotation",
            "axis": "yaw",
            "target": -30.0,
            "unit": "deg",
            "tolerance": 8.0,
            "correction_allowed": True,
        }
    ]
    assert context["phase_execution_summary"] == {}
    assert "phase_transition_policy" in context["context_schema"]["planner_context_fields"]
    assert "phase_completion_criteria" in context["context_schema"]["planner_context_fields"]
    assert "phase_execution_summary" in context["context_schema"]["planner_context_fields"]
    assert "uav.connect" in action_constraints["lifecycle_tools_excluded"]
    assert "uav.connect" in action_constraints["forbidden_tools"]
    assert "uav.set_pose" in action_constraints["forbidden_tools"]
    assert "env.observe" in action_constraints["forbidden_tools"]
    assert action_constraints["action_schemas"]["uav.hover"]["args"] == {}
    assert action_constraints["action_schemas"]["uav.turn_relative"]["args"] == {"yaw_deg": "number"}
    assert action_constraints["action_schemas"]["uav.move_body"]["args_one_of"][0]["direction"] == (
        "forward|backward|left|right|up|down"
    )
    assert "flight_memory" not in context
    assert "recent_observations" not in context
    assert "instruction_history" not in context
    assert context["metadata"]["run_id"] == state.run_id
    assert context["metadata"]["current_phase_index"] == state.current_phase_index
    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert fragments["image"]["content"]["recent_observation_sources"][-1] == "phase_step"
    assert fragments["image"]["model_inputs"] == []
    assert fragments["image"]["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"
    assert "visual_processing" not in fragments["image"]["content"]
    assert fragments["action"]["content"]["recent_actions"][-1]["tool"] == "uav.hover"


def test_context_builder_handles_missing_observation(tmp_path):
    state = make_state()

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))

    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert fragments["image"]["warnings"] == ["no_recent_image_frames"]
    assert fragments["image"]["model_inputs"] == []
    assert fragments["image"]["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"
    assert "visual_processing" not in fragments["image"]["content"]
    assert fragments["action"]["content"]["recent_actions"] == []
    assert context["current_phase"]["name"] == "orient_leftward"


def test_context_builder_default_image_attaches_real_rgb_without_caption(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "frame.png"
    rgb_path.write_bytes(b"fake-png")
    state.record_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
            "source": "phase_step",
        }
    )

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))

    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    image = fragments["image"]
    assert image["content"]["visual_evidence"]["interpretation_status"] == "image_attached_not_interpreted"
    assert image["content"]["latest_frame_metadata"]["rgb"]["path"] == str(rgb_path)
    assert image["model_inputs"][0]["type"] == "image"
    assert image["model_inputs"][0]["encoding"] == "data_url"
    assert image["model_inputs"][0]["metadata"]["source_path"] == str(rgb_path)
    assert "visual_processing" not in image["content"]
    assert "visual_description" not in image["content"]


def test_context_builder_runs_custom_multimodal_image_only_when_configured(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "frame.png"
    rgb_path.write_bytes(b"fake-png")
    state.record_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
            "source": "phase_step",
        }
    )
    seen_phases = []

    def caption_client(config, payload):
        prompt_context = json.loads(payload["messages"][1]["content"][0]["text"])
        seen_phases.append(prompt_context["current_phase"])
        return {
            "parsed": {
                "visual_description": {
                    "phase": prompt_context["current_phase"],
                    "phase_relevance": "入口候选可见。",
                    "scene_summary": "建筑底层入口候选位于画面中景。",
                    "spatial_overview": "midground-center: 建筑底层开口。",
                    "relevant_objects": [
                        {
                            "name": "建筑底层开口",
                            "position": "midground-center",
                            "role": "target_candidate",
                            "phase_relevance": "与入口目标同类。",
                            "description": "矩形开口位于建筑底层。",
                            "confidence": "high",
                        }
                    ],
                    "free_space": [],
                    "landmarks": [],
                    "phase_relevant_cues": ["建筑底层存在矩形开口"],
                    "uncertainty": [],
                    "confidence": "medium",
                }
            },
            "duration_sec": 0.1,
            "usage": {"total_tokens": 12},
        }

    registry = ContextModuleRegistry(
        {
            "image": MultimodalImageContextModule(
                model_config=ModelConfig(
                    model_name="qwen3-vl-flash",
                    api_key="secret-key",
                    base_url="https://example.test/v1",
                ),
                caption_client=caption_client,
            )
        }
    )

    context = ObservationContextBuilder(registry=registry).build(state, make_memory(tmp_path, state))

    image = context["fragments"][0]
    assert image["name"] == "image"
    assert seen_phases[0]["name"] == "orient_leftward"
    assert image["content"]["visual_processing"]["status"] == "captioned"
    assert image["content"]["visual_description"]["phase"]["name"] == "orient_leftward"
    assert image["model_inputs"][0]["type"] == "text"
    assert image["model_inputs"][0]["role"] == "visual_caption"
    assert "data:image/png;base64," not in image["model_inputs"][0]["text"]


def test_context_builder_includes_phase_execution_summary(tmp_path):
    state = make_state()
    state.record_observation({"pose": {"x": 0.0, "y": 0.0, "z": -2.0, "velocity": {"x": 9.0}}})
    state.ensure_current_phase_progress({"pose": {"x": 0.0, "y": 0.0, "z": -2.0, "velocity": {"x": 9.0}}})
    state.update_current_phase_progress(
        {"pose": {"x": 1.25, "y": 0.0, "z": -2.0, "velocity": {"x": 8.0}}},
        last_window_measurements={"planar_displacement_m": 1.25},
    )

    context = ObservationContextBuilder().build(
        state,
        make_memory(tmp_path, state, actions=[{"tool": "uav.hover", "args": {}, "result": {"ok": True}}]),
    )

    assert context["phase_execution_summary"]["phase"] == "orient_leftward"
    assert context["phase_execution_summary"]["phase_key"] == "1:orient_leftward"
    assert context["phase_execution_summary"] == state.to_dict()["current_phase_progress"]
    assert context["phase_execution_summary"]["cumulative_measurements"]["planar_displacement_m"] == 1.25
    assert context["phase_execution_summary"]["last_window_measurements"]["planar_displacement_m"] == 1.25
    assert "latest_window_measurements" not in context["phase_execution_summary"]
    assert "start_pose" not in context["phase_execution_summary"]
    assert "latest_pose" not in context["phase_execution_summary"]
    assert "phase_local_frame" in context["phase_execution_summary"]
    assert "velocity" not in context["phase_execution_summary"]["world_pose"]["start"]
    assert "velocity" not in context["phase_execution_summary"]["world_pose"]["latest"]
    assert "velocity" not in state.to_dict()["latest_observation"]["pose"]


def test_context_builder_action_limits_come_from_safety_config(tmp_path):
    state = make_state()
    action_config = UavActionConfig(
        linear_velocity_mps=1.5,
        vertical_velocity_mps=0.75,
        move_duration_sec=1.25,
        yaw_step_deg=15.0,
    )
    safety_config = UavSafetyConfig(
        max_yaw_step_deg=25.0,
        max_linear_velocity_mps=6.0,
        max_vertical_velocity_mps=3.0,
        max_move_duration_sec=8.0,
        max_yaw_scan_accum_deg=480.0,
    )

    context = ObservationContextBuilder(action_config=action_config, safety_config=safety_config).build(
        state,
        make_memory(tmp_path, state),
    )

    constraints = context["allowed_action_constraints"]
    turn_limits = constraints["action_schemas"]["uav.turn_relative"]["limits"]
    move_limits = constraints["action_schemas"]["uav.move_body"]["limits"]
    assert turn_limits == {"yaw_deg_min": -25.0, "yaw_deg_max": 25.0}
    assert move_limits == {
        "abs_vx_vy_max": 6.0,
        "abs_vz_max": 3.0,
        "duration_max": 8.0,
    }
    assert constraints["yaw_step_deg"] == 25.0
    assert constraints["linear_velocity_mps"] == 6.0
    assert constraints["vertical_velocity_mps"] == 3.0
    assert constraints["move_duration_sec"] == 8.0
    assert constraints["nominal_action_scale"]["move_duration_sec"] == 1.25
    assert constraints["safety_limits"] == safety_config.to_dict()


def test_context_builder_exposes_yaw_scan_budget(tmp_path):
    state = make_state()
    state.yaw_scan_accum_deg = 450.0
    safety_config = UavSafetyConfig(max_yaw_scan_accum_deg=480.0)

    context = ObservationContextBuilder(safety_config=safety_config).build(state, make_memory(tmp_path, state))

    safety_constraints = context["safety_constraints"]
    assert safety_constraints["yaw_scan_accum_deg"] == 450.0
    assert safety_constraints["yaw_scan_remaining_deg"] == 30.0
    assert safety_constraints["max_yaw_scan_accum_deg"] == 480.0
    assert safety_constraints["yaw_scan_budget_exceeded"] is False


def test_context_builder_loads_default_project_map_memory(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory" / "map"
    memory_dir.mkdir(parents=True)
    (memory_dir / "map_memory.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "map_id": "m1",
                        "scene_id": "scene_1",
                        "landmark_description": "black street lamp candidate",
                        "region_hint": "start_area_right",
                        "confidence": "medium",
                        "requires_reconfirmation": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    state = make_state()

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    map_fragment = next(fragment for fragment in context["fragments"] if fragment["name"] == "map")

    assert map_fragment["content"]["recalled_map_memories"][0]["landmark_description"] == "black street lamp candidate"
    assert map_fragment["facts"] == []
    assert map_fragment["inferences"][0]["type"] == "map_memory_hint"
    assert map_fragment["risks"][0]["type"] == "map_memory_requires_reconfirmation"


def test_context_builder_default_project_map_memory_filters_other_maps(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory" / "map"
    memory_dir.mkdir(parents=True)
    (memory_dir / "map_memory.json").write_text(
        json.dumps({"entries": [{"map_id": "m2", "scene_id": "scene_1", "landmark_description": "other map clue"}]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    state = make_state()

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    map_fragment = next(fragment for fragment in context["fragments"] if fragment["name"] == "map")

    assert map_fragment["content"]["recalled_map_memories"] == []
    assert map_fragment["inferences"] == []
    assert map_fragment["risks"] == []


def test_context_builder_requires_flight_memory_manifest():
    state = make_state()

    with pytest.raises(ValueError, match="requires a FlightMemory manifest"):
        ObservationContextBuilder().build(state)
