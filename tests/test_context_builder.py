"""Test model-ready observation context building."""

from __future__ import annotations

import base64
import json

import pytest

from src.agent.planning.model_inputs import build_model_user_content
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


def write_tiny_png(path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        goal="orient toward the left-side entrance search area",
        done_when=["left-side entrance search area is visible"],
        observe=["latest_observation", "phase_execution_summary"],
        continue_when=["left-side area is not yet visible"],
        metrics=[
            {
                "name": "yaw_delta",
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
    assert context["semantic_goal"] == state.semantic_plan.task_goal.to_dict()
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
    assert "phase_transition_policy" not in context
    assert "phase_completion_criteria" not in context
    assert context["current_phase"]["metrics"] == [
        {
            "name": "yaw_delta",
            "target": -30.0,
            "unit": "deg",
            "tolerance": 8.0,
            "correction_allowed": True,
        }
    ]
    assert context["phase_execution_summary"] == {}
    assert "phase_completion_criteria" not in context["context_schema"]["planner_context_fields"]
    assert "phase_execution_summary" in context["context_schema"]["planner_context_fields"]
    assert "forbidden_tools" not in action_constraints
    assert "lifecycle_tools_excluded" not in action_constraints
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
    instruction = fragments["instruction"]["content"]
    assert instruction["current_instruction"]["text"] == state.user_instruction
    assert "current_phase" not in instruction
    assert instruction["instruction_history"][0]["instruction"] == state.user_instruction
    assert "semantic_plan" not in instruction["instruction_history"][0]
    assert "phases" not in instruction["instruction_history"][0]
    assert "completion_criteria" not in instruction["instruction_history"][0]
    assert fragments["image"]["content"]["selected_frames"][0]["role"] == "current_observation"
    assert fragments["image"]["model_inputs"] == []
    assert fragments["image"]["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"
    assert "visual_processing" not in fragments["image"]["content"]
    image_schema = context["context_schema"]["module_content_fields"]["image"]["selected_frames"]
    assert "current_observation" in image_schema
    assert "phase_start_reference" in image_schema
    assert "previous" not in image_schema
    assert "history_anchor" not in image_schema
    assert "motion_history_summary" in context["context_schema"]["module_content_fields"]["action"]
    assert fragments["action"]["content"]["recent_actions"][-1]["tool"] == "uav.hover"
    assert fragments["action"]["content"]["recent_actions"][-1]["history_only"] is True


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
    write_tiny_png(rgb_path)
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
    assert image["content"]["selected_frames"][0]["role"] == "current_observation"
    assert image["content"]["selected_frames"][0]["input_id"] == "image_1_current_observation"
    assert image["content"]["selected_frames"][0]["usage"] == "Use as the current visual evidence for this decision."
    assert image["content"]["selected_frames"][0]["observation_id"] == "obs_0001"
    assert image["content"]["selected_frames"][0]["frame_id"] == "frame_0001"
    assert image["metadata"]["selected_frame_metadata"][0]["path"] == str(rgb_path)
    assert "recent_rgb_frame_metadata" not in image["metadata"]
    assert image["model_inputs"][0]["type"] == "image"
    assert image["model_inputs"][0]["input_id"] == "image_1_current_observation"
    assert image["model_inputs"][0]["role"] == "current_observation"
    assert image["model_inputs"][0]["usage"] == "Use as the current visual evidence for this decision."
    assert image["model_inputs"][0]["encoding"] == "data_url"
    assert set(image["model_inputs"][0]) == {"type", "input_id", "role", "usage", "mime_type", "encoding", "data"}
    assert "visual_processing" not in image["content"]
    assert "visual_description" not in image["content"]


def test_context_builder_default_image_attaches_key_rgb_frames(tmp_path):
    state = make_state()
    observations = []
    paths = []
    for index in range(6):
        rgb_path = tmp_path / f"frame_{index}.png"
        rgb_path.write_bytes(b"fake-png")
        paths.append(str(rgb_path))
        observations.append(
            {
                "scene_instance_id": "scene_1",
                "agent_id": "uav_1",
                "modalities": ["rgb"],
                "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
                "source": f"phase_window_{index}",
                "timestamp": f"t{index}",
            }
        )

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, observations=observations))

    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    image = fragments["image"]
    roles = [item["role"] for item in image["content"]["selected_frames"]]
    source_paths = [item["path"] for item in image["metadata"]["selected_frame_metadata"]]
    assert roles == [
        "current_observation",
        "phase_start_reference",
    ]
    assert source_paths == [paths[5], paths[0]]
    assert [item["input_id"] for item in image["content"]["selected_frames"]] == [
        "image_1_current_observation",
        "image_2_phase_start_reference",
    ]
    assert len(image["model_inputs"]) == 2
    assert all(item["encoding"] == "data_url" for item in image["model_inputs"])
    assert all("metadata" not in item and "source" not in item for item in image["model_inputs"])


def test_context_builder_default_image_deduplicates_key_rgb_frames(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "same.png"
    rgb_path.write_bytes(b"fake-png")
    observations = [
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
            "source": f"phase_window_{index}",
            "timestamp": f"t{index}",
        }
        for index in range(5)
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, observations=observations))

    image = {fragment["name"]: fragment for fragment in context["fragments"]}["image"]
    assert image["content"]["selected_frames"][0]["role"] == "current_observation"
    assert len(image["metadata"]["selected_frame_metadata"]) == 1
    assert "recent_rgb_frame_metadata" not in image["metadata"]
    assert len(image["model_inputs"]) == 1


def test_context_builder_default_image_prefers_current_phase_anchor(tmp_path):
    state = make_state()
    observations = []
    paths = []
    for index in range(4):
        rgb_path = tmp_path / f"anchored_{index}.png"
        rgb_path.write_bytes(b"fake-png")
        paths.append(str(rgb_path))
        observations.append(
            {
                "scene_instance_id": "scene_1",
                "agent_id": "uav_1",
                "modalities": ["rgb"],
                "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
                "source": f"source_{index}",
                "timestamp": f"t{index}",
            }
        )
    memory = make_memory(tmp_path, state, observations=observations)
    memory.set_observation_anchor("task_initial", memory.recent_observation_refs[1])
    memory.set_observation_anchor("phase_start", memory.recent_observation_refs[2])

    context = ObservationContextBuilder().build(state, memory)

    image = {fragment["name"]: fragment for fragment in context["fragments"]}["image"]
    roles = [item["role"] for item in image["content"]["selected_frames"]]
    source_paths = [item["path"] for item in image["metadata"]["selected_frame_metadata"]]
    assert roles == ["current_observation", "phase_start_reference"]
    assert source_paths == [paths[3], paths[2]]


def test_model_user_content_omits_fragment_debug_metadata(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "frame.png"
    write_tiny_png(rgb_path)
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

    content = build_model_user_content(context)
    text = content[0]["text"] if isinstance(content, list) else content

    assert "selected_frame_metadata" not in text
    assert str(rgb_path) not in text
    assert "safety_constraints" not in text
    assert "observation_policy" not in text
    assert "yaw_scan_remaining_deg" not in text


def test_context_builder_action_trend_summarizes_recent_actions(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    trend = action["content"]["action_trend"]
    summary = action["content"]["motion_history_summary"]
    assert trend["tool_sequence"] == ["uav.turn_relative", "uav.turn_relative", "uav.turn_relative"]
    assert trend["repetition"] == {"last_tool": "uav.turn_relative", "consecutive_count": 3}
    assert trend["motion_pattern"]["type"] == "alternating_yaw"
    assert summary["scope"] == "current_phase_when_available"
    assert summary["recent_yaw_deg"] == [5.0, -5.0, 5.0]
    assert summary["recent_yaw_sign_changes"] == 2
    assert summary["recent_yaw_pattern"] == "alternating_yaw"
    assert summary["history_only"] is True
    assert len(action["content"]["recent_actions"]) == 2
    assert all(item["history_only"] is True for item in action["content"]["recent_actions"])


def test_context_builder_action_trend_keeps_mixed_yaw_from_becoming_repeated(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 10}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 10}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -10}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    trend = action["content"]["action_trend"]
    summary = action["content"]["motion_history_summary"]
    assert trend["motion_pattern"]["type"] == "mixed"
    assert "history-only mixed corrections" in trend["motion_pattern"]["summary"]
    assert summary["recent_yaw_deg"] == [10.0, 10.0, -10.0, -5.0]
    assert summary["recent_yaw_sign_changes"] == 1
    assert summary["recent_yaw_pattern"] == "mixed"


def test_context_builder_action_trend_detects_repeated_move(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    assert action["content"]["action_trend"]["motion_pattern"]["type"] == "repeated_move"


def test_context_builder_runs_custom_multimodal_image_only_when_configured(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "frame.png"
    write_tiny_png(rgb_path)
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
    assert context["phase_execution_summary"]["measurements"]["planar_m"] == 1.25
    assert context["phase_execution_summary"]["last_window"]["planar_m"] == 1.25
    assert "latest_window_measurements" not in context["phase_execution_summary"]
    assert "start_pose" not in context["phase_execution_summary"]
    assert "latest_pose" not in context["phase_execution_summary"]
    assert "phase_local_frame" not in context["phase_execution_summary"]
    assert "world_pose" not in context["phase_execution_summary"]
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
    assert "yaw_step_deg" not in constraints
    assert "linear_velocity_mps" not in constraints
    assert "vertical_velocity_mps" not in constraints
    assert "move_duration_sec" not in constraints
    assert "safety_limits" not in constraints
    assert constraints["nominal_action_scale"] == {
        "linear_velocity_mps": 1.5,
        "vertical_velocity_mps": 0.75,
    }


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
