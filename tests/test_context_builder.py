"""Test model-ready observation context building."""

from __future__ import annotations

import base64
from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image

from src.agent.planning.model_inputs import build_model_user_content
from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.agent.semantic.task_parser import TaskParser
from plugins.research.multimodal_image.context_modules import MultimodalImageContextModule
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.modules.image import (
    DefaultImageContextModule,
    _banner_height,
    _image_banner,
    _rgb_depth_overlay_data_url,
)
from src.memory.context.modules.instruction import _summarize_instruction_history
from src.memory.context.modules.action import DefaultActionContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.modules.pose import DefaultPoseContextModule
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.plugins.skills.selection import ModelSkillSelector
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.safety_config import UavSafetyConfig


def write_tiny_png(path) -> None:
    path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    )


def write_test_png(path, color=(12, 34, 56), size=(320, 240)) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


def decode_data_url_image(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def test_context_registry_preserves_explicit_skill_selector_identity():
    config = ModelConfig(
        model_name="test-model",
        api_key="secret",
        base_url="https://example.test/v1",
    )
    selector = ModelSkillSelector(config)

    registry = ContextModuleRegistry.default(skill_selector=selector)
    builder = ObservationContextBuilder(skill_selector=selector)

    assert registry.modules["skills"].skill_selector is selector
    assert builder.composer.registry.modules["skills"].skill_selector is selector


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        goal="orient toward the left-side entrance search area",
        done_when=["left-side entrance search area is visible"],
        continue_when=["left-side area is not yet visible"],
    )
    return MissionState("run_test", result.instruction, result.plan)


def test_instruction_history_excludes_active_record_by_id_not_text():
    records = [
        {
            "id": "inst_0001",
            "instruction": "repeat this instruction",
            "semantic_plan": {"intent": "first"},
        },
        {
            "id": "inst_0002",
            "instruction": "repeat this instruction",
            "semantic_plan": {"intent": "second"},
        },
    ]

    history = _summarize_instruction_history(
        records,
        active_instruction_id="inst_0002",
        max_items=5,
    )

    assert history == [
        {
            "id": "inst_0001",
            "instruction": "repeat this instruction",
            "intent": "first",
        }
    ]


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


def test_semantic_evidence_capabilities_follow_module_order_and_depth_config():
    registry = ContextModuleRegistry(
        {
            "image": DefaultImageContextModule(include_current_depth=True),
            "pose": DefaultPoseContextModule(),
            "action": DefaultActionContextModule(),
            "map": DefaultMapContextModule(),
        },
        module_order=["image", "pose", "action", "map"],
    )

    capabilities = registry.semantic_evidence_capabilities()

    assert capabilities[:3] == [
        "NOW RGB: current visible appearance and image-space relations.",
        "PAST RGB when distinct: phase-start comparison only; not current visibility.",
        "NOW metric depth overlay: approximate UAV-camera-to-scene distance at visible pixels; target pixels provide approximate UAV-to-target range.",
    ]
    assert capabilities[3].startswith("UAV ground clearance")
    assert capabilities[4].startswith("Action history:")
    assert capabilities[5].startswith("Map memory:")
    assert ObservationContextBuilder(registry=registry).semantic_evidence_capabilities() == capabilities


def test_semantic_evidence_capabilities_filter_disabled_invalid_duplicate_and_over_budget():
    class DescriptionOnlyModule:
        def __init__(self, evidence):
            self.evidence = evidence

        def describe_output(self):
            return {"semantic_evidence": self.evidence}

    long_item = "long capability " + ("x" * 240)
    modules = {
        "disabled": DescriptionOnlyModule(["disabled capability"]),
        "first": DescriptionOnlyModule([" first   capability ", 7, "", long_item]),
        "second": DescriptionOnlyModule(
            ["first capability", *[f"capability {index}" for index in range(10)]]
        ),
        "broken": DescriptionOnlyModule("not a list"),
    }
    registry = ContextModuleRegistry(
        modules,
        disabled=["disabled"],
        module_order=["disabled", "first", "second", "broken"],
    )

    capabilities = registry.semantic_evidence_capabilities()

    assert capabilities[0] == "first capability"
    assert capabilities.count("first capability") == 1
    assert len(capabilities) == ContextModuleRegistry.MAX_SEMANTIC_EVIDENCE_ITEMS
    assert max(map(len, capabilities)) <= ContextModuleRegistry.MAX_SEMANTIC_EVIDENCE_CHARS
    assert "disabled capability" not in capabilities


def test_image_semantic_evidence_omits_depth_when_depth_input_is_disabled():
    capabilities = ContextModuleRegistry(
        {"image": DefaultImageContextModule(include_current_depth=False)},
        module_order=["image"],
    ).semantic_evidence_capabilities()

    assert any(item.startswith("NOW RGB:") for item in capabilities)
    assert not any("depth" in item.lower() for item in capabilities)


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
    assert context["phase_neighborhood"] == {
        "previous": None,
        "next": {
            "index": state.semantic_plan.phases[1].index,
            "name": state.semantic_plan.phases[1].name,
            "goal": state.semantic_plan.phases[1].goal,
            "status": "upcoming",
        },
    }
    assert context["safety_constraints"] == {
        "hover_required": False,
        "yaw_scan_offset_deg": 0.0,
        "yaw_scan_extent_deg": 0.0,
        "yaw_scan_remaining_deg": 480.0,
        "max_yaw_scan_extent_deg": 480.0,
        "yaw_scan_budget_exceeded": False,
        "yaw_scan_turn_count": 0,
        "yaw_scan_recharge_count": 0,
        "yaw_scan_recharge_distance_m": 0.5,
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
    assert "metrics" not in context["current_phase"]
    assert context["phase_execution_summary"] == {}
    assert "phase_completion_criteria" not in context["context_schema"]["planner_context_fields"]
    assert "phase_execution_summary" in context["context_schema"]["planner_context_fields"]
    assert "phase_neighborhood" in context["context_schema"]["planner_context_fields"]
    assert "forbidden_tools" not in action_constraints
    assert "lifecycle_tools_excluded" not in action_constraints
    assert action_constraints["action_schemas"]["uav.hover"]["args"] == {}
    assert action_constraints["action_schemas"]["uav.turn_relative"]["args"] == {"yaw_deg": "number"}
    assert action_constraints["action_schemas"]["uav.move_body"]["args_one_of"] == [
        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"}
    ]
    assert "flight_memory" not in context
    assert "recent_observations" not in context
    assert "instruction_history" not in context
    assert context["metadata"]["run_id"] == state.run_id
    assert context["metadata"]["current_phase_index"] == state.current_phase_index
    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert "custom_map" not in fragments
    instruction = fragments["instruction"]["content"]
    assert instruction["current_instruction"]["text"] == state.user_instruction
    assert "current_phase" not in instruction
    assert instruction["instruction_history"] == []
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
    recent_execution = fragments["action"]["content"]["motion_history_summary"]["recent_execution"]
    assert recent_execution[-1] == {"tool": "uav.hover", "args": {}}


def test_context_builder_exposes_only_compact_adjacent_phase_semantics(tmp_path):
    state = make_state()
    state.current_phase_index = 1

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))

    previous = context["phase_neighborhood"]["previous"]
    next_phase = context["phase_neighborhood"]["next"]
    assert previous == {
        "index": state.semantic_plan.phases[0].index,
        "name": state.semantic_plan.phases[0].name,
        "goal": state.semantic_plan.phases[0].goal,
        "status": "completed",
    }
    assert next_phase == {
        "index": state.semantic_plan.phases[2].index,
        "name": state.semantic_plan.phases[2].name,
        "goal": state.semantic_plan.phases[2].goal,
        "status": "upcoming",
    }
    assert set(previous) == {"index", "name", "goal", "status"}
    assert set(next_phase) == {"index", "name", "goal", "status"}
    assert "done_when" not in previous
    assert "continue_when" not in next_phase
    assert "metrics" not in next_phase
    assert context["current_phase"] == state.semantic_plan.phases[1].to_dict()

    state.current_phase_index = len(state.semantic_plan.phases) - 1
    final_context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    assert final_context["phase_neighborhood"]["previous"]["status"] == "completed"
    assert final_context["phase_neighborhood"]["next"] is None


def test_context_builder_reports_runtime_resolved_observation_modalities(tmp_path):
    state = make_state()
    state.observation_modalities = ("rgb", "pose")

    context = ObservationContextBuilder().build(
        state,
        make_memory(tmp_path, state),
    )

    assert context["observation_policy"]["modalities"] == ["rgb", "pose"]


def test_context_builder_handles_missing_observation(tmp_path):
    state = make_state()

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))

    fragments = {fragment["name"]: fragment for fragment in context["fragments"]}
    assert fragments["image"]["warnings"] == ["no_recent_image_frames"]
    assert fragments["image"]["model_inputs"] == []
    assert fragments["image"]["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"
    assert "visual_processing" not in fragments["image"]["content"]
    assert fragments["action"]["content"]["motion_history_summary"]["recent_execution"] == []
    assert "phase_strategy" not in fragments["action"]["content"]
    assert context["current_phase"]["name"] == "orient_leftward"


def test_context_builder_injects_only_current_phase_strategy_snapshot(tmp_path):
    state = make_state()
    strategy = {
        "stages": [
            "Move laterally until the route is open",
            "Move forward until the target has been passed",
        ],
        "active_stage": 1,
        "reason": "The lateral relation is established",
    }
    state.update_current_phase_strategy(strategy)

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]

    assert "phase_strategy" not in action["content"]
    assert context["phase_strategy"] == {
        "stages": strategy["stages"],
        "active_stage": 1,
        "previous_reason": "The lateral relation is established",
    }
    assert list(context)[:5] == [
        "active_instruction",
        "current_phase",
        "phase_strategy",
        "phase_neighborhood",
        "semantic_goal",
    ]
    model_content = build_model_user_content(context)
    model_text = model_content if isinstance(model_content, str) else model_content[-1]["text"]
    model_context = json.loads(model_text)
    assert model_context["phase_strategy"] == context["phase_strategy"]
    assert "reason" not in model_context["phase_strategy"]


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
    assert image["content"]["selected_frames"][0]["usage"] == "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions."
    assert image["content"]["selected_frames"][0]["observation_id"] == "obs_0001"
    assert image["content"]["selected_frames"][0]["frame_id"] == "frame_0001"
    assert image["metadata"]["selected_frame_metadata"][0]["path"] == str(rgb_path)
    assert "recent_rgb_frame_metadata" not in image["metadata"]
    assert image["model_inputs"][0]["type"] == "image"
    assert image["model_inputs"][0]["input_id"] == "image_1_current_observation"
    assert image["model_inputs"][0]["role"] == "current_observation"
    assert image["model_inputs"][0]["usage"] == "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions."
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
        write_test_png(rgb_path, color=(index, index + 1, index + 2))
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
        "phase_start_reference",
        "current_observation",
    ]
    assert source_paths == [paths[0], paths[5]]
    assert [item["input_id"] for item in image["content"]["selected_frames"]] == [
        "image_1_phase_start_reference",
        "image_2_current_observation",
    ]
    assert [item["type"] for item in image["model_inputs"]] == ["text", "image", "image", "text"]
    assert all(item["encoding"] == "data_url" for item in image["model_inputs"][1:3])
    assert all("metadata" not in item and "source" not in item for item in image["model_inputs"])
    assert [item["usage"] for item in image["model_inputs"][1:3]] == [
        "PAST | PHASE START. Comparison only.",
        "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions.",
    ]
    assert image["model_inputs"][0]["role"] == "temporal_evidence_rule"
    assert image["model_inputs"][0]["text"].startswith("TEMPORAL ROLE RULE:")
    assert image["model_inputs"][3]["role"] == "current_state_decision_reminder"
    assert image["model_inputs"][3]["text"].startswith("CURRENT-STATE CHECK:")
    assert image["model_inputs"][0] == {
        "type": "text",
        "role": "temporal_evidence_rule",
        "text": (
            "TEMPORAL ROLE RULE: PAST is historical comparison evidence only. "
            "Current visibility, image position, approximate range, and the next action "
            "must be judged from NOW. An object visible only in PAST is not currently visible."
        ),
    }
    model_content = build_model_user_content(context)
    assert isinstance(model_content, list)
    assert [part["type"] for part in model_content] == [
        "text", "text", "image_url", "text", "image_url", "text", "text"
    ]
    assert model_content[0]["text"].startswith("TEMPORAL ROLE RULE:")
    assert model_content[1]["text"] == "PAST | PHASE START. Comparison only."
    assert model_content[3]["text"] == "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions."
    assert model_content[5]["text"].startswith("CURRENT-STATE CHECK:")
    text_context = json.loads(model_content[6]["text"])
    assert not any(fragment.get("name") == "image" for fragment in text_context.get("fragments", []))
    assert "visual_evidence" not in model_content[6]["text"]
    assert "selected_frames" not in model_content[6]["text"]


def test_context_builder_image_can_omit_phase_start_reference(tmp_path):
    state = make_state()
    observations = []
    for index in range(4):
        rgb_path = tmp_path / f"compressed_{index}.png"
        write_test_png(
            rgb_path,
            color=(index + 20, index + 30, index + 40),
        )
        observations.append(
            {
                "scene_instance_id": "scene_1",
                "agent_id": "uav_1",
                "modalities": ["rgb"],
                "frames": {
                    "rgb": {"path": str(rgb_path), "mock": False}
                },
                "source": f"phase_window_{index}",
            }
        )
    default_registry = ContextModuleRegistry.default()
    modules = dict(default_registry.modules)
    modules["image"] = DefaultImageContextModule(
        include_phase_start_reference=False
    )
    registry = ContextModuleRegistry(
        modules,
        disabled=default_registry.disabled,
        module_order=default_registry.module_order,
    )

    context = ObservationContextBuilder(registry=registry).build(
        state,
        make_memory(tmp_path, state, observations=observations),
    )

    image = {
        fragment["name"]: fragment for fragment in context["fragments"]
    }["image"]
    assert [
        item["role"] for item in image["content"]["selected_frames"]
    ] == ["current_observation"]
    assert [item["role"] for item in image["model_inputs"]] == [
        "current_observation",
        "current_state_decision_reminder",
    ]
    assert image["model_inputs"][0]["input_id"] == (
        "image_1_current_observation"
    )
    assert "PAST RGB" not in " ".join(
        modules["image"].describe_output()["semantic_evidence"]
    )


def test_context_builder_optionally_attaches_aligned_current_rgb_depth_overlay(tmp_path):
    state = make_state()
    observations = []
    for index in range(2):
        rgb_path = tmp_path / f"rgb_{index}.png"
        depth_path = tmp_path / f"depth_{index}.npy"
        write_test_png(rgb_path, color=(20 + index, 40 + index, 60 + index), size=(6, 4))
        np.save(depth_path, np.full((4, 6), 3.5 + index, dtype=np.float32))
        observations.append(
            {
                "scene_instance_id": "scene_1",
                "agent_id": "uav_1",
                "modalities": ["rgb", "depth"],
                "frames": {
                    "rgb": {"path": str(rgb_path), "mock": False},
                    "depth": {
                        "path": str(depth_path),
                        "mock": False,
                        "depth_unit": "meter",
                    },
                },
                "source": f"phase_window_{index}",
            }
        )
    default_registry = ContextModuleRegistry.default()
    modules = dict(default_registry.modules)
    modules["image"] = DefaultImageContextModule(include_current_depth=True)
    registry = ContextModuleRegistry(
        modules,
        disabled=default_registry.disabled,
        module_order=default_registry.module_order,
    )

    context = ObservationContextBuilder(registry=registry).build(
        state,
        make_memory(tmp_path, state, observations=observations),
    )

    image = {fragment["name"]: fragment for fragment in context["fragments"]}["image"]
    assert [item["role"] for item in image["model_inputs"]] == [
        "temporal_evidence_rule",
        "phase_start_reference",
        "current_rgb_depth_overlay",
        "current_state_decision_reminder",
    ]
    assert image["content"]["visual_evidence"]["has_depth"] is True
    assert image["content"]["selected_frames"][-1]["role"] == "current_depth"
    assert image["metadata"]["current_depth_metadata"]["path"] == str(tmp_path / "depth_1.npy")
    overlay_image = decode_data_url_image(image["model_inputs"][2]["data"])
    assert overlay_image.width == 6
    assert overlay_image.height > 4
    overlay_scene_pixel = overlay_image.getpixel((3, _banner_height(6, 4) + 2))
    assert overlay_scene_pixel != (21, 41, 61)

    model_content = build_model_user_content(context)
    assert isinstance(model_content, list)
    assert [part["type"] for part in model_content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
        "text",
    ]
    assert model_content[0]["text"].startswith("TEMPORAL ROLE RULE:")
    assert model_content[3]["text"].startswith("NOW | RGB + METRIC DEPTH OVERLAY.")
    assert model_content[5]["text"].startswith("CURRENT-STATE CHECK:")
    recovery_content = build_model_user_content(context, image_mode="current_only")
    assert isinstance(recovery_content, list)
    assert [part["type"] for part in recovery_content] == ["text", "image_url", "text", "text"]
    assert recovery_content[0]["text"].startswith("NOW | RGB + METRIC DEPTH OVERLAY.")
    assert recovery_content[2]["text"].startswith("CURRENT-STATE CHECK:")


def test_rgb_depth_overlay_uses_fixed_scale_not_per_frame_normalization(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    write_test_png(rgb_path, color=(40, 80, 120), size=(6, 4))
    first = np.full((4, 6), 3.5, dtype=np.float32)
    second = np.full((4, 6), 3.5, dtype=np.float32)
    first[0, 0] = 5.0
    second[0, 0] = 9.5
    np.save(first_path, first)
    np.save(second_path, second)

    first_url = _rgb_depth_overlay_data_url(
        rgb_path,
        first_path,
        metadata={},
        depth_display_max_m=10.0,
    )
    second_url = _rgb_depth_overlay_data_url(
        rgb_path,
        second_path,
        metadata={},
        depth_display_max_m=10.0,
    )

    assert first_url is not None
    assert second_url is not None
    first_image = decode_data_url_image(first_url)
    second_image = decode_data_url_image(second_url)
    scene_top = _banner_height(6, 4)
    assert first_image.getpixel((3, scene_top + 2)) == second_image.getpixel((3, scene_top + 2))
    assert first_image.getpixel((0, scene_top)) != second_image.getpixel((0, scene_top))


def test_rgb_depth_overlay_rejects_misaligned_shapes(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    write_test_png(rgb_path, color=(40, 80, 120), size=(6, 4))
    np.save(depth_path, np.full((3, 6), 3.5, dtype=np.float32))

    assert (
        _rgb_depth_overlay_data_url(
            rgb_path,
            depth_path,
            metadata={},
            depth_display_max_m=10.0,
        )
        is None
    )


def test_model_user_content_current_only_omits_past_and_temporal_rule(tmp_path):
    state = make_state()
    observations = []
    for index in range(2):
        rgb_path = tmp_path / f"frame_{index}.png"
        write_test_png(rgb_path, color=(index + 10, index + 20, index + 30))
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
    context = ObservationContextBuilder().build(
        state,
        make_memory(tmp_path, state, observations=observations),
    )

    model_content = build_model_user_content(context, image_mode="current_only")

    assert isinstance(model_content, list)
    assert [part["type"] for part in model_content] == ["text", "image_url", "text", "text"]
    assert model_content[0]["text"] == (
        "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions."
    )
    combined_text = "\n".join(
        part["text"] for part in model_content if part["type"] == "text"
    )
    assert "PAST | PHASE START" not in combined_text
    assert "PAST is historical comparison evidence only" not in combined_text
    assert model_content[2]["text"].startswith("CURRENT-STATE CHECK:")


def test_context_builder_default_image_deduplicates_key_rgb_frames(tmp_path):
    state = make_state()
    rgb_path = tmp_path / "same.png"
    write_test_png(rgb_path)
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
    assert [item["role"] for item in image["model_inputs"]] == [
        "current_observation",
        "current_state_decision_reminder",
    ]


def test_context_builder_default_image_prefers_current_phase_anchor(tmp_path):
    state = make_state()
    observations = []
    paths = []
    for index in range(4):
        rgb_path = tmp_path / f"anchored_{index}.png"
        write_test_png(rgb_path, color=(index, index + 1, index + 2))
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
    assert roles == ["phase_start_reference", "current_observation"]
    assert source_paths == [paths[2], paths[3]]


def test_context_builder_annotates_model_images_without_changing_source_pixels(tmp_path):
    state = make_state()
    current_path = tmp_path / "current.png"
    reference_path = tmp_path / "reference.png"
    write_test_png(current_path, color=(10, 40, 90))
    write_test_png(reference_path, color=(90, 40, 10))
    current_bytes = current_path.read_bytes()
    reference_bytes = reference_path.read_bytes()
    observations = [
        {
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(reference_path), "mock": False}},
            "source": "phase_start",
        },
        {
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(current_path), "mock": False}},
            "source": "phase_window",
        },
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, observations=observations))
    image = {fragment["name"]: fragment for fragment in context["fragments"]}["image"]
    reference = decode_data_url_image(image["model_inputs"][1]["data"])
    current = decode_data_url_image(image["model_inputs"][2]["data"])

    assert image["model_inputs"][1]["mime_type"] == "image/png"
    assert image["model_inputs"][2]["mime_type"] == "image/png"
    assert current.width == reference.width == 320
    assert current.height > 240
    assert reference.height > 240
    current_banner_height = current.height - 240
    reference_banner_height = reference.height - 240
    assert current.getpixel((0, 0)) == (28, 122, 72)
    assert reference.getpixel((0, 0)) == (190, 117, 20)
    assert _image_banner("current_observation")[0] == "NOW | CURRENT OBSERVATION"
    assert _image_banner("phase_start_reference")[0] == "PAST | PHASE START"
    assert any(pixel == (255, 255, 255) for pixel in current.crop((0, 0, current.width, current_banner_height)).getdata())
    assert any(pixel == (255, 255, 255) for pixel in reference.crop((0, 0, reference.width, reference_banner_height)).getdata())
    assert list(current.crop((0, current_banner_height, 320, current.height)).getdata()) == [(10, 40, 90)] * (320 * 240)
    assert list(reference.crop((0, reference_banner_height, 320, reference.height)).getdata()) == [(90, 40, 10)] * (320 * 240)
    assert current_path.read_bytes() == current_bytes
    assert reference_path.read_bytes() == reference_bytes


def test_context_builder_does_not_attach_historical_image_when_current_is_unreadable(tmp_path):
    state = make_state()
    reference_path = tmp_path / "reference.png"
    current_path = tmp_path / "current.png"
    write_test_png(reference_path)
    current_path.write_bytes(b"not-an-image")
    observations = [
        {"modalities": ["rgb"], "frames": {"rgb": {"path": str(reference_path), "mock": False}}},
        {"modalities": ["rgb"], "frames": {"rgb": {"path": str(current_path), "mock": False}}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, observations=observations))
    image = {fragment["name"]: fragment for fragment in context["fragments"]}["image"]

    assert image["model_inputs"] == []
    assert image["content"]["visual_evidence"]["attached_image_count"] == 0


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
    text = (
        "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if isinstance(content, list)
        else content
    )

    assert "selected_frame_metadata" not in text
    assert str(rgb_path) not in text
    assert "safety_constraints" not in text
    assert "observation_policy" not in text
    assert "yaw_scan_remaining_deg" not in text
    assert "visual_search_progress" not in text
    assert "phase_start_relative_uav_motion" in text
    assert "not target-relative distance or bearing" in text
    assert '"direction"' not in text


def test_context_builder_motion_history_preserves_recent_execution_order(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    summary = action["content"]["motion_history_summary"]
    assert summary["recent_execution_order"] == "oldest_to_newest"
    assert action["metadata"]["visual_search_progress"] == {
        "scope": "current_stationary_viewpoint",
        "yaw_offset_deg": 0.0,
        "yaw_extent_deg": 0.0,
        "yaw_remaining_deg": 480.0,
        "turn_count": 0,
        "recharge_count": 0,
        "recharge_distance_m": 0.5,
    }
    assert summary["recent_execution"] == [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}},
    ]
    assert "action_trend" not in action["content"]
    assert "recent_actions" not in action["content"]


def test_context_builder_motion_history_keeps_only_five_successful_actions(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": value}, "result": {"ok": True}}
        for value in (5, 10, 15, 20, 25, 30)
    ]
    actions.insert(4, {"tool": "uav.turn_relative", "args": {"yaw_deg": 99}, "result": {"ok": False}})

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    summary = action["content"]["motion_history_summary"]
    assert [item["args"]["yaw_deg"] for item in summary["recent_execution"]] == [10, 15, 20, 25, 30]
    assert all("ok" not in item for item in summary["recent_execution"])
    assert action["risks"] == [
        {
            "type": "recent_action_failure",
            "value": {"tool": "uav.turn_relative", "args": {"yaw_deg": 99}},
            "source": "DefaultActionContextModule",
        }
    ]


def test_context_builder_action_history_hides_legacy_move_direction(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    recent_args = [
        item["args"]
        for item in action["content"]["motion_history_summary"]["recent_execution"]
    ]
    assert recent_args == [
        {"duration": 0.3, "legacy_named_motion": True},
        {"duration": 0.3, "legacy_named_motion": True},
    ]
    assert "direction" not in json.dumps(action["content"], ensure_ascii=False)


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
    assert registry.semantic_evidence_capabilities() == [
        "Current image-derived text: visual facts from the latest RGB when processing succeeds; not raw image evidence."
    ]


def test_context_builder_includes_phase_execution_summary(tmp_path):
    state = make_state()
    state.record_observation({"pose": {"x": 0.0, "y": 0.0, "z": -2.0, "velocity": {"x": 9.0}}})
    state.ensure_current_phase_progress({"pose": {"x": 0.0, "y": 0.0, "z": -2.0, "velocity": {"x": 9.0}}})
    state.update_current_phase_progress(
        {"pose": {"x": 1.25, "y": 0.0, "z": -2.0, "velocity": {"x": 8.0}}},
        last_window_measurements={"planar_displacement_m": 1.25},
    )
    state.update_current_phase_progress(
        {"pose": {"x": 1.5, "y": 0.0, "z": -2.0, "velocity": {"x": 7.0}}}
    )

    context = ObservationContextBuilder().build(
        state,
        make_memory(tmp_path, state, actions=[{"tool": "uav.hover", "args": {}, "result": {"ok": True}}]),
    )

    assert context["phase_execution_summary"]["phase"] == "orient_leftward"
    assert context["phase_execution_summary"]["phase_key"] == "1:orient_leftward"
    assert context["phase_execution_summary"] == state.to_dict()["current_phase_progress"]
    assert context["phase_execution_summary"]["measurements"]["planar_m"] == 1.5
    assert context["phase_execution_summary"]["last_window"]["planar_m"] == 1.25
    assert "latest_window_measurements" not in context["phase_execution_summary"]
    assert "start_pose" not in context["phase_execution_summary"]
    assert "latest_pose" not in context["phase_execution_summary"]
    assert "phase_local_frame" not in context["phase_execution_summary"]
    assert "world_pose" not in context["phase_execution_summary"]
    assert "velocity" not in state.to_dict()["latest_observation"]["pose"]
    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    summary = action["content"]["motion_history_summary"]
    assert summary["phase_start_relative_uav_motion"]["planar_m"] == 1.5
    assert summary["last_window_uav_motion"] == {
        "forward_m": 0.0,
        "right_m": 0.0,
        "up_m": 0.0,
        "planar_m": 1.25,
        "yaw_delta_deg": 0.0,
    }
    assert "max_window_count" not in summary
    assert "tools_executed" not in summary


def test_context_builder_preserves_measured_last_window_across_planning_refresh(tmp_path):
    plan = SemanticPlan(
        intent="inspect_view",
        task_goal=TaskGoal(target="current view", final_state="inspection complete"),
        phases=[
            SemanticPhase(
                index=1,
                total=1,
                name="inspect_view",
                goal="inspect the current view",
                done_when=["the current view is inspected"],
                continue_when=["the current view is not yet inspected"],
            )
        ],
    )
    state = MissionState("run_motion_refresh", "inspect the current view", plan)
    state.record_observation(
        {"pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "euler_rad": {"yaw": 0.0}}}
    )
    state.ensure_current_phase_progress(state.latest_observation)
    state.update_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": -0.8, "z": -2.0}, "euler_rad": {"yaw": 0.0}}},
        last_window_measurements={
            "forward_displacement_m": 0.0,
            "right_displacement_m": -0.8,
            "up_displacement_m": 0.0,
            "planar_displacement_m": 0.8,
            "vertical_delta_m": 0.0,
            "yaw_delta_deg": 0.0,
        },
    )
    state.update_current_phase_progress(
        {"pose": {"position": {"x": 0.0, "y": -0.8, "z": -2.0}, "euler_rad": {"yaw": 0.0}}}
    )

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    action = next(fragment for fragment in context["fragments"] if fragment["name"] == "action")

    assert action["content"]["motion_history_summary"]["last_window_uav_motion"] == {
        "forward_m": 0.0,
        "right_m": -0.8,
        "up_m": 0.0,
        "planar_m": 0.8,
        "yaw_delta_deg": 0.0,
    }


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
        max_yaw_scan_extent_deg=480.0,
    )

    context = ObservationContextBuilder(action_config=action_config, safety_config=safety_config).build(
        state,
        make_memory(tmp_path, state),
    )

    constraints = context["allowed_action_constraints"]
    turn_schema = constraints["action_schemas"]["uav.turn_relative"]
    move_schema = constraints["action_schemas"]["uav.move_body"]
    turn_limits = turn_schema["limits"]
    move_limits = move_schema["limits"]
    assert turn_limits == {"yaw_deg_min": -25.0, "yaw_deg_max": 25.0}
    assert turn_schema["argument_semantics"] == {
        "yaw_deg": "Signed relative yaw in degrees: negative turns left; positive turns right."
    }
    assert move_limits == {
        "abs_vx_vy_max": 6.0,
        "abs_vz_max": 3.0,
        "duration_max": 8.0,
    }
    assert move_schema["argument_semantics"] == {
        "vx": "Positive moves forward; negative moves backward.",
        "vy": "Positive moves right; negative moves left.",
        "vz": "Positive moves up; negative moves down.",
        "duration": "Positive duration in seconds.",
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
    state.yaw_scan_offset_deg = 120.0
    state.yaw_scan_extent_deg = 450.0
    state.yaw_scan_turn_count = 7
    state.yaw_scan_recharge_count = 2
    safety_config = UavSafetyConfig(
        max_yaw_scan_extent_deg=480.0,
        yaw_scan_recharge_distance_m=0.5,
    )

    context = ObservationContextBuilder(safety_config=safety_config).build(state, make_memory(tmp_path, state))

    safety_constraints = context["safety_constraints"]
    assert safety_constraints["yaw_scan_offset_deg"] == 120.0
    assert safety_constraints["yaw_scan_extent_deg"] == 450.0
    assert safety_constraints["yaw_scan_remaining_deg"] == 30.0
    assert safety_constraints["max_yaw_scan_extent_deg"] == 480.0
    assert safety_constraints["yaw_scan_budget_exceeded"] is False
    assert safety_constraints["yaw_scan_turn_count"] == 7
    assert safety_constraints["yaw_scan_recharge_count"] == 2
    assert safety_constraints["yaw_scan_recharge_distance_m"] == 0.5

    action = {
        fragment["name"]: fragment for fragment in context["fragments"]
    }["action"]
    assert action["metadata"]["visual_search_progress"] == {
        "scope": "current_stationary_viewpoint",
        "yaw_offset_deg": 120.0,
        "yaw_extent_deg": 450.0,
        "yaw_remaining_deg": 30.0,
        "turn_count": 7,
        "recharge_count": 2,
        "recharge_distance_m": 0.5,
    }


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
