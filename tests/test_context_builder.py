"""Test model-ready observation context building."""

from __future__ import annotations

import base64
from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image

from src.agent.planning.model_inputs import (
    build_model_user_content,
    sanitize_context_for_model_text,
)
from src.agent.mission_state import MissionState
from plugins.research.multimodal_image.context_modules import MultimodalImageContextModule
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.modules.image import (
    DefaultImageContextModule,
    _rgb_depth_overlay_data_url,
)
from src.memory.context.modules.action import DefaultActionContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.modules.pose import DefaultPoseContextModule
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
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


def test_model_text_keeps_skill_guidance_without_duplicating_callable_schema():
    context = {
        "fragments": [
            {
                "name": "skills",
                "content": {
                    "selected_skills": [
                        {
                            "name": "metric_depth_sample_region",
                            "description": "Measure depth.",
                            "skill_text": "Use a tight interior ROI.",
                            "capability": {"call_name": "skill.metric_depth.sample_region"},
                        }
                    ],
                    "callable_capabilities": [
                        {"call_name": "skill.metric_depth.sample_region"}
                    ],
                },
            }
        ]
    }

    sanitized = sanitize_context_for_model_text(context)

    content = sanitized["fragments"][0]["content"]
    assert content["selected_skills"] == [
        {
            "name": "metric_depth_sample_region",
            "description": "Measure depth.",
            "skill_text": "Use a tight interior ROI.",
        }
    ]
    assert "callable_capabilities" not in content


def decode_data_url_image(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")



def make_state() -> MissionState:
    return MissionState("run_test", "左转寻找建筑入口并停在入口前")



def make_memory(tmp_path, state: MissionState, actions=None, observations=None) -> FlightMemory:
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_test", "m1", "scene_1", "uav_1", active_instruction=state.user_instruction, memory_reader=reader)
    memory.update_instruction_ref(writer.write_instruction(state.user_instruction, {}), state.user_instruction)
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

    assert capabilities[:2] == [
        "One chronological RGB observation; transcript projection labels it HISTORY or CURRENT.",
        "Metric-depth fusion: qualitative range colors only; explicit meter claims require a grounded-candidate distance result.",
    ]
    assert capabilities[2].startswith("UAV ground clearance")
    assert capabilities[3].startswith("Measured instruction-relative UAV self-motion")
    assert capabilities[4].startswith("Map memory:")
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

    assert any(item.startswith("One chronological RGB") for item in capabilities)
    assert not any("depth" in item.lower() for item in capabilities)


def test_context_builder_instruction_plan_observation_and_safety(tmp_path):
    from src.agent.planning.task_plan import TaskPlan
    state = make_state()
    state.protocol_context = "Caller protocol"
    state.task_plan = TaskPlan()
    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state))
    assert context["active_instruction"] == state.user_instruction
    assert context["protocol_context"] == "Caller protocol"
    assert context["task_plan"] == state.task_plan.to_dict()
    assert not {"current_phase", "phase_strategy", "phase_neighborhood", "semantic_goal"} & context.keys()
    assert context["safety_constraints"]["max_yaw_scan_extent_deg"] == 480.0
    assert "allowed_action_constraints" not in context
    assert context["observation_policy"]["sample_timing"] == "after_each_physical_window"
    fragments = {item["name"]: item for item in context["fragments"]}
    assert "instruction_history" not in fragments["instruction"]["content"]
    assert "loaded_knowledge" in fragments["skills"]["content"]



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
    assert "recent_execution" not in fragments["action"]["content"]
    assert "phase_strategy" not in fragments["action"]["content"]
    assert "current_phase" not in context



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
    assert "Original RGB" in image["content"]["selected_frames"][0]["usage"]
    assert "fused colors" not in image["content"]["selected_frames"][0]["usage"]
    assert image["content"]["selected_frames"][0]["observation_id"] == "obs_0001"
    assert image["content"]["selected_frames"][0]["frame_id"] == "frame_0001"
    assert image["metadata"]["selected_frame_metadata"][0]["path"] == str(rgb_path)
    assert "recent_rgb_frame_metadata" not in image["metadata"]
    assert image["model_inputs"][0]["type"] == "image"
    assert image["model_inputs"][0]["input_id"] == "image_1_current_observation"
    assert image["model_inputs"][0]["role"] == "current_observation"
    assert image["model_inputs"][0]["usage"] == image["content"]["selected_frames"][0]["usage"]
    assert image["model_inputs"][0]["encoding"] == "data_url"
    assert image["model_inputs"][0]["observation_id"] == "obs_0001"
    assert image["model_inputs"][0]["frame_id"] == "frame_0001"
    assert image["model_inputs"][0]["artifact_ref"] == str(rgb_path)
    assert "visual_processing" not in image["content"]
    assert "visual_description" not in image["content"]


def test_context_builder_default_image_attaches_only_latest_checkpoint_frame(tmp_path):
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
    assert roles == ["current_observation"]
    assert source_paths == [paths[5]]
    assert [item["input_id"] for item in image["content"]["selected_frames"]] == [
        "image_1_current_observation",
    ]
    assert [item["type"] for item in image["model_inputs"]] == ["image"]
    assert image["model_inputs"][0]["encoding"] == "data_url"
    assert all("metadata" not in item and "source" not in item for item in image["model_inputs"])
    model_content = build_model_user_content(context)
    assert isinstance(model_content, list)
    assert [part["type"] for part in model_content] == ["text", "image_url", "text"]
    assert "observation_id=obs_0006" in model_content[0]["text"]
    text_context = json.loads(model_content[2]["text"])
    assert not any(fragment.get("name") == "image" for fragment in text_context.get("fragments", []))
    assert "visual_evidence" not in model_content[2]["text"]
    assert "selected_frames" not in model_content[2]["text"]


def test_context_builder_image_contract_has_no_anchor_replay_option(tmp_path):
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
    modules["image"] = DefaultImageContextModule()
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
    assert [item["role"] for item in image["model_inputs"]] == ["current_observation"]
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
    assert [item["role"] for item in image["model_inputs"]] == ["current_rgb_depth_overlay"]
    assert image["content"]["visual_evidence"]["has_depth"] is True
    assert image["content"]["selected_frames"][-1]["role"] == "current_depth"
    assert image["metadata"]["current_depth_metadata"]["path"] == str(tmp_path / "depth_1.npy")
    overlay_image = decode_data_url_image(image["model_inputs"][0]["data"])
    assert overlay_image.width == 6
    assert overlay_image.height > 4
    assert overlay_image.height == 46
    assert image["model_inputs"][0]["scene_width"] == 6
    assert image["model_inputs"][0]["scene_height"] == 4
    assert image["model_inputs"][0]["display_width"] == 6
    assert image["model_inputs"][0]["display_height"] == 46
    assert image["model_inputs"][0]["color_bar_height"] == 42
    overlay_scene_pixel = overlay_image.getpixel((3, 2))
    assert overlay_scene_pixel != (21, 41, 61)

    model_content = build_model_user_content(context)
    assert isinstance(model_content, list)
    assert [part["type"] for part in model_content] == ["text", "image_url", "text"]
    assert "observation_id=obs_0002" in model_content[0]["text"]
    assert "RGB + METRIC DEPTH OVERLAY." in model_content[0]["text"]
    recovery_content = build_model_user_content(context, image_mode="current_only")
    assert isinstance(recovery_content, list)
    assert [part["type"] for part in recovery_content] == ["text", "image_url", "text"]
    assert "RGB + METRIC DEPTH OVERLAY." in recovery_content[0]["text"]


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
    assert first_image.getpixel((3, 2)) == second_image.getpixel((3, 2))
    assert first_image.getpixel((0, 0)) != second_image.getpixel((0, 0))


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
    assert [part["type"] for part in model_content] == ["text", "image_url", "text"]
    assert "observation_id=obs_0002" in model_content[0]["text"]
    combined_text = "\n".join(
        part["text"] for part in model_content if part["type"] == "text"
    )
    assert "PAST | PHASE START" not in combined_text
    assert "PAST is historical comparison evidence only" not in combined_text


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
    assert [item["role"] for item in image["model_inputs"]] == ["current_observation"]


def test_context_builder_default_image_ignores_phase_anchor_for_media(tmp_path):
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
    assert roles == ["current_observation"]
    assert source_paths == [paths[3]]


def test_context_builder_does_not_annotate_image_pixels(tmp_path):
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
    current = decode_data_url_image(image["model_inputs"][0]["data"])

    assert image["model_inputs"][0]["mime_type"] == "image/png"
    assert current.size == (320, 240)
    assert list(current.getdata()) == [(10, 40, 90)] * (320 * 240)
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

    with pytest.raises(OSError):
        ObservationContextBuilder().build(state, make_memory(tmp_path, state, observations=observations))


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
    assert "field_semantics" in text  # no pose: unknown measurements are not invented
    assert "not target distance" in text
    assert '"direction"' not in text


def test_context_builder_action_module_does_not_duplicate_transcript_history(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "result": {"ok": True}},
        {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    summary = action["content"]
    assert "recent_execution_order" not in summary
    assert action["metadata"]["visual_search_progress"] == {
        "scope": "current_stationary_viewpoint",
        "yaw_offset_deg": 0.0,
        "yaw_extent_deg": 0.0,
        "yaw_remaining_deg": 480.0,
        "turn_count": 0,
        "recharge_count": 0,
        "recharge_distance_m": 0.5,
    }
    assert "recent_execution" not in summary
    assert "action_trend" not in action["content"]
    assert "recent_actions" not in action["content"]


def test_context_builder_action_module_does_not_proxy_success_or_failure_records(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.turn_relative", "args": {"yaw_deg": value}, "result": {"ok": True}}
        for value in (5, 10, 15, 20, 25, 30)
    ]
    actions.insert(4, {"tool": "uav.turn_relative", "args": {"yaw_deg": 99}, "result": {"ok": False}})

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    summary = action["content"]
    assert "recent_execution" not in summary
    assert action["risks"] == []


def test_context_builder_action_module_omits_legacy_action_arguments(tmp_path):
    state = make_state()
    actions = [
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
        {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 0.3}, "result": {"ok": True}},
    ]

    context = ObservationContextBuilder().build(state, make_memory(tmp_path, state, actions=actions))

    action = {fragment["name"]: fragment for fragment in context["fragments"]}["action"]
    assert "recent_execution" not in action["content"]
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
        seen_phases.append((prompt_context.get("task_plan") or {}))
        return {
            "parsed": {
                "visual_description": {
                    "phase": (prompt_context.get("task_plan") or {}),
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
    assert seen_phases[0] == {}
    assert image["content"]["visual_processing"]["status"] == "captioned"
    assert image["content"]["visual_description"]["phase"] == {"name": "", "goal": ""}
    assert image["model_inputs"][0]["type"] == "text"
    assert image["model_inputs"][0]["role"] == "visual_caption"
    assert "data:image/png;base64," not in image["model_inputs"][0]["text"]
    assert registry.semantic_evidence_capabilities() == [
        "Current image-derived text: visual facts from the latest RGB when processing succeeds; not raw image evidence."
    ]




def test_context_builder_keeps_state_without_tool_declarations(tmp_path):
    state = make_state()
    context = ObservationContextBuilder(safety_config=UavSafetyConfig(max_yaw_step_deg=25)).build(
        state, make_memory(tmp_path, state))
    assert "allowed_action_constraints" not in context
    assert "allowed_action_constraints" not in context["context_schema"]["planner_context_fields"]
    assert context["safety_constraints"]["hover_required"] == state.hover_required


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
