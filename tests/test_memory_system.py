"""Test FlyAgent memory system MVP behavior."""

from __future__ import annotations

import base64
import json

import pytest
from PIL import Image

from src.agent.messages import AgentMessage
from src.agent.semantic.task_parser import TaskParser
from src.memory.context.composer import ContextComposer
from plugins.research.multimodal_image.context_modules import MultimodalImageContextModule
from src.memory.flight_memory import FlightMemory
from src.memory.context.modules.base import ContextModule
from src.memory.context.registry import ContextModuleConfigError, ContextModuleRegistry
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader
from src.plugins import PluginRegistry
from src.runtime.configs.model_config import ModelConfig


def write_test_png(path, color=(12, 34, 56), size=(320, 240)) -> None:
    Image.new("RGB", size, color).save(path, format="PNG")


def make_parse_result():
    return TaskParser(mode="rule_based").parse("右转寻找黑色路灯并停下")


def make_stage_context(
    result=None,
    current_phase: dict | None = None,
    phase_neighborhood: dict | None = None,
    phase_execution_summary: dict | None = None,
) -> StageContextInput:
    result = result or make_parse_result()
    return StageContextInput(
        active_instruction=result.instruction,
        semantic_plan=result.plan.to_dict(),
        current_phase=current_phase if current_phase is not None else result.plan.phases[0].to_dict(),
        phase_neighborhood=phase_neighborhood or {},
        semantic_goal=result.plan.task_goal.to_dict(),
        safety_constraints={"hover_required": False},
        allowed_action_constraints={"planner_tools": ["uav.hover", "env.observe"]},
        phase_execution_summary=phase_execution_summary or {},
        metadata={"test": True},
    )


def make_memory(tmp_path) -> FlightMemory:
    result = make_parse_result()
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    instruction_ref = writer.write_instruction("向前飞行5米", {"intent": "move"})
    memory.update_instruction_ref(instruction_ref, result.instruction)
    action_write = writer.write_action("uav.hover", {}, {"ok": True, "action": "hover", "message": "ok"})
    memory.append_action_ref(action_write["action_ref"])
    observation_write = writer.write_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 1.0, "y": 0.0, "z": -2.0},
            "frames": {"rgb": {"path": "rgb.png"}, "depth": {"path": "depth.npy"}},
            "source": "phase_step",
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_pose_ref(observation_write["pose_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])
    return memory


def test_memory_event_writer_and_reader_round_trip(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    instruction_ref = writer.write_instruction("向前飞行5米", {"intent": "move"})
    action_write = writer.write_action(
        "uav.move_body",
        {"direction": "forward"},
        {"ok": True, "data": {"pose": {"x": 5.0, "y": 0.0, "z": -2.0}}},
    )
    observation_write = writer.write_observation(
        {
            "source": "phase_step",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 5.0, "y": 0.0, "z": -2.0},
            "frames": {"rgb": {"path": "rgb.png", "mock": True}, "depth": {"path": "depth.npy"}},
        }
    )

    assert reader.read_ref(instruction_ref)["instruction"] == "向前飞行5米"
    assert reader.read_ref(action_write["action_ref"])["result"]["data"]["pose"]["x"] == 5.0
    assert reader.read_ref(action_write["pose_ref"])["source"] == "action_result"
    observation = reader.read_ref(observation_write["observation_ref"])
    assert observation["pose_ref"]["id"].startswith("pose_")
    assert len(observation["frame_refs"]) == 2


def test_memory_event_writer_preserves_pose_coordinate_frame(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)

    explicit_ref = writer.write_pose(
        {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "coordinate_frame": "unrealcv_world_m",
        },
        "observation",
    )
    generic_ref = writer.write_pose(
        {"position": {"x": 4.0, "y": 5.0, "z": 6.0}},
        "observation",
    )

    assert reader.read_ref(explicit_ref)["coordinate_frame"] == "unrealcv_world_m"
    assert (
        reader.read_ref(generic_ref)["coordinate_frame"]
        == "canonical_unspecified"
    )


def test_memory_event_writer_records_agent_messages_as_refs(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    message = AgentMessage("final", "任务完成，我已保持悬停。", confidence="high")

    message_ref = writer.write_agent_message("phase_judge", "hover", message.to_dict())
    memory.append_agent_message_ref(message_ref)

    record = reader.read_ref(message_ref)
    data = memory.to_dict()

    assert record["id"] == "msg_0001"
    assert record["source"] == "phase_judge"
    assert record["phase"] == "hover"
    assert record["message"]["type"] == "final"
    assert data["recent_agent_message_refs"] == [message_ref]
    assert "任务完成" not in json.dumps(data, ensure_ascii=False)


def test_stage_context_input_to_dict_has_expected_fields():
    result = make_parse_result()
    stage_context = make_stage_context(result)

    data = stage_context.to_dict()

    assert data["active_instruction"] == result.instruction
    assert data["semantic_plan"] == result.plan.to_dict()
    assert data["current_phase"] == result.plan.phases[0].to_dict()
    assert data["phase_neighborhood"] == {}
    assert data["semantic_goal"] == result.plan.task_goal.to_dict()
    assert data["safety_constraints"] == {"hover_required": False}
    assert data["allowed_action_constraints"]["planner_tools"] == ["uav.hover", "env.observe"]
    assert data["metadata"] == {"test": True}


def test_context_composer_outputs_uniform_fragments(tmp_path):
    result = make_parse_result()
    memory = make_memory(tmp_path)
    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(result),
    )

    assert context["active_instruction"] == memory.active_instruction
    assert context["current_phase"]["name"]
    assert context["phase_neighborhood"] == {}
    assert context["semantic_goal"] == result.plan.task_goal.to_dict()
    assert [fragment["name"] for fragment in context["fragments"]] == [
        "instruction",
        "skills",
        "pose",
        "action",
        "image",
        "map",
    ]
    for fragment in context["fragments"]:
        assert {"facts", "inferences", "goals", "risks", "budget", "metadata"} <= set(fragment)
        assert "priority" in fragment["budget"]
    assert context["metadata"]["budget"]["max_tokens_estimate"] == 2000
    assert context["metadata"]["budget"]["final_tokens_estimate"] >= 0
    assert context["metadata"]["budget"]["budget_applied"] is False
    assert context["metadata"]["budget"]["trimmed_fragments"] == []
    assert "context_schema" in context
    assert "planner_context_fields" in context["context_schema"]
    assert "fragment_fields" in context["context_schema"]
    assert set(context["context_schema"]["module_content_fields"]) == {
        "instruction",
        "skills",
        "image",
        "pose",
        "action",
        "map",
    }
    assert "selected_skills" in context["context_schema"]["module_content_fields"]["skills"]
    assert "agl_m" in context["context_schema"]["module_content_fields"]["pose"]
    assert "latest_pose" not in context["context_schema"]["module_content_fields"]["pose"]
    assert "selected_frames" in context["context_schema"]["module_content_fields"]["image"]


def test_context_composer_loads_rules_and_user_preferences_from_markdown(tmp_path):
    rules_path = tmp_path / "FLYAGENT.md"
    user_path = tmp_path / "USER.md"
    rules_path.write_text("Always prefer safe hover.", encoding="utf-8")
    user_path.write_text("User prefers concise action reasons.", encoding="utf-8")

    context = ContextComposer(rules_path=rules_path, user_preferences_path=user_path).compose(
        MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()),
        make_stage_context(),
    )

    assert context["rules"] == {"text": "Always prefer safe hover.", "source": str(rules_path)}
    assert context["user_preferences"] == {"text": "User prefers concise action reasons.", "source": str(user_path)}
    assert "flight_memory" not in context
    assert "recent_actions" not in context


def test_context_composer_loads_default_memory_rules_and_preferences(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "FLYAGENT.md").write_text("Default flyagent rule.", encoding="utf-8")
    (memory_dir / "USER.md").write_text("Default user preference.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    flight_memory = make_memory(tmp_path)
    context = ContextComposer().compose(
        MemorySourceBundle(flight_memory, MapMemoryStore()),
        make_stage_context(),
    )
    flight_memory_json = flight_memory.to_dict()

    assert context["rules"]["text"] == "Default flyagent rule."
    assert context["rules"]["source"] == "memory\\FLYAGENT.md" or context["rules"]["source"] == "memory/FLYAGENT.md"
    assert context["user_preferences"]["text"] == "Default user preference."
    assert "Default flyagent rule." not in json.dumps(flight_memory_json, ensure_ascii=False)
    assert "Default user preference." not in json.dumps(flight_memory_json, ensure_ascii=False)


def test_context_composer_allows_explicit_rules_to_override_file_values(tmp_path):
    rules_path = tmp_path / "FLYAGENT.md"
    rules_path.write_text("file rules", encoding="utf-8")

    context = ContextComposer(
        rules={"text": "runtime rules", "mode": "test"},
        user_preferences={"text": "runtime preferences"},
        rules_path=rules_path,
        user_preferences_path=tmp_path / "missing_USER.md",
    ).compose(MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()), make_stage_context())

    assert context["rules"]["text"] == "runtime rules"
    assert context["rules"]["mode"] == "test"
    assert context["rules"]["source"] == str(rules_path)
    assert context["user_preferences"] == {"text": "runtime preferences"}


def test_context_composer_preserves_fragment_content_when_over_budget(tmp_path):
    class LargeLowPriorityModule(ContextModule):
        name = "map"

        def build(self, sources, stage_context):
            return ContextFragment(
                name="map",
                enabled=True,
                source="LargeLowPriorityModule",
                content={"items": [{"index": index} for index in range(6)]},
                facts=[{"index": index} for index in range(6)],
                inferences=[{"index": index} for index in range(6)],
                risks=[{"index": index} for index in range(6)],
                artifact_refs=[f"artifact_{index}" for index in range(6)],
                warnings=[f"warning_{index}" for index in range(6)],
                budget=BudgetSpec(priority=1, tokens_estimate=1000, max_items=6),
            )

        def describe_output(self):
            return {"summary": "Large map test module.", "content_fields": {"items": "Synthetic list items."}}

    registry = ContextModuleRegistry({"map": LargeLowPriorityModule()})

    context = ContextComposer(registry, max_tokens_estimate=500).compose(
        MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()),
        make_stage_context(),
    )
    map_fragment = _fragment(context, "map")

    assert map_fragment["content"]["items"] == [{"index": index} for index in range(6)]
    assert map_fragment["facts"] == [{"index": index} for index in range(6)]
    assert map_fragment["inferences"] == [{"index": index} for index in range(6)]
    assert map_fragment["risks"] == [{"index": index} for index in range(6)]
    assert map_fragment["artifact_refs"] == [f"artifact_{index}" for index in range(6)]
    assert map_fragment["warnings"] == [f"warning_{index}" for index in range(6)]
    assert context["metadata"]["budget"]["original_tokens_estimate"] == 1000
    assert context["metadata"]["budget"]["final_tokens_estimate"] == 1000
    assert context["metadata"]["budget"]["budget_applied"] is False
    assert context["metadata"]["budget"]["trimmed_fragments"] == []


def test_context_composer_preserves_all_selected_skills_when_over_budget(tmp_path):
    class LargeSkillsModule(ContextModule):
        name = "skills"

        def build(self, sources, stage_context):
            return ContextFragment(
                name="skills",
                enabled=True,
                source="LargeSkillsModule",
                content={
                    "selected_skills": [
                        {"name": "vertical_landing_and_altitude_control", "skill_text": "landing guidance"},
                        {"name": "approach_and_standoff", "skill_text": "approach guidance"},
                    ]
                },
                budget=BudgetSpec(priority=95, tokens_estimate=2400, max_items=8),
            )

        def describe_output(self):
            return {
                "summary": "Synthetic skills test module.",
                "content_fields": {"selected_skills": "Synthetic selected skills."},
            }

    registry = ContextModuleRegistry({"skills": LargeSkillsModule()})

    context = ContextComposer(registry, max_tokens_estimate=500).compose(
        MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()),
        make_stage_context(),
    )
    skills_fragment = _fragment(context, "skills")

    assert [item["name"] for item in skills_fragment["content"]["selected_skills"]] == [
        "vertical_landing_and_altitude_control",
        "approach_and_standoff",
    ]
    assert context["metadata"]["budget"]["original_tokens_estimate"] == 2400
    assert context["metadata"]["budget"]["final_tokens_estimate"] == 2400
    assert context["metadata"]["budget"]["budget_applied"] is False
    assert context["metadata"]["budget"]["trimmed_fragments"] == []


def test_action_module_reads_jsonl_refs_and_outputs_unified_motion_history(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", active_instruction="鍙宠浆瀵绘壘榛戣壊璺伅", memory_reader=reader)
    for index in range(6):
        write_result = writer.write_action(
            "uav.turn_relative",
            {"yaw_deg": 10.0},
            {"ok": True, "action": "turn_relative", "message": "ok", "data": {"pose": {"yaw_deg": index * 10.0}}},
        )
        memory.append_action_ref(write_result["action_ref"])
        memory.append_pose_ref(write_result["pose_ref"])
    write_result = writer.write_action(
        "uav.hover",
        {},
        {"ok": True, "action": "hover", "message": "ok", "data": {"pose": {"yaw_deg": 60.0}}},
    )
    memory.append_action_ref(write_result["action_ref"])
    memory.append_pose_ref(write_result["pose_ref"])

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(
            current_phase={"name": "stop"},
            phase_execution_summary={
                "phase": "stop",
                "window_count": 7,
                "measurements": {"yaw_delta_deg": 60.0},
                "tool_facts": {
                    "turn_yaw_success_sum_deg": 60.0,
                    "tools_executed": {"uav.turn_relative": 6},
                },
            },
        ),
    )
    action = _fragment(context, "action")

    summary = action["content"]["motion_history_summary"]
    assert "window_count" not in summary
    assert summary["phase_start_relative_uav_motion"] == {
        "forward_m": 0.0,
        "right_m": 0.0,
        "up_m": 0.0,
        "planar_m": 0.0,
        "yaw_delta_deg": 60.0,
    }
    assert action["metadata"]["visual_search_progress"] == {
        "scope": "current_stationary_viewpoint",
        "yaw_offset_deg": 0.0,
        "yaw_extent_deg": 0.0,
        "yaw_remaining_deg": 0.0,
        "turn_count": 0,
        "recharge_count": 0,
        "recharge_distance_m": 0.0,
    }
    assert "recent_execution_order" not in summary
    assert "recent_execution" not in summary
    assert "motion_history_summary_sources" in action["metadata"]
    assert "recent_actions" not in action["content"]
    assert "action_trend" not in action["content"]
    assert "action_effect_summaries" not in action["content"]
    assert action["facts"] == []
    assert action["inferences"] == []


def test_pose_and_image_modules_read_jsonl_refs(tmp_path):
    memory = make_memory(tmp_path)

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "align_to_landmark"}),
    )
    pose = _fragment(context, "pose")
    image = _fragment(context, "image")

    assert pose["content"] == {}
    assert pose["metadata"]["latest_pose"] == {"x": 1.0, "y": 0.0, "z": -2.0}
    assert "recent_pose_history" not in pose["content"]
    assert image["content"]["visual_evidence"]["has_rgb"] is True
    assert image["content"]["visual_evidence"]["has_depth"] is False
    assert image["content"]["selected_frames"][0]["role"] == "current_observation"
    assert image["metadata"]["selected_frame_metadata"][0]["path"] == "rgb.png"
    assert image["artifact_refs"] == ["rgb.png"]
    assert image["model_inputs"] == []
    assert image["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"


def test_default_image_module_attaches_latest_real_rgb_as_model_input(tmp_path):
    rgb_path = tmp_path / "frame.png"
    write_test_png(rgb_path)
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    observation_write = writer.write_observation(
        {
            "source": "automatic_phase_window",
            "modalities": ["rgb", "pose"],
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "search_visual_target"}),
    )
    image = _fragment(context, "image")
    model_input = image["model_inputs"][0]

    assert model_input["type"] == "image"
    assert model_input["input_id"] == "image_1_current_observation"
    assert model_input["role"] == "current_observation"
    assert "Original RGB" in model_input["usage"]
    assert "fused colors" not in model_input["usage"]
    assert model_input["encoding"] == "data_url"
    assert model_input["data"].startswith("data:image/png;base64,")
    assert model_input["data"] == "data:image/png;base64," + base64.b64encode(rgb_path.read_bytes()).decode("ascii")
    assert model_input["observation_id"] == "obs_0001"
    assert model_input["frame_id"] == "frame_0001"
    assert model_input["artifact_ref"] == str(rgb_path)
    assert image["content"]["selected_frames"][0]["input_id"] == model_input["input_id"]
    assert image["content"]["selected_frames"][0]["role"] == "current_observation"
    assert image["metadata"]["selected_frame_metadata"][0]["path"] == str(rgb_path)
    assert image["content"]["visual_evidence"]["interpretation_status"] == "image_attached_not_interpreted"
    assert image["content"]["visual_evidence"]["attached_image_count"] == 1
    assert "visual_description" not in image["content"]


def test_image_module_keeps_three_recent_rgb_frames_only(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    for index in range(4):
        observation_write = writer.write_observation(
            {
                "source": f"phase_step_{index}",
                "modalities": ["rgb", "depth", "pose"],
                "pose": {"x": float(index), "y": 0.0, "z": -2.0},
                "frames": {
                    "rgb": {"path": f"rgb_{index}.png"},
                    "depth": {"path": f"depth_{index}.npy"},
                },
            }
        )
        memory.append_observation_ref(observation_write["observation_ref"])
        memory.append_pose_ref(observation_write["pose_ref"])
        memory.append_frame_refs(observation_write["frame_refs"])

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "search_landmark"}),
    )
    image = _fragment(context, "image")

    assert [item["path"] for item in image["metadata"]["recent_rgb_frame_metadata"]] == [
        "rgb_1.png",
        "rgb_2.png",
        "rgb_3.png",
    ]
    assert image["artifact_refs"] == [
        "rgb_1.png",
        "rgb_2.png",
        "rgb_3.png",
    ]
    assert all("depth" not in ref for ref in image["artifact_refs"])


def test_image_module_ignores_temporal_anchors_for_checkpoint_media(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    observation_refs = []
    for index in range(3):
        rgb_path = tmp_path / f"temporal_{index}.png"
        write_test_png(rgb_path)
        written = writer.write_observation(
            {
                "source": f"window_{index}",
                "modalities": ["rgb"],
                "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
            }
        )
        observation_refs.append(written["observation_ref"])
        memory.append_observation_ref(written["observation_ref"])
        memory.append_frame_refs(written["frame_refs"])

    memory.set_observation_anchor("phase_start", observation_refs[0])
    memory.set_observation_anchor("before_last_action_window", observation_refs[1])
    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "navigate"}),
    )
    image = _fragment(context, "image")
    selected = image["content"]["selected_frames"]

    assert [item["role"] for item in selected] == ["current_observation"]
    assert [item["observation_id"] for item in selected] == ["obs_0003"]
    assert [
        item["role"] for item in image["model_inputs"] if item.get("type") == "image"
    ] == ["current_observation"]

    memory.set_observation_anchor("before_last_action_window", observation_refs[0])
    merged = _fragment(
        ContextComposer().compose(
            MemorySourceBundle(memory, MapMemoryStore()),
            make_stage_context(current_phase={"name": "navigate"}),
        ),
        "image",
    )
    assert [item["role"] for item in merged["content"]["selected_frames"]] == [
        "current_observation"
    ]


def test_multimodal_image_module_captions_latest_real_rgb(tmp_path):
    rgb_path = tmp_path / "frame.png"
    rgb_path.write_bytes(b"fake-png")
    seen_payloads = []

    def caption_client(config, payload):
        seen_payloads.append(payload)
        user_content = payload["messages"][1]["content"]
        prompt_context = json.loads(user_content[0]["text"])
        assert prompt_context["current_phase"]["name"] == "search_landmark"
        assert user_content[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
        return {
            "parsed": {
                "visual_description": {
                    "phase": {
                        "name": "search_landmark",
                        "goal": "identify the entrance candidate",
                        "done_when": ["entrance candidate is visible"],
                        "continue_when": ["entrance identity is unclear"],
                    },
                    "phase_relevance": "可见建筑底层入口候选。",
                    "scene_summary": "建筑底层入口候选位于画面中景。",
                    "spatial_overview": "midground-center: 建筑底层开口。",
                    "relevant_objects": [
                        {
                            "name": "建筑底层开口",
                            "position": "midground-center",
                            "role": "target_candidate",
                            "phase_relevance": "与当前入口目标同类。",
                            "description": "矩形开口位于建筑底层。",
                            "confidence": "high",
                        }
                    ],
                    "free_space": ["前景铺装区域开阔"],
                    "landmarks": ["背景高层建筑"],
                    "phase_relevant_cues": ["建筑底层存在矩形开口"],
                    "uncertainty": ["未见明确入口标识"],
                    "confidence": "medium",
                }
            },
            "duration_sec": 0.25,
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        }

    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    observation_write = writer.write_observation(
        {
            "source": "automatic_phase_window",
            "modalities": ["rgb", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": -2.0},
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_pose_ref(observation_write["pose_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])

    fragment = MultimodalImageContextModule(
        model_config=ModelConfig(model_name="qwen3-vl-flash", api_key="secret-key", base_url="https://example.test/v1"),
        caption_client=caption_client,
    ).build(
        MemorySourceBundle(memory),
        make_stage_context(
            current_phase={
                "name": "search_landmark",
                "goal": "identify the entrance candidate",
                "done_when": ["entrance candidate is visible"],
                "continue_when": ["entrance identity is unclear"],
            }
        ),
    ).to_dict()

    model_input = fragment["model_inputs"][0]
    assert seen_payloads
    assert model_input["type"] == "text"
    assert model_input["role"] == "visual_caption"
    assert model_input["metadata"]["source_path"] == str(rgb_path)
    assert model_input["metadata"]["strategy"] == "qwen_vl_phase_caption_v3"
    assert "data:image/png;base64," not in model_input["text"]
    assert fragment["content"]["visual_description"]["phase"]["name"] == "search_landmark"
    assert fragment["content"]["visual_processing"]["strategy"] == "qwen_vl_phase_caption_v3"
    assert fragment["content"]["visual_processing"]["status"] == "captioned"
    assert fragment["content"]["visual_evidence"]["interpretation_status"] == "captioned"
    assert fragment["metadata"]["selected_frame_metadata"][0]["path"] == str(rgb_path)
    assert "recent_rgb_frame_metadata" not in fragment["metadata"]
    assert fragment["artifact_refs"] == [str(rgb_path)]


def test_multimodal_image_module_falls_back_for_mock_or_missing_rgb(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    observation_write = writer.write_observation(
        {
            "source": "automatic_phase_window",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(tmp_path / "missing.png"), "mock": True}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])

    fragment = MultimodalImageContextModule().build(
        MemorySourceBundle(memory),
        make_stage_context(current_phase={"name": "search_landmark"}),
    ).to_dict()

    assert fragment["model_inputs"] == []
    assert fragment["content"]["visual_processing"]["strategy"] == "phase_aware_caption"
    assert fragment["content"]["visual_processing"]["status"] == "metadata_only_not_interpreted"
    assert fragment["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_not_interpreted"


def test_multimodal_image_module_falls_back_when_caption_unavailable(tmp_path):
    rgb_path = tmp_path / "frame.png"
    rgb_path.write_bytes(b"fake-png")
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    observation_write = writer.write_observation(
        {
            "source": "automatic_phase_window",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])

    fragment = MultimodalImageContextModule(
        model_config=ModelConfig(model_name="qwen3-vl-flash", api_key="", base_url="https://example.test/v1"),
        caption_client=lambda *_: pytest.fail("caption client should not be called without an API key"),
    ).build(
        MemorySourceBundle(memory),
        make_stage_context(current_phase={"name": "search_landmark"}),
    ).to_dict()

    assert fragment["model_inputs"] == []
    assert "visual_description" not in fragment["content"]
    assert fragment["content"]["visual_processing"]["status"] == "metadata_only_caption_skipped"
    assert fragment["content"]["visual_evidence"]["interpretation_status"] == "metadata_only_caption_skipped"
    assert "visual_caption_skipped:no_model_api_key" in fragment["warnings"]


def test_pose_module_keeps_three_recent_pose_refs(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    for yaw in [0.0, 10.0, 20.0, 30.0]:
        memory.append_pose_ref(
            writer.write_pose(
                {"x": 0.0, "y": 0.0, "z": -2.0, "yaw_deg": yaw, "velocity": {"x": 9.0}},
                "test",
            )
        )

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "stop"}),
    )
    pose = _fragment(context, "pose")

    assert pose["metadata"]["pose_history_count"] == 3
    assert pose["metadata"]["recent_pose_delta"]["dyaw_deg"] == 10.0
    assert "velocity" not in pose["metadata"]["latest_pose"]
    assert pose["content"] == {}
    assert pose["facts"] == []


def test_pose_module_exposes_only_optional_agl_to_model_content(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    memory.append_pose_ref(
        writer.write_pose(
            {
                "x": 10.0,
                "y": -4.0,
                "z": 2.5,
                "yaw_deg": -170.0,
                "agl_m": 2.15,
                "coordinate_frame": "unrealcv_world_m",
            },
            "observation",
        )
    )

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "land"}),
    )
    pose = _fragment(context, "pose")

    assert pose["content"]["agl_m"] == 2.15
    assert "target distance" in pose["content"]["field_semantics"]["agl_m"]
    assert pose["metadata"]["latest_pose"]["yaw_deg"] == -170.0
    assert set(pose["content"]) == {"agl_m", "field_semantics"}


def test_flight_memory_json_is_manifest_not_raw_history(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    instruction_ref = writer.write_instruction("向前飞行5米", {"intent": "move"})
    action_write = writer.write_action(
        "uav.hover",
        {},
        {"ok": True, "action": "hover", "message": "ok", "data": {"pose": {"x": 0.0}}},
    )
    observation_write = writer.write_observation({"source": "phase_step", "modalities": ["pose"], "pose": {"x": 0.0}})
    memory.update_instruction_ref(instruction_ref, "向前飞行5米")
    memory.append_action_ref(action_write["action_ref"])
    memory.append_pose_ref(action_write["pose_ref"])
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_pose_ref(observation_write["pose_ref"])

    data = memory.to_dict()

    assert data["instruction_history"][0]["ref"] == "memory/instructions.jsonl#1"
    assert data["recent_action_refs"][0]["ref"] == "memory/actions.jsonl#1"
    assert data["recent_observation_refs"][0]["ref"] == "memory/observations.jsonl#1"
    assert "recent_actions" not in data
    assert "recent_observations" not in data
    assert "latest_pose_ref" not in data
    assert "pose_ref: same_as_previous_observation" not in json.dumps(data)
    assert reader.read_ref(data["recent_action_refs"][0])["result"]["data"]["pose"] == {"x": 0.0}


def test_flight_memory_keeps_long_ref_window_for_modules(tmp_path):
    writer = MemoryEventWriter(tmp_path)
    reader = MemoryRecordReader(tmp_path)
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1", memory_reader=reader)
    for index in range(60):
        memory.update_instruction_ref(writer.write_instruction(f"instruction {index}"), f"instruction {index}")
        action_write = writer.write_action("uav.hover", {}, {"ok": True, "action": "hover", "message": str(index)})
        memory.append_action_ref(action_write["action_ref"])
        observation_write = writer.write_observation(
            {
                "source": "phase_step",
                "modalities": ["pose"],
                "pose": {"x": float(index), "y": 0.0, "z": -2.0},
            }
        )
        memory.append_observation_ref(observation_write["observation_ref"])
        memory.append_pose_ref(observation_write["pose_ref"])

    data = memory.to_dict()

    assert len(memory.instruction_history) == 50
    assert len(memory.recent_action_refs) == 50
    assert len(memory.recent_observation_refs) == 50
    assert len(data["recent_action_refs"]) == 50
    assert len(data["recent_observation_refs"]) == 50
    assert reader.read_ref(data["instruction_history"][0])["instruction"] == "instruction 10"


def test_registry_can_disable_modules(tmp_path):
    registry = ContextModuleRegistry.from_dict(
        {
            "context_modules": {
                "instruction": "default",
                "skills": False,
                "image": False,
                "pose": "default",
                "action": "default",
                "map": False,
            }
        }
    )
    result = make_parse_result()
    context = ContextComposer(registry).compose(
        MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()),
        make_stage_context(result),
    )

    assert [fragment["name"] for fragment in context["fragments"]] == ["instruction", "pose", "action"]
    assert context["disabled_fragments"] == ["skills", "image", "map"]
    assert set(context["context_schema"]["module_content_fields"]) == {"instruction", "pose", "action"}
    assert "image" not in context["context_schema"]["module_content_fields"]
    assert "map" not in context["context_schema"]["module_content_fields"]


def test_default_modules_describe_their_outputs():
    registry = ContextModuleRegistry.default()

    descriptions = registry.describe_modules()

    assert set(descriptions) == {"instruction", "skills", "image", "pose", "action", "map"}
    assert "current_instruction" in descriptions["instruction"]["content_fields"]
    assert "selected_skills" in descriptions["skills"]["content_fields"]
    assert "selected_frames" in descriptions["image"]["content_fields"]
    assert "agl_m" in descriptions["pose"]["content_fields"]
    assert "latest_pose" in descriptions["pose"]["metadata_fields"]
    assert "latest_pose" not in descriptions["pose"]["content_fields"]
    assert "motion_history_summary" in descriptions["action"]["content_fields"]
    assert "field_semantics" in descriptions["action"]["content_fields"]
    assert "recent_actions" not in descriptions["action"]["content_fields"]
    assert "action_trend" not in descriptions["action"]["content_fields"]
    assert "motion_history_summary_sources" in descriptions["action"]["metadata_fields"]
    assert "visual_search_progress" in descriptions["action"]["metadata_fields"]
    assert "recalled_map_memories" in descriptions["map"]["content_fields"]


def test_registry_loads_plugin_context_module():
    plugin_registry = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/research"],
                "enabled": ["research.multimodal_image"],
            }
        }
    )

    registry = ContextModuleRegistry.from_dict(
        {"context_modules": {"image": "research.multimodal_image"}},
        plugin_registry=plugin_registry,
    )

    assert registry.modules["image"].__class__.__name__ == "PluginContextModule"
    assert registry.describe_modules()["image"]["content_fields"]["visual_processing"]


def test_registry_rejects_unknown_plugin_context_module():
    with pytest.raises(ContextModuleConfigError, match="Unknown plugin context module"):
        ContextModuleRegistry.from_dict({"context_modules": {"image": "missing.plugin"}})


def test_map_memory_store_manifest_recall_and_map_filter(tmp_path):
    store = MapMemoryStore()
    store.add(
        {
            "map_id": "m1",
            "scene_id": "scene_1",
            "landmark_description": "black street lamp near road",
            "pose_hint": {"x": 1.0, "y": 2.0, "z": -2.0},
            "confidence": "medium",
        }
    )
    store.add({"map_id": "m2", "landmark_description": "red vehicle"})
    path = store.save(tmp_path / "map_memory.json")

    loaded = MapMemoryStore.from_file(path)
    recalled = loaded.recall("m1", "scene_1")
    other_map = loaded.recall("m3")

    assert len(recalled) == 1
    assert recalled[0]["coordinate_frame"] == "airsim_ned"
    assert recalled[0]["requires_reconfirmation"] is True
    assert other_map == []


def test_map_context_module_outputs_recall_as_inference_and_risk():
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1")
    store = MapMemoryStore()
    store.add({"map_id": "m1", "scene_id": "scene_1", "landmark_description": "building entrance"})

    context = ContextComposer().compose(
        MemorySourceBundle(memory, store),
        make_stage_context(current_phase={"name": "search_landmark"}),
    )
    map_fragment = _fragment(context, "map")

    assert map_fragment["facts"] == []
    assert map_fragment["inferences"][0]["type"] == "map_memory_hint"
    assert map_fragment["risks"][0]["type"] == "map_memory_requires_reconfirmation"


def test_flight_memory_reset_pose_epoch_clears_flight_recent_refs(tmp_path):
    memory = make_memory(tmp_path)

    memory.reset_pose_epoch()

    assert memory.reset_epoch == 1
    assert memory.recent_action_refs == []
    assert memory.recent_observation_refs == []
    assert memory.recent_pose_refs == []
    assert memory.recent_frame_refs == []
    assert memory.instruction_history


def _fragment(context: dict, name: str) -> dict:
    return next(fragment for fragment in context["fragments"] if fragment["name"] == name)
