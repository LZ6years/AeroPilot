"""Test FlyAgent memory system MVP behavior."""

from __future__ import annotations

import json

import pytest

from src.agent.messages import AgentMessage
from src.agent.semantic.task_parser import TaskParser
from src.memory.context.composer import ContextComposer
from src.memory.flight_memory import FlightMemory
from src.memory.context.modules.base import ContextModule
from src.memory.context.registry import ContextModuleConfigError, ContextModuleRegistry
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader


def make_parse_result():
    return TaskParser(mode="rule_based").parse("右转寻找黑色路灯并停下")


def make_stage_context(result=None, current_phase: dict | None = None) -> StageContextInput:
    result = result or make_parse_result()
    return StageContextInput(
        active_instruction=result.instruction,
        semantic_plan=result.plan.to_dict(),
        current_phase=current_phase if current_phase is not None else result.plan.phases[0].to_dict(),
        semantic_goal=result.plan.goal.to_dict(),
        safety_constraints={"hover_required": False},
        allowed_action_constraints={"planner_tools": ["uav.hover", "env.observe"]},
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
    assert data["semantic_goal"] == result.plan.goal.to_dict()
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
    assert context["semantic_goal"] == result.plan.goal.to_dict()
    assert [fragment["name"] for fragment in context["fragments"]] == [
        "instruction",
        "pose",
        "action",
        "image",
        "map",
    ]
    for fragment in context["fragments"]:
        assert {"facts", "inferences", "goals", "risks", "budget", "metadata"} <= set(fragment)
        assert "priority" in fragment["budget"]
    assert context["metadata"]["budget"]["max_tokens_estimate"] == 1200
    assert context["metadata"]["budget"]["final_tokens_estimate"] <= 1200
    assert "context_schema" in context
    assert "planner_context_fields" in context["context_schema"]
    assert "fragment_common_fields" in context["context_schema"]
    assert set(context["context_schema"]["module_content_fields"]) == {"instruction", "image", "pose", "action", "map"}
    assert "latest_pose" in context["context_schema"]["module_content_fields"]["pose"]["content_fields"]
    assert "recent_rgb_frame_metadata" in context["context_schema"]["module_content_fields"]["image"]["content_fields"]


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


def test_context_composer_trims_low_priority_list_content_when_over_budget(tmp_path):
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

    assert map_fragment["content"]["items"] == [{"index": 5}]
    assert map_fragment["facts"] == [{"index": 5}]
    assert map_fragment["inferences"] == [{"index": 5}]
    assert map_fragment["risks"] == [{"index": 5}]
    assert map_fragment["artifact_refs"] == ["artifact_5"]
    assert map_fragment["warnings"] == ["warning_5"]
    assert context["metadata"]["budget"]["budget_applied"] is True
    assert context["metadata"]["budget"]["trimmed_fragments"][0]["name"] == "map"


def test_action_module_reads_jsonl_refs_and_outputs_compact_recent_actions(tmp_path):
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
        make_stage_context(current_phase={"name": "stop"}),
    )
    action = _fragment(context, "action")

    assert len(action["content"]["recent_actions"]) == 5
    assert "data" not in action["content"]["recent_actions"][-1]["result"]
    assert "action_effect_summaries" not in action["content"]
    assert action["inferences"] == []


def test_pose_and_image_modules_read_jsonl_refs(tmp_path):
    memory = make_memory(tmp_path)

    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        make_stage_context(current_phase={"name": "align_to_landmark"}),
    )
    pose = _fragment(context, "pose")
    image = _fragment(context, "image")

    assert pose["content"]["latest_pose"] == {"x": 1.0, "y": 0.0, "z": -2.0}
    assert image["content"]["visual_evidence"]["has_rgb"] is True
    assert image["content"]["visual_evidence"]["has_depth"] is False
    assert image["content"]["latest_frame_metadata"]["rgb"]["path"] == "rgb.png"
    assert image["content"]["recent_rgb_frame_metadata"][0]["path"] == "rgb.png"
    assert image["artifact_refs"] == ["rgb.png"]


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

    assert [item["path"] for item in image["content"]["recent_rgb_frame_metadata"]] == [
        "rgb_1.png",
        "rgb_2.png",
        "rgb_3.png",
    ]
    assert image["artifact_refs"] == ["rgb_1.png", "rgb_2.png", "rgb_3.png"]
    assert all("depth" not in ref for ref in image["artifact_refs"])


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

    assert [item["yaw_deg"] for item in pose["content"]["recent_pose_history"]] == [10.0, 20.0, 30.0]
    assert "velocity" not in pose["content"]["latest_pose"]
    assert all("velocity" not in item for item in pose["content"]["recent_pose_history"])
    assert "velocity" not in pose["facts"][0]["value"]


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
            "modules": {
                "instruction": "default",
                "image": False,
                "pose": "default",
                "action": "default",
                "map": False,
            },
            "custom_modules": {},
        }
    )
    result = make_parse_result()
    context = ContextComposer(registry).compose(
        MemorySourceBundle(make_memory(tmp_path), MapMemoryStore()),
        make_stage_context(result),
    )

    assert [fragment["name"] for fragment in context["fragments"]] == ["instruction", "pose", "action"]
    assert context["disabled_fragments"] == ["image", "map"]
    assert set(context["context_schema"]["module_content_fields"]) == {"instruction", "pose", "action"}
    assert "image" not in context["context_schema"]["module_content_fields"]
    assert "map" not in context["context_schema"]["module_content_fields"]


def test_default_modules_describe_their_outputs():
    registry = ContextModuleRegistry.default()

    descriptions = registry.describe_modules()

    assert set(descriptions) == {"instruction", "image", "pose", "action", "map"}
    assert "active_instruction" in descriptions["instruction"]["content_fields"]
    assert "latest_frame_metadata" in descriptions["image"]["content_fields"]
    assert "latest_pose" in descriptions["pose"]["content_fields"]
    assert "recent_actions" in descriptions["action"]["content_fields"]
    assert "recalled_map_memories" in descriptions["map"]["content_fields"]


def test_registry_loads_custom_module(tmp_path, monkeypatch):
    package = tmp_path / "custom_memory"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "paper_x.py").write_text(
        "\n".join(
            [
                "from src.memory.context.modules.base import ContextModule",
                "from src.memory.context.schemas import BudgetSpec, ContextFragment",
                "",
                "class PaperXImageMemoryModule(ContextModule):",
                "    name = 'image'",
                "    def build(self, sources, stage_context):",
                "        return ContextFragment(",
                "            name='image', enabled=True, source='PaperXImageMemoryModule',",
                "            content={'paper': 'x', 'phase': stage_context.current_phase}, confidence='medium',",
                "            budget=BudgetSpec(priority=81, tokens_estimate=123)",
                "        )",
                "    def describe_output(self):",
                "        return {'summary': 'Paper X image memory.', 'content_fields': {'paper': 'Paper method label', 'phase': 'Current phase'}}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ContextModuleRegistry.from_dict(
        {
            "modules": {"image": "paper_x_image_memory"},
            "custom_modules": {
                "paper_x_image_memory": {
                    "module_path": "custom_memory.paper_x",
                    "class_name": "PaperXImageMemoryModule",
                }
            },
        }
    )

    fragment = registry.modules["image"].build(
        MemorySourceBundle(make_memory(tmp_path)),
        make_stage_context(current_phase={"name": "custom_phase"}),
    )

    assert fragment.source == "PaperXImageMemoryModule"
    assert fragment.content == {"paper": "x", "phase": {"name": "custom_phase"}}
    assert registry.describe_modules()["image"]["content_fields"]["paper"] == "Paper method label"


def test_registry_rejects_custom_module_without_output_description(tmp_path, monkeypatch):
    package = tmp_path / "custom_memory_missing_description"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "paper_x.py").write_text(
        "\n".join(
            [
                "from src.memory.context.modules.base import ContextModule",
                "from src.memory.context.schemas import ContextFragment",
                "",
                "class PaperXImageMemoryModule(ContextModule):",
                "    name = 'image'",
                "    def build(self, sources, stage_context):",
                "        return ContextFragment(name='image', enabled=True, source='PaperXImageMemoryModule')",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ContextModuleConfigError, match="must implement describe_output"):
        ContextModuleRegistry.from_dict(
            {
                "modules": {"image": "paper_x_image_memory"},
                "custom_modules": {
                    "paper_x_image_memory": {
                        "module_path": "custom_memory_missing_description.paper_x",
                        "class_name": "PaperXImageMemoryModule",
                    }
                },
            }
        )


def test_registry_rejects_unknown_custom_module():
    with pytest.raises(ContextModuleConfigError, match="Unknown custom context module"):
        ContextModuleRegistry.from_dict({"modules": {"image": "missing"}, "custom_modules": {}})


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
