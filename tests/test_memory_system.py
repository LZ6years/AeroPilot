"""Test FlyAgent memory system MVP behavior."""

from __future__ import annotations

import pytest

from src.agent.task_parser import TaskParser
from src.memory.context_composer import ContextComposer
from src.memory.flight_memory import FlightMemory
from src.memory.registry import ContextModuleConfigError, ContextModuleRegistry
from src.memory.schemas import MemorySourceBundle
from src.memory.stores import MapMemoryStore


def make_memory() -> FlightMemory:
    result = make_parse_result()
    memory = FlightMemory("run_1", "m1", "scene_1", "uav_1")
    memory.update_instruction(result.instruction, result.plan.to_dict())
    memory.append_action("uav.hover", {}, {"ok": True})
    memory.append_observation(
        {
            "scene_instance_id": "scene_1",
            "agent_id": "uav_1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 1.0, "y": 0.0, "z": -2.0},
            "frames": {"rgb": {"path": "rgb.png"}, "depth": {"path": "depth.npy"}},
            "source": "phase_step",
        }
    )
    return memory


def make_parse_result():
    return TaskParser(mode="rule_based").parse("右转寻找黑色路灯并停下")


def test_context_composer_outputs_uniform_fragments():
    result = make_parse_result()
    memory = make_memory()
    context = ContextComposer().compose(
        MemorySourceBundle(memory, MapMemoryStore()),
        {
            "current_phase": result.plan.phases[0].to_dict(),
            "semantic_goal": result.plan.goal.to_dict(),
            "semantic_plan": result.plan.to_dict(),
        },
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


def test_registry_can_disable_modules():
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
        MemorySourceBundle(make_memory(), MapMemoryStore()),
        {
            "current_phase": result.plan.phases[0].to_dict(),
            "semantic_goal": result.plan.goal.to_dict(),
            "semantic_plan": result.plan.to_dict(),
        },
    )

    assert [fragment["name"] for fragment in context["fragments"]] == ["instruction", "pose", "action"]
    assert context["disabled_fragments"] == ["image", "map"]


def test_registry_loads_custom_module(tmp_path, monkeypatch):
    package = tmp_path / "custom_memory"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "paper_x.py").write_text(
        "\n".join(
            [
                "from src.memory.modules.base import ContextModule",
                "from src.memory.schemas import BudgetSpec, ContextFragment",
                "",
                "class PaperXImageMemoryModule(ContextModule):",
                "    name = 'image'",
                "    def build(self, sources, task_context):",
                "        return ContextFragment(",
                "            name='image', enabled=True, source='PaperXImageMemoryModule',",
                "            content={'paper': 'x'}, confidence='medium',",
                "            budget=BudgetSpec(priority=81, tokens_estimate=123)",
                "        )",
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

    fragment = registry.modules["image"].build(MemorySourceBundle(make_memory()), {})

    assert fragment.source == "PaperXImageMemoryModule"
    assert fragment.content == {"paper": "x"}


def test_registry_rejects_unknown_custom_module():
    with pytest.raises(ContextModuleConfigError, match="Unknown custom context module"):
        ContextModuleRegistry.from_dict({"modules": {"image": "missing"}, "custom_modules": {}})


def test_flight_memory_reset_pose_epoch_clears_flight_recent_memory():
    memory = make_memory()

    memory.reset_pose_epoch()

    assert memory.reset_epoch == 1
    assert memory.recent_actions == []
    assert memory.recent_observations == []
    assert memory.instruction_history
