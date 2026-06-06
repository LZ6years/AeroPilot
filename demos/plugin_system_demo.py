"""Plugin system demo: knowledge skills, optional callable skills, and planner context.

Run:
    python demos/plugin_system_demo.py
    python demos/plugin_system_demo.py --selector real
    python demos/plugin_system_demo.py --instruction "Turn to the direction of the person."

This demo does not start UE and does not execute UAV motion. It shows:
1. how plugin folders become enabled skills,
2. how selected skills enter the `skills` context fragment,
3. how builtin planner tools remain available,
4. how optional callable skill capabilities would appear if enabled.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.mission_state import MissionState
from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.planning.tool_contract_validator import ToolContractValidator
from src.agent.semantic.schemas import SemanticGoal, SemanticPhase, SemanticPlan, SemanticTarget
from src.memory.context.composer import ContextComposer
from src.memory.context.modules import (
    DefaultActionContextModule,
    DefaultImageContextModule,
    DefaultInstructionContextModule,
    DefaultMapContextModule,
    DefaultPoseContextModule,
    DefaultSkillsContextModule,
)
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader, RunMemoryStore
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry
from src.plugins.skills import ModelSkillSelector, SkillContentLoader
from src.plugins.skills.selection import SkillSelectionResult
from src.tools.dispatcher import ToolDispatcher


DEFAULT_INSTRUCTION = "Approach the person and keep a safe distance."
DEFAULT_SELECTED_SKILLS = [
    "search_under_uncertainty",
    "visual_target_task_reasoning",
    "approach_and_standoff",
]
PROBE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAn0lEQVR4nO3XwQ2AIBAEQYz9d7aR"
    "I0hBSgMDY3pPZ1Xk2mE4kzQAAAAAAAAA4G7fBvB9lH3qfPfrkPzQ5qV2cO8fO7yC5hY2nT1h"
    "0oQ8yCn0Y2aQY8rF2CStS5N2nFqM0mZVtW5H2m8b0o7a9Y5Q2Kxv0wAAAAAAAAAAAPgGLs4E"
    "nK1l8GkAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class DemoResult:
    run_dir: Path
    selector_mode: str
    registry: PluginRegistry
    selected_skill_names: list[str]
    selection: dict[str, Any]
    loaded_skill_summaries: list[dict[str, Any]]
    planner_context: dict[str, Any]
    capability_result: dict[str, Any]


class DemoSkillSelector:
    def __init__(self, selected_skill_names: list[str]) -> None:
        self.selected_skill_names = list(selected_skill_names)

    def select(self, selection_context: dict[str, Any], catalog) -> SkillSelectionResult:
        accepted, rejected = catalog.validate_selection(self.selected_skill_names)
        return SkillSelectionResult(
            requested_skill_names=list(self.selected_skill_names),
            accepted_skill_names=[skill.name for skill in accepted],
            rejected_skill_names=rejected,
            status="selected" if accepted else "none_selected",
            warnings=[f"skill_selection_rejected:{name}" for name in rejected],
        )


class DemoEnvClient:
    """Minimal placeholder; callable skill demo does not use UAV actions."""


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = _make_run_dir(Path(args.output_root))
    result = run_demo(run_dir, args.instruction, args.selector)
    write_demo_outputs(result)
    print(json.dumps(_console_summary(result), ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show UAV-Agent plugin knowledge skills and planner context.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--selector", choices=["demo", "real"], default="demo")
    return parser


def run_demo(run_dir: Path, instruction: str, selector_mode: str) -> DemoResult:
    rgb_path = _write_probe_rgb(run_dir)
    plan = _semantic_plan_for_instruction(instruction)
    plugin_registry = PluginRegistry.from_config()
    stage_context = _stage_context(instruction, plan, plugin_registry)
    memory, map_memory = _memory_with_observation(run_dir, instruction, plan, rgb_path)

    skills_module = DefaultSkillsContextModule(
        plugin_registry=plugin_registry,
        skill_selector=_skill_selector(selector_mode),
        content_loader=SkillContentLoader(max_skill_chars=900, max_reference_chars=900),
    )
    registry = ContextModuleRegistry(
        {
            "instruction": DefaultInstructionContextModule(),
            "skills": skills_module,
            "image": DefaultImageContextModule(),
            "pose": DefaultPoseContextModule(),
            "action": DefaultActionContextModule(),
            "map": DefaultMapContextModule(),
        },
        module_order=["instruction", "skills", "image", "pose", "action", "map"],
    )
    planner_context = ContextComposer(registry).compose(
        MemorySourceBundle(memory, map_memory, RunMemoryStore()),
        stage_context,
    )
    skills = _fragment(planner_context, "skills")
    selected_skill_names = [
        item["name"]
        for item in skills.get("content", {}).get("selected_skills", [])
        if item.get("name")
    ]
    capability_result = _run_capability_demo(plugin_registry, planner_context, rgb_path)
    return DemoResult(
        run_dir=run_dir,
        selector_mode=selector_mode,
        registry=plugin_registry,
        selected_skill_names=selected_skill_names,
        selection=dict(skills.get("content", {}).get("selection", {})),
        loaded_skill_summaries=list(skills.get("content", {}).get("selected_skills", [])),
        planner_context=planner_context,
        capability_result=capability_result,
    )


def _skill_selector(selector_mode: str):
    if selector_mode == "real":
        return ModelSkillSelector()
    return DemoSkillSelector(DEFAULT_SELECTED_SKILLS)


def _run_capability_demo(plugin_registry: PluginRegistry, context: dict[str, Any], rgb_path: Path) -> dict[str, Any]:
    tool_name = "skill.person_detector.detect"
    if tool_name not in plugin_registry.skill_capability_catalog.planner_tools():
        return {
            "status": "skipped",
            "reason": "skill capability is not enabled in configs/plugins.json",
            "registered_skill_tools": plugin_registry.skill_capability_catalog.planner_tools(),
        }
    args = {"image_source": "latest_rgb", "min_confidence": 0.5}
    plan = ShortHorizonPlan(
        "approach_person_safely",
        [PlannedStep.tool_step(tool_name, args, "confirm person visibility before moving")],
    )
    ToolContractValidator().validate(plan, "approach_person_safely", context)
    dispatcher = ToolDispatcher(DemoEnvClient(), plugin_registry=plugin_registry)  # type: ignore[arg-type]
    result = dispatcher.call(
        "demo_scene",
        "demo_uav",
        tool_name,
        args,
        runtime_context={
            "latest_observation": {
                "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
            }
        },
    )
    return {
        "planned_tool": tool_name,
        "args": args,
        "registered_skill_tools": plugin_registry.skill_capability_catalog.planner_tools(),
        "result": result.to_dict(),
    }


def _memory_with_observation(
    run_dir: Path,
    instruction: str,
    plan: SemanticPlan,
    rgb_path: Path,
) -> tuple[FlightMemory, MapMemoryStore]:
    writer = MemoryEventWriter(run_dir)
    reader = MemoryRecordReader(run_dir)
    memory = FlightMemory("plugin_demo", "demo_map", "demo_scene", "demo_uav", memory_reader=reader)
    memory.update_instruction_ref(writer.write_instruction(instruction, plan.to_dict()), instruction)
    action_write = writer.write_action("uav.hover", {}, {"ok": True, "action": "hover", "message": "demo stable hover"})
    memory.append_action_ref(action_write["action_ref"])
    observation_write = writer.write_observation(
        {
            "source": "plugin_system_demo",
            "scene_instance_id": "demo_scene",
            "agent_id": "demo_uav",
            "modalities": ["rgb", "pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": -2.0, "yaw_deg": 0.0},
            "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_pose_ref(observation_write["pose_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])
    map_memory = MapMemoryStore()
    map_memory.add(
        {
            "map_id": "demo_map",
            "scene_id": "demo_scene",
            "landmark_description": "The person may be near the open street area.",
            "requires_reconfirmation": True,
            "confidence": "low",
        }
    )
    return memory, map_memory


def write_demo_outputs(result: DemoResult) -> None:
    files = {
        "01_config_snapshot.json": _config_snapshot(result.registry),
        "02_discovered_skills.json": result.registry.skill_catalog.listings(),
        "03_selection_result.json": result.selection,
        "04_loaded_skill_summaries.json": result.loaded_skill_summaries,
        "05_latest_context.json": result.planner_context,
        "06_capability_call_result.json": result.capability_result,
    }
    for name, data in files.items():
        _write_json(result.run_dir / name, data)
    (result.run_dir / "plugin_system_walkthrough.md").write_text(build_walkthrough_markdown(result), encoding="utf-8")


def build_walkthrough_markdown(result: DemoResult) -> str:
    skills = _fragment(result.planner_context, "skills")
    constraints = result.planner_context.get("allowed_action_constraints", {})
    return "\n".join(
        [
            "# UAV-Agent Plugin System Demo",
            "",
            "This report shows the unified skill framework.",
            "",
            "## 1. Enabled Skills",
            "",
            "Skills come from `plugins/<category>/<plugin_name>/<skill_folder>/SKILL.md`.",
            "",
            "```json",
            json.dumps(result.registry.skill_catalog.listings(), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 2. Selected Skill Guidance",
            "",
            "`skills.facts` stays empty. Skill content is guidance, not observed truth.",
            "",
            "```json",
            json.dumps(
                {
                    "selected_skill_names": result.selected_skill_names,
                    "facts": skills.get("facts"),
                    "available_capabilities": skills.get("content", {}).get("available_capabilities"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## 3. Planner Tools",
            "",
            "Builtin tools remain builtin. Callable skills add `skill.*` tools only when enabled by plugin config.",
            "",
            "```json",
            json.dumps(
                {
                    "planner_tools": constraints.get("planner_tools"),
                    "skill_schema": constraints.get("action_schemas", {}).get("skill.person_detector.detect"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## 4. Optional Callable Skill Result",
            "",
            "The default config keeps capability plugins disabled, so this section normally reports `skipped`.",
            "",
            "```json",
            json.dumps(result.capability_result, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 5. Code Reading Order",
            "",
            "1. `src/plugins/contracts/skill.py`",
            "2. `src/plugins/skills/markdown.py`",
            "3. `src/plugins/core/discovery.py`",
            "4. `src/memory/context/modules/skills.py`",
            "5. `src/perception/context_builder.py`",
            "6. `src/agent/planning/tool_contract_validator.py`",
            "7. `src/tools/dispatcher.py`",
            "8. `configs/plugins.json`",
            "",
        ]
    )


def _config_snapshot(registry: PluginRegistry) -> dict[str, Any]:
    return {
        "enabled_plugins": list(registry.enabled_plugins),
        "skill_count": len(registry.skill_catalog.skills),
        "callable_skill_tools": registry.skill_capability_catalog.planner_tools(),
        "context_module_config": dict(registry.context_module_config),
        "warnings": list(registry.warnings),
    }


def _semantic_plan_for_instruction(instruction: str) -> SemanticPlan:
    return SemanticPlan(
        intent="approach_visual_target",
        goal=SemanticGoal(
            target=SemanticTarget(description="person", attributes={"category": "human"}),
            final_state="UAV approaches the person while keeping safe distance.",
        ),
        phases=[
            SemanticPhase(
                index=1,
                total=1,
                name="approach_person_safely",
                meaning="Confirm the person visually and approach conservatively.",
                completion_criteria=[{"type": "safe_visual_approach", "target": "person"}],
            )
        ],
        uncertainties=[],
    )


def _stage_context(instruction: str, plan: SemanticPlan, plugin_registry: PluginRegistry) -> StageContextInput:
    return StageContextInput(
        active_instruction=instruction,
        semantic_plan=plan.to_dict(),
        current_phase=plan.phases[0].to_dict(),
        semantic_goal=plan.goal.to_dict(),
        allowed_action_constraints=ObservationContextBuilder(plugin_registry=plugin_registry)._allowed_action_constraints(),
        metadata={"demo": "plugin_system_demo"},
    )


def _write_probe_rgb(run_dir: Path) -> Path:
    rgb_dir = run_dir / "artifacts" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    path = rgb_dir / "probe_rgb.png"
    path.write_bytes(base64.b64decode(PROBE_PNG_BASE64))
    return path


def _make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"plugin_system_demo_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _fragment(context: dict[str, Any], name: str) -> dict[str, Any]:
    for fragment in context.get("fragments", []):
        if fragment.get("name") == name:
            return fragment
    return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _console_summary(result: DemoResult) -> dict[str, Any]:
    return {
        "run_dir": str(result.run_dir),
        "read_first": str(result.run_dir / "plugin_system_walkthrough.md"),
        "selector_mode": result.selector_mode,
        "selected_skill_names": result.selected_skill_names,
        "planner_context": str(result.run_dir / "05_latest_context.json"),
        "capability_result": str(result.run_dir / "06_capability_call_result.json"),
    }


if __name__ == "__main__":
    main()
