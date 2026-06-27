"""Manual real-model probe for the UAV-Agent plugin system MVP.

This script is intentionally not a pytest test. It exercises the live plugin
selection and context injection path with the current .env model settings, then
writes JSON artifacts under runs/ for manual inspection.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow import UavFlowBenchmarkRunner, UavFlowEvalConfig
from src.agent.mission_state import MissionState
from src.agent.semantic.schemas import SemanticPhase, SemanticPlan, TaskGoal
from src.memory.context.modules.skills import DefaultSkillsContextModule
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader, RunMemoryStore
from src.perception.context_builder import ObservationContextBuilder
from src.plugins import PluginRegistry


DEFAULT_INSTRUCTIONS = [
    "Turn to the direction of the person.",
    "Move to a position 3.5 meters from the person.",
    "Approach the person and keep a safe distance.",
]
DEFAULT_CASES_PATH = REPO_ROOT / "tests" / "fixtures" / "uav_knowledge_skill_selection_cases.json"

PROBE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAn0lEQVR4nO3XwQ2AIBAEQYz9d7aR"
    "I0hBSgMDY3pPZ1Xk2mE4kzQAAAAAAAAA4G7fBvB9lH3qfPfrkPzQ5qV2cO8fO7yC5hY2nT1h"
    "0oQ8yCn0Y2aQY8rF2CStS5N2nFqM0mZVtW5H2m8b0o7a9Y5Q2Kxv0wAAAAAAAAAAAPgGLs4E"
    "nK1l8GkAAAAASUVORK5CYII="
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-model plugin system probes.")
    parser.add_argument("--output-root", default="runs", help="Directory where probe artifacts are written.")
    parser.add_argument("--eval-id", default="uav_flow_eval_20260605_plugin_mvp_real_smoke")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="JSON skill-selection case fixture.")
    parser.add_argument("--run-uav-flow", action="store_true", help="Also run a small real UnrealCV UAV-Flow smoke.")
    args = parser.parse_args()

    run_dir = _create_probe_dir(Path(args.output_root))
    rgb_path = _write_probe_rgb(run_dir)
    print(f"plugin_system_probe_dir={run_dir}")

    selector_results = _run_real_skill_selection(run_dir, Path(args.cases))
    context_result = _run_context_injection(run_dir, rgb_path)
    research_result = _run_research_image_plugin(run_dir, rgb_path)
    uav_flow_result = _run_uav_flow_smoke(args.eval_id) if args.run_uav_flow else {"status": "skipped"}

    summary = {
        "run_dir": str(run_dir),
        "selector": selector_results,
        "context": context_result,
        "research_multimodal_image": research_result,
        "uav_flow": uav_flow_result,
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(_compact_summary(summary), ensure_ascii=False, indent=2))


def _create_probe_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"plugin_system_real_probe_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_probe_rgb(run_dir: Path) -> Path:
    rgb_dir = run_dir / "artifacts" / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    path = rgb_dir / "probe_rgb.png"
    path.write_bytes(base64.b64decode(PROBE_PNG_BASE64))
    return path


def _run_real_skill_selection(run_dir: Path, cases_path: Path) -> dict[str, Any]:
    registry = PluginRegistry.from_config()
    module = DefaultSkillsContextModule(plugin_registry=registry)
    cases = _load_selection_cases(cases_path)
    results = []
    for case in cases:
        instruction = case["instruction"]
        plan = _semantic_plan_for_instruction(instruction)
        memory = FlightMemory(
            "plugin_probe",
            "probe_map",
            "probe_scene",
            "probe_uav",
            active_instruction=instruction,
        )
        fragment = module.build(MemorySourceBundle(memory), _stage_context(instruction, plan)).to_dict()
        selected = [item["name"] for item in fragment["content"]["selected_skills"]]
        expected_any = list(case.get("expected_any", []))
        forbidden = list(case.get("forbidden", []))
        max_selected = int(case.get("max_selected", 99))
        skills_metadata = fragment.get("metadata", {})
        result = {
            "instruction": instruction,
            "expected_any": expected_any,
            "forbidden": forbidden,
            "max_selected": max_selected,
            "available_skill_count": len(skills_metadata.get("available_skill_names", [])),
            "selection": skills_metadata.get("selection", {}),
            "selected_skill_names": selected,
            "expected_hit": not expected_any or bool(set(selected) & set(expected_any)),
            "forbidden_hits": sorted(set(selected) & set(forbidden)),
            "over_selected": len(selected) > max_selected,
            "warnings": fragment["warnings"],
        }
        results.append(result)
    metrics = _selection_metrics(results)
    payload = {"cases_path": str(cases_path), "metrics": metrics, "results": results}
    _write_json(run_dir / "real_skill_selection.json", payload)
    (run_dir / "real_skill_selection.md").write_text(_selection_markdown(payload), encoding="utf-8")
    return payload


def _run_context_injection(run_dir: Path, rgb_path: Path) -> dict[str, Any]:
    instruction = "Approach the person and keep a safe distance."
    plan = _semantic_plan_for_instruction(instruction)
    memory, state = _memory_and_state_with_observation(run_dir, instruction, plan, rgb_path)
    builder = ObservationContextBuilder(
        enable_model_skill_selection=True,
        map_memory=MapMemoryStore(),
        run_memory=RunMemoryStore(),
    )
    context = builder.build(state, memory)
    context_path = run_dir / "contexts" / "latest_context.json"
    _write_json(context_path, context)
    skills = _fragment(context, "skills")
    image = _fragment(context, "image")
    result = {
        "context_path": str(context_path),
        "fragment_names": [fragment["name"] for fragment in context["fragments"]],
        "skills_selected_skill_names": [
            item["name"] for item in skills.get("content", {}).get("selected_skills", [])
        ],
        "skills_selection": skills.get("metadata", {}).get("selection", {}),
        "skills_facts_count": len(skills.get("facts", [])),
        "image_recent_rgb_count": len(image.get("content", {}).get("recent_rgb_frame_metadata", [])),
        "pose_fragment_present": _fragment(context, "pose") is not None,
        "action_fragment_present": _fragment(context, "action") is not None,
    }
    _write_json(run_dir / "context_injection_summary.json", result)
    return result


def _run_research_image_plugin(run_dir: Path, rgb_path: Path) -> dict[str, Any]:
    data = {
        "plugins": {
            "roots": ["plugins/knowledge", "plugins/research"],
            "enabled": [
                "knowledge.uav_flight",
                "research.multimodal_image",
            ],
        },
        "context_modules": {
            "instruction": "default",
            "skills": "default",
            "image": "research.multimodal_image",
            "pose": "default",
            "action": "default",
            "map": "default",
        },
    }
    plugin_registry = PluginRegistry.discover(data)
    registry = ContextModuleRegistry.from_dict(
        data,
        plugin_registry=plugin_registry,
        enable_model_skill_selection=False,
    )
    instruction = "Approach the person and keep a safe distance."
    plan = _semantic_plan_for_instruction(instruction)
    memory, state = _memory_and_state_with_observation(run_dir / "research_image", instruction, plan, rgb_path)
    builder = ObservationContextBuilder(
        registry=registry,
        plugin_registry=plugin_registry,
        enable_model_skill_selection=False,
        map_memory=MapMemoryStore(),
        run_memory=RunMemoryStore(),
    )
    context = builder.build(state, memory)
    context_path = run_dir / "research_multimodal_image_context.json"
    _write_json(context_path, context)
    image = _fragment(context, "image")
    result = {
        "context_path": str(context_path),
        "image_source": image.get("source"),
        "visual_processing": image.get("content", {}).get("visual_processing", {}),
        "has_visual_description": "visual_description" in image.get("content", {}),
        "warnings": image.get("warnings", []),
    }
    _write_json(run_dir / "research_multimodal_image_summary.json", result)
    return result


def _run_uav_flow_smoke(eval_id: str) -> dict[str, Any]:
    config = UavFlowEvalConfig(
        real_unrealcv=True,
        reuse_real_unrealcv_env=True,
        classes=["Turn", "Approach"],
        limit_per_class=2,
        enable_sr_judge=True,
        enable_visual_diagnostics=True,
        runtime_loop_guard_windows=100,
        parser_mode="model",
        metadata={"probe": "plugin_system_mvp"},
    )
    result = UavFlowBenchmarkRunner(config).run(eval_id=eval_id)
    return {
        "status": "ran",
        "eval_dir": result.get("eval_dir"),
        "case_count": len(result.get("results", [])),
        "summary_paths": result.get("summary_paths", {}),
    }


def _memory_and_state_with_observation(
    run_dir: Path,
    instruction: str,
    plan: SemanticPlan,
    rgb_path: Path,
) -> tuple[FlightMemory, MissionState]:
    writer = MemoryEventWriter(run_dir)
    reader = MemoryRecordReader(run_dir)
    memory = FlightMemory(
        "plugin_probe",
        "probe_map",
        "probe_scene",
        "probe_uav",
        memory_reader=reader,
        active_instruction=instruction,
    )
    instruction_ref = writer.write_instruction(instruction, plan.to_dict())
    memory.update_instruction_ref(instruction_ref, instruction)
    action_write = writer.write_action(
        "uav.turn_relative",
        {"yaw_deg": 5.0},
        {"ok": True, "action": "turn_relative", "message": "ok", "data": {"pose": {"yaw_deg": 5.0}}},
    )
    memory.append_action_ref(action_write["action_ref"])
    memory.append_pose_ref(action_write["pose_ref"])
    observation = {
        "source": "plugin_system_real_probe",
        "scene_instance_id": "probe_scene",
        "agent_id": "probe_uav",
        "modalities": ["rgb", "pose"],
        "pose": {"x": 0.0, "y": 0.0, "z": -2.0, "yaw_deg": 5.0},
        "frames": {"rgb": {"path": str(rgb_path), "mock": False}},
    }
    observation_write = writer.write_observation(observation)
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_pose_ref(observation_write["pose_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])
    state = MissionState("plugin_probe", instruction, plan)
    state.record_observation(observation)
    state.ensure_current_phase_progress(observation)
    return memory, state


def _semantic_plan_for_instruction(instruction: str) -> SemanticPlan:
    lower = instruction.lower()
    if "turn" in lower:
        intent = "turn_to_visual_target"
        final_state = "UAV faces the person."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="rotate_to_face_person",
            goal="Locate the person visually and rotate until the person is centered enough to face.",
            done_when=["person is identifiable", "view is roughly aligned with the person"],
            observe=["latest_observation", "phase_execution_summary"],
            continue_when=["person identity is unclear", "view is not plausibly aligned"],
            metrics=[],
        )
    elif "move" in lower:
        intent = "move_to_standoff_distance"
        final_state = "UAV reaches a 3.5 meter standoff distance from the person."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="locate_and_move_to_person_standoff",
            goal="Locate the person and move to the requested distance.",
            done_when=["person is identifiable", "standoff distance is plausibly near 3.5 meters"],
            observe=["latest_observation", "phase_start_reference", "phase_execution_summary"],
            continue_when=["person identity is unclear", "standoff distance is not yet plausible"],
            metrics=[],
        )
    else:
        intent = "approach_visual_target"
        final_state = "UAV approaches the person while keeping safe distance."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="approach_person_safely",
            goal="Confirm the person visually and approach conservatively.",
            done_when=["person is identifiable", "approach remains safe"],
            observe=["latest_observation", "phase_start_reference", "phase_execution_summary"],
            continue_when=["person identity is unclear", "safe approach relation is unclear"],
            metrics=[],
        )
    return SemanticPlan(
        intent=intent,
        task_goal=TaskGoal(
            target="person",
            target_attributes={"category": "human"},
            final_state=final_state,
        ),
        phases=[phase],
        uncertainties=[],
    )


def _stage_context(instruction: str, plan: SemanticPlan) -> StageContextInput:
    return StageContextInput(
        active_instruction=instruction,
        semantic_plan=plan.to_dict(),
        current_phase=plan.phases[0].to_dict(),
        semantic_goal=plan.task_goal.to_dict(),
    )


def _fragment(context: dict[str, Any], name: str) -> dict[str, Any]:
    for fragment in context.get("fragments", []):
        if fragment.get("name") == name:
            return fragment
    return {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_selection_cases(cases_path: Path) -> list[dict[str, Any]]:
    if cases_path.exists():
        data = json.loads(cases_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict) and str(item.get("instruction", "")).strip()]
    return [{"instruction": instruction, "expected_any": [], "forbidden": [], "max_selected": 3} for instruction in DEFAULT_INSTRUCTIONS]


def _selection_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    invalid = sum(1 for item in results if item["selection"].get("rejected_skill_names"))
    expected_hits = sum(1 for item in results if item["expected_hit"])
    forbidden_cases = sum(1 for item in results if item["forbidden_hits"])
    over_selected = sum(1 for item in results if item["over_selected"])
    all_selected = sum(1 for item in results if len(item["selected_skill_names"]) >= item["available_skill_count"] > 0)
    none_selected = sum(1 for item in results if not item["selected_skill_names"])
    selected_counts = [len(item["selected_skill_names"]) for item in results]
    return {
        "total_cases": total,
        "expected_hit_rate": expected_hits / total if total else 0.0,
        "invalid_selection_cases": invalid,
        "forbidden_hit_cases": forbidden_cases,
        "over_selected_cases": over_selected,
        "all_selected_rate": all_selected / total if total else 0.0,
        "none_selected_rate": none_selected / total if total else 0.0,
        "average_selected_count": sum(selected_counts) / total if total else 0.0,
    }


def _selection_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Real Skill Selection Probe",
        "",
        f"Cases: `{payload['cases_path']}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(payload["metrics"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Cases",
        "",
        "| instruction | selected | expected_hit | forbidden_hits | over_selected |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["instruction"]).replace("|", "\\|"),
                    ", ".join(item["selected_skill_names"]) or "(none)",
                    str(item["expected_hit"]),
                    ", ".join(item["forbidden_hits"]) or "-",
                    str(item["over_selected"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": summary["run_dir"],
        "selector_metrics": summary["selector"]["metrics"],
        "selector_statuses": [
            {
                "instruction": item["instruction"],
                "status": item["selection"].get("status"),
                "selected": item["selected_skill_names"],
                "warnings": item["warnings"],
            }
            for item in summary["selector"]["results"][:5]
        ],
        "context": summary["context"],
        "research_multimodal_image": summary["research_multimodal_image"],
        "uav_flow": summary["uav_flow"],
    }


if __name__ == "__main__":
    main()
