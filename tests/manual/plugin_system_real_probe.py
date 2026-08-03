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
        plan = _semantic_plan_for_case(case)
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
        expected_references = _expected_reference_map(case.get("expected_references", {}))
        forbidden_references = _expected_reference_map(case.get("forbidden_references", {}))
        max_selected = int(case.get("max_selected", 99))
        skills_metadata = fragment.get("metadata", {})
        accepted_references = skills_metadata.get("selection", {}).get("accepted_references", {})
        missing_references = _missing_expected_references(expected_references, accepted_references)
        forbidden_reference_hits = _forbidden_reference_hits(forbidden_references, accepted_references)
        result = {
            "instruction": instruction,
            "expected_any": expected_any,
            "forbidden": forbidden,
            "expected_references": expected_references,
            "forbidden_references": forbidden_references,
            "max_selected": max_selected,
            "available_skill_count": len(skills_metadata.get("available_skill_names", [])),
            "selection": skills_metadata.get("selection", {}),
            "selected_skill_names": selected,
            "semantic_context": {
                "intent": plan.intent,
                "target": plan.task_goal.target,
                "final_state": plan.task_goal.final_state,
                "current_phase": plan.phases[0].to_dict(),
            },
            "expected_hit": not expected_any or bool(set(selected) & set(expected_any)),
            "expected_reference_hit": not missing_references,
            "missing_expected_references": missing_references,
            "forbidden_reference_hits": forbidden_reference_hits,
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
        execution_mode="attach",
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
    if _has_any(lower, ["do not finish", "decide whether", "confirm whether", "complete", "sufficient"]):
        intent = "completion_evidence_check"
        target = _target_from_instruction(lower)
        final_state = "The UAV has enough evidence to decide whether the current objective is complete."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="assess_current_objective_evidence",
            goal="Judge whether current visual evidence and recent progress are sufficient for the requested objective.",
            done_when=["current evidence supports completion or clearly shows more evidence is needed"],
            continue_when=["evidence is stale, ambiguous, or missing for the requested objective"],
            metrics=[],
        )
        attributes = {"evidence_role": "completion_check"}
    elif _has_any(lower, ["land", "landing"]):
        intent = "vertical_landing"
        target = _target_from_instruction(lower)
        final_state = "UAV reaches a controlled landing or low-altitude hold near the requested region."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="controlled_landing_near_requested_region",
            goal="Confirm the requested landing region or relation, then make controlled altitude progress toward landing.",
            done_when=[
                "landing region or requested relation is plausible",
                "altitude or surface-related progress supports landing or low-altitude hold",
            ],
            continue_when=[
                "landing region is unclear",
                "altitude or surface progress is insufficient",
                "continuing descent would be unsafe",
            ],
            metrics=_vertical_metric_from_instruction(lower, default_drop_m=4.0),
        )
        attributes = {"relation": _relation_from_instruction(lower)}
    elif _has_any(lower, ["descend", "ascend", "altitude"]):
        intent = "altitude_control"
        target = "requested altitude"
        final_state = "UAV reaches the requested altitude change and stabilizes."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="controlled_altitude_change",
            goal="Make bounded vertical progress toward the requested altitude change.",
            done_when=["requested altitude change is within tolerance", "UAV is stable after vertical movement"],
            continue_when=["altitude progress is insufficient", "vertical motion is overshooting or unsafe"],
            metrics=_vertical_metric_from_instruction(lower),
        )
        attributes = {"path_angle_role": "vertical_path" if "degree" in lower else "unspecified"}
    elif _has_any(lower, ["through", "past", "between", "pass ", " passage", "route"]):
        intent = "route_traversal"
        target = _target_from_instruction(lower)
        final_state = "UAV makes route progress through or past the requested cue while preserving the requested side relation."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="traverse_requested_route",
            goal="Identify the route cue or side relation and make bounded route progress.",
            done_when=["route cue is identified", "recent motion supports through, past, between, or side-route progress"],
            continue_when=["route cue is unclear", "only yaw or visual centering has occurred without route progress"],
            metrics=[],
        )
        attributes = {"relation": _relation_from_instruction(lower)}
    elif _has_any(lower, ["search", "find", "locate", "look for", "missing", "hidden"]):
        intent = "search_or_locate_target"
        target = _target_from_instruction(lower)
        final_state = "UAV has found a plausible target or route cue before acting on it."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="search_for_requested_cue",
            goal="Search for and confirm the requested target or route cue.",
            done_when=["target or route cue is visible or strongly identifiable"],
            continue_when=["target or route cue is absent, ambiguous, or confused with similar candidates"],
            metrics=[],
        )
        attributes = {}
    elif _has_any(lower, ["turn", "face", "look", "center", "view", "visible", "inspect"]):
        intent = "turn_to_visual_target"
        target = _target_from_instruction(lower)
        final_state = "UAV has the requested visual target or region in view."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="orient_to_visual_target",
            goal="Use current visual evidence to face, view, or confirm the requested target or region.",
            done_when=["target or region is visible", "view is roughly aligned with the visual objective"],
            continue_when=["visual target is unclear", "view is not plausibly aligned"],
            metrics=[],
        )
        attributes = {}
    elif _has_any(lower, ["move", "approach", "follow", "near", "distance", "closer"]):
        intent = "approach_or_standoff"
        target = _target_from_instruction(lower)
        final_state = "UAV reaches the requested relative position or distance while staying stable."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="approach_or_hold_relative_position",
            goal="Confirm the target relation and move or hold toward the requested distance or relative position.",
            done_when=["target relation is identifiable", "requested distance or relative position is plausible"],
            continue_when=["target relation is unclear", "requested relative position is not yet plausible"],
            metrics=[],
        )
        attributes = {"relation": _relation_from_instruction(lower)}
    else:
        intent = "general_visual_navigation"
        target = _target_from_instruction(lower)
        final_state = "UAV makes conservative progress on the requested visual navigation task."
        phase = SemanticPhase(
            index=1,
            total=1,
            name="general_visual_navigation_step",
            goal="Understand the current visual navigation objective and choose conservative task-relevant progress.",
            done_when=["current evidence supports the requested objective"],
            continue_when=["objective evidence is unclear or movement progress is insufficient"],
            metrics=[],
        )
        attributes = {}
    return SemanticPlan(
        intent=intent,
        task_goal=TaskGoal(
            target=target,
            target_attributes={key: value for key, value in attributes.items() if value},
            final_state=final_state,
        ),
        phases=[phase],
    )


def _semantic_plan_for_case(case: dict[str, Any]) -> SemanticPlan:
    instruction = str(case["instruction"])
    hint = case.get("semantic_hint")
    if isinstance(hint, dict):
        return _semantic_plan_from_hint(instruction, hint)
    return _semantic_plan_for_instruction(instruction)


def _semantic_plan_from_hint(instruction: str, hint: dict[str, Any]) -> SemanticPlan:
    target = str(hint.get("target", "")).strip() or _target_from_instruction(instruction.lower())
    final_state = str(hint.get("final_state", "")).strip() or f"UAV satisfies: {instruction}"
    phase = SemanticPhase(
        index=1,
        total=1,
        name=str(hint.get("phase_name", "")).strip() or "hinted_task_step",
        goal=str(hint.get("phase_goal", "")).strip() or final_state,
        done_when=[str(item).strip() for item in hint.get("done_when", []) if str(item).strip()],
        continue_when=[str(item).strip() for item in hint.get("continue_when", []) if str(item).strip()],
        metrics=[dict(item) for item in hint.get("metrics", []) if isinstance(item, dict)],
    )
    return SemanticPlan(
        intent=str(hint.get("intent", "")).strip() or "hinted_task",
        task_goal=TaskGoal(
            target=target,
            target_attributes=dict(hint.get("target_attributes", {})),
            final_state=final_state,
        ),
        phases=[phase],
    )


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _relation_from_instruction(text: str) -> str:
    relations = []
    for word, relation in [
        ("left side", "left_side"),
        ("right side", "right_side"),
        ("front side", "front_side"),
        ("back side", "back_side"),
        ("behind", "behind"),
        ("between", "between"),
        ("near", "near"),
    ]:
        if word in text:
            relations.append(relation)
    return ",".join(relations)


def _target_from_instruction(text: str) -> str:
    targets = [
        "sun umbrella",
        "sunshade",
        "target structure",
        "landmark",
        "person",
        "pedestrians",
        "vehicle",
        "car",
        "truck",
        "cyclist",
        "tree",
        "trees",
        "building",
        "buildings",
        "doorway",
        "entrance",
        "storefront",
        "monument",
        "sign",
        "alley",
    ]
    for target in targets:
        if target in text:
            return target
    if "surface" in text or "ground" in text:
        return "landing surface"
    return "requested target or region"


def _vertical_metric_from_instruction(text: str, default_drop_m: float | None = None) -> list[dict[str, Any]]:
    amount = _first_number(text)
    if amount is None:
        amount = default_drop_m
    if amount is None:
        return []
    sign = 1.0 if "ascend" in text else -1.0
    return [
        {
            "name": "up_distance",
            "target": sign * float(amount),
            "unit": "m",
            "tolerance": max(0.5, min(1.0, float(amount) * 0.1)),
            "correction_allowed": True,
        }
    ]


def _first_number(text: str) -> float | None:
    current = []
    for char in text:
        if char.isdigit() or char == ".":
            current.append(char)
        elif current:
            break
    if not current:
        return None
    try:
        return float("".join(current))
    except ValueError:
        return None


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
    reference_hits = sum(1 for item in results if item["expected_reference_hit"])
    reference_expectation_cases = sum(1 for item in results if item["expected_references"])
    missing_reference_cases = sum(1 for item in results if item["missing_expected_references"])
    forbidden_reference_cases = sum(1 for item in results if item["forbidden_reference_hits"])
    forbidden_cases = sum(1 for item in results if item["forbidden_hits"])
    over_selected = sum(1 for item in results if item["over_selected"])
    all_selected = sum(1 for item in results if len(item["selected_skill_names"]) >= item["available_skill_count"] > 0)
    none_selected = sum(1 for item in results if not item["selected_skill_names"])
    selected_counts = [len(item["selected_skill_names"]) for item in results]
    return {
        "total_cases": total,
        "expected_hit_rate": expected_hits / total if total else 0.0,
        "expected_reference_hit_rate": reference_hits / total if total else 0.0,
        "reference_expectation_cases": reference_expectation_cases,
        "missing_expected_reference_cases": missing_reference_cases,
        "forbidden_reference_cases": forbidden_reference_cases,
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
        "| instruction | selected | expected_hit | reference_hit | forbidden_hits | forbidden_reference_hits | over_selected |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["results"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["instruction"]).replace("|", "\\|"),
                    ", ".join(item["selected_skill_names"]) or "(none)",
                    str(item["expected_hit"]),
                    str(item["expected_reference_hit"]),
                    ", ".join(item["forbidden_hits"]) or "-",
                    json.dumps(item["forbidden_reference_hits"], ensure_ascii=False) if item["forbidden_reference_hits"] else "-",
                    str(item["over_selected"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _expected_reference_map(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    expected: dict[str, list[str]] = {}
    for skill_name, paths in raw.items():
        if not isinstance(paths, list):
            continue
        cleaned = [str(path).strip().replace("\\", "/") for path in paths if str(path).strip()]
        if cleaned:
            expected[str(skill_name).strip()] = cleaned
    return expected


def _missing_expected_references(
    expected: dict[str, list[str]],
    accepted: Any,
) -> dict[str, list[str]]:
    if not expected:
        return {}
    accepted_map = accepted if isinstance(accepted, dict) else {}
    missing: dict[str, list[str]] = {}
    for skill_name, paths in expected.items():
        accepted_paths = {
            str(path).strip().replace("\\", "/")
            for path in accepted_map.get(skill_name, [])
            if str(path).strip()
        }
        absent = [path for path in paths if path not in accepted_paths]
        if absent:
            missing[skill_name] = absent
    return missing


def _forbidden_reference_hits(
    forbidden: dict[str, list[str]],
    accepted: Any,
) -> dict[str, list[str]]:
    if not forbidden:
        return {}
    accepted_map = accepted if isinstance(accepted, dict) else {}
    hits: dict[str, list[str]] = {}
    for skill_name, paths in forbidden.items():
        accepted_paths = {
            str(path).strip().replace("\\", "/")
            for path in accepted_map.get(skill_name, [])
            if str(path).strip()
        }
        matched = [path for path in paths if path in accepted_paths]
        if matched:
            hits[skill_name] = matched
    return hits


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": summary["run_dir"],
        "selector_metrics": summary["selector"]["metrics"],
        "selector_statuses": [
            {
                "instruction": item["instruction"],
                "status": item["selection"].get("status"),
                "selected": item["selected_skill_names"],
                "missing_expected_references": item["missing_expected_references"],
                "forbidden_reference_hits": item["forbidden_reference_hits"],
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
