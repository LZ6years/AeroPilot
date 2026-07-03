"""Build UAV-Flow class coverage cases for the real skill selector probe.

This script does not call the selector directly. It converts UAV-Flow benchmark
instructions into the case-fixture format consumed by
``tests/manual/plugin_system_real_probe.py`` so selector behavior can be checked
across every UAV-Flow task class without putting benchmark-specific rules into
the selector prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases, load_classification
from evaluation.benchmarks.uav_flow.config import DEFAULT_UAV_FLOW_REPO


DEFAULT_OUTPUT = REPO_ROOT / "tests" / "manual_outputs" / "uav_flow_skill_selection_cases.json"


def _semantic_hint(
    *,
    intent: str,
    phase_name: str,
    phase_goal: str,
    final_state: str,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "target": "requested UAV-Flow target",
        "final_state": final_state,
        "phase_name": phase_name,
        "phase_goal": phase_goal,
        "done_when": [final_state],
        "observe": ["latest_observation", "phase_execution_summary"],
        "continue_when": ["task-relevant progress is insufficient"],
        "metrics": list(metrics or []),
    }


CLASS_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "Turn": {
        "expected_any": ["visual_target_task_reasoning"],
        "forbidden": ["route_traversal_and_passage", "vertical_landing_and_altitude_control"],
        "max_selected": 2,
        "semantic_hint": _semantic_hint(
            intent="turn_to_visual_target",
            phase_name="orient_to_visual_target",
            phase_goal="Face or visually align with the requested target.",
            final_state="The UAV has the requested target in view.",
        ),
    },
    "Move": {
        "expected_any": ["approach_and_standoff"],
        "forbidden": ["route_traversal_and_passage", "vertical_landing_and_altitude_control"],
        "max_selected": 2,
        "semantic_hint": _semantic_hint(
            intent="approach_or_standoff",
            phase_name="approach_or_hold_relative_position",
            phase_goal="Move toward the requested target-relative distance or position.",
            final_state="The UAV reaches the requested relative distance or position.",
        ),
    },
    "Shift": {
        "expected_any": [],
        "forbidden": [
            "visual_target_task_reasoning",
            "approach_and_standoff",
            "route_traversal_and_passage",
            "vertical_landing_and_altitude_control",
            "orbit_and_surround",
        ],
        "max_selected": 1,
        "semantic_hint": _semantic_hint(
            intent="primitive_translation",
            phase_name="execute_numeric_body_translation",
            phase_goal="Execute a numeric body-frame translation.",
            final_state="The requested numeric translation is complete.",
            metrics=[{"name": "planar_distance", "target": "instruction_distance_m"}],
        ),
    },
    "Rotate": {
        "expected_any": [],
        "forbidden": [
            "visual_target_task_reasoning",
            "approach_and_standoff",
            "route_traversal_and_passage",
            "vertical_landing_and_altitude_control",
            "orbit_and_surround",
        ],
        "max_selected": 1,
        "semantic_hint": _semantic_hint(
            intent="primitive_yaw_rotation",
            phase_name="execute_numeric_yaw_rotation",
            phase_goal="Execute a numeric yaw rotation.",
            final_state="The requested numeric yaw rotation is complete.",
            metrics=[{"name": "yaw_delta_deg", "target": "instruction_yaw_deg"}],
        ),
    },
    "Surround": {
        "expected_any": ["orbit_and_surround"],
        "forbidden": ["route_traversal_and_passage", "vertical_landing_and_altitude_control"],
        "max_selected": 2,
        "semantic_hint": _semantic_hint(
            intent="orbit_or_surround_target",
            phase_name="orbit_requested_target",
            phase_goal="Maintain the requested target-relative radius and move around the target.",
            final_state="The UAV has made sustained orbit progress around the requested target.",
        ),
    },
    "Ascend/Descend": {
        "expected_any": ["vertical_landing_and_altitude_control"],
        "forbidden": ["visual_target_task_reasoning", "route_traversal_and_passage", "orbit_and_surround"],
        "max_selected": 2,
        "semantic_hint": _semantic_hint(
            intent="altitude_control",
            phase_name="controlled_altitude_change",
            phase_goal="Make bounded vertical progress toward the requested altitude change.",
            final_state="The UAV reaches the requested altitude change and stabilizes.",
            metrics=[{"name": "up_distance", "target": "instruction_vertical_distance_m"}],
        ),
    },
    "Approach": {
        "expected_any": ["approach_and_standoff"],
        "forbidden": ["route_traversal_and_passage", "vertical_landing_and_altitude_control"],
        "max_selected": 3,
        "semantic_hint": _semantic_hint(
            intent="approach_or_standoff",
            phase_name="approach_or_hold_relative_position",
            phase_goal="Move toward the requested target-relative position.",
            final_state="The UAV reaches the requested target-relative position.",
        ),
    },
    "Retreat": {
        "expected_any": ["approach_and_standoff"],
        "forbidden": ["route_traversal_and_passage", "vertical_landing_and_altitude_control"],
        "max_selected": 2,
        "semantic_hint": _semantic_hint(
            intent="approach_or_standoff",
            phase_name="increase_target_standoff",
            phase_goal="Increase distance from the requested target while preserving awareness.",
            final_state="The UAV has backed away from the requested target.",
        ),
    },
    "Pass": {
        "expected_any": ["route_traversal_and_passage"],
        "forbidden": ["vertical_landing_and_altitude_control", "orbit_and_surround"],
        "max_selected": 3,
        "semantic_hint": _semantic_hint(
            intent="route_traversal",
            phase_name="traverse_requested_route",
            phase_goal="Go through, past, between, or along the requested route cue.",
            final_state="The UAV makes route progress through or past the requested cue.",
        ),
    },
    "Land": {
        "expected_any": ["vertical_landing_and_altitude_control"],
        "forbidden": ["route_traversal_and_passage", "orbit_and_surround"],
        "max_selected": 3,
        "semantic_hint": _semantic_hint(
            intent="vertical_landing",
            phase_name="controlled_landing_near_requested_region",
            phase_goal="Make controlled altitude progress toward the requested landing relation.",
            final_state="The UAV reaches a controlled landing or low-altitude hold near the requested region.",
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UAV-Flow skill-selection probe cases.")
    parser.add_argument("--uav-flow-repo", default=str(_default_uav_flow_repo()))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit-per-class", type=int, default=3)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--instruction-field", default="instruction_unified")
    args = parser.parse_args()

    repo = Path(args.uav_flow_repo)
    cases = build_cases(
        repo,
        classes=args.classes,
        limit_per_class=args.limit_per_class,
        instruction_field=args.instruction_field,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_summary(output, cases), ensure_ascii=False, indent=2))
    print(f"Run selector probe: python tests/manual/plugin_system_real_probe.py --cases {output}")


def build_cases(
    uav_flow_repo: Path,
    *,
    classes: list[str] | None = None,
    limit_per_class: int = 3,
    instruction_field: str = "instruction_unified",
) -> list[dict[str, Any]]:
    selected_classes = classes or _classes_from_repo(uav_flow_repo)
    loaded = load_cases(
        uav_flow_repo,
        classes=selected_classes,
        limit_per_class=limit_per_class,
        instruction_field=instruction_field,
    )
    return [_case_to_probe_item(case) for case in loaded if case.class_name in CLASS_EXPECTATIONS]


def _case_to_probe_item(case: UavFlowCase) -> dict[str, Any]:
    expectation = CLASS_EXPECTATIONS[str(case.class_name)]
    item = {
        "instruction": case.selected_instruction,
        "uav_flow_class": case.class_name,
        "uav_flow_case_id": case.case_id,
        "source_file": case.file_name,
        "expected_any": list(expectation["expected_any"]),
        "forbidden": list(expectation["forbidden"]),
        "max_selected": int(expectation["max_selected"]),
        "semantic_hint": dict(expectation["semantic_hint"]),
    }
    return item


def _classes_from_repo(uav_flow_repo: Path) -> list[str]:
    classified = uav_flow_repo / "classified_instr.json"
    return [name for name in load_classification(classified) if name in CLASS_EXPECTATIONS]


def _default_uav_flow_repo() -> Path:
    configured = Path(DEFAULT_UAV_FLOW_REPO)
    if (configured / "classified_instr.json").exists():
        return configured
    local_ref = REPO_ROOT / ".external_refs" / "UAV-Flow" / "UAV-Flow-Eval"
    return local_ref


def _summary(output: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in cases:
        class_name = str(item.get("uav_flow_class", ""))
        counts[class_name] = counts.get(class_name, 0) + 1
    return {
        "output": str(output),
        "total_cases": len(cases),
        "classes": counts,
    }


if __name__ == "__main__":
    main()
