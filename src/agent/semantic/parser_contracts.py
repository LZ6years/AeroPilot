"""Internal contracts for two-stage semantic task parsing.

These internal structures ensure the skill-informed criteria stage cannot
rewrite phase structure chosen from user intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.semantic.schemas import SemanticPlan
from src.agent.semantic.validator import semantic_plan_from_dict


STRUCTURE_TOP_LEVEL_FIELDS = {"intent", "task_goal", "phases"}
STRUCTURE_PHASE_FIELDS = {"index", "total", "name", "goal"}
CRITERIA_FIELDS = {"index", "name", "done_when", "observe", "continue_when", "metrics"}
_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


class TaskParserContractError(ValueError):
    """Raised when a model response violates a two-stage parser contract."""


@dataclass(frozen=True)
class SemanticPlanStructure:
    """Immutable user-intent structure produced before skill selection."""

    intent: str
    task_goal: dict[str, Any]
    phases: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticPlanStructure":
        if not isinstance(data, dict):
            raise TaskParserContractError("semantic structure must be an object")
        unknown = set(data) - STRUCTURE_TOP_LEVEL_FIELDS
        if unknown:
            raise TaskParserContractError(f"semantic structure contains unsupported fields: {sorted(unknown)}")
        intent = _required_string(data.get("intent"), "intent")
        task_goal = _validate_task_goal(data.get("task_goal"))
        raw_phases = data.get("phases")
        if not isinstance(raw_phases, list) or not raw_phases:
            raise TaskParserContractError("semantic structure phases must be a non-empty list")
        total = len(raw_phases)
        phases = [
            _validate_structure_phase(item, expected_index=index, expected_total=total)
            for index, item in enumerate(raw_phases, start=1)
        ]
        return cls(intent=intent, task_goal=task_goal, phases=phases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "task_goal": dict(self.task_goal),
            "phases": [dict(item) for item in self.phases],
        }


def validate_phase_criteria(
    data: dict[str, Any],
    phase: dict[str, Any],
    *,
    instruction: str | None = None,
    task_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a criteria-only response against its immutable phase."""

    if not isinstance(data, dict):
        raise TaskParserContractError("phase criteria must be an object")
    unknown = set(data) - CRITERIA_FIELDS
    if unknown:
        raise TaskParserContractError(f"phase criteria contains unsupported fields: {sorted(unknown)}")
    if set(data) != CRITERIA_FIELDS:
        missing = sorted(CRITERIA_FIELDS - set(data))
        raise TaskParserContractError(f"phase criteria is missing fields: {missing}")
    index = _required_int(data.get("index"), "phase criteria index")
    name = _required_string(data.get("name"), "phase criteria name")
    if index != int(phase["index"]) or name != str(phase["name"]):
        raise TaskParserContractError("phase criteria index/name does not match immutable phase structure")
    done_when = _string_list(data.get("done_when"), "done_when", require_nonempty=True)
    observe = _string_list(data.get("observe"), "observe")
    continue_when = _string_list(data.get("continue_when"), "continue_when")
    metrics = data.get("metrics")
    if not isinstance(metrics, list) or any(not isinstance(item, dict) for item in metrics):
        raise TaskParserContractError("phase criteria metrics must be a list of objects")
    if instruction is not None:
        _validate_visual_alignment_precision(instruction, done_when)
        _validate_numeric_provenance(
            instruction,
            done_when=done_when,
            continue_when=continue_when,
            metrics=metrics,
            task_goal=task_goal or {},
        )
    return {
        "index": index,
        "name": name,
        "done_when": done_when,
        "observe": observe,
        "continue_when": continue_when,
        "metrics": [dict(item) for item in metrics],
    }


def merge_semantic_plan(
    structure: SemanticPlanStructure,
    criteria_by_phase: list[dict[str, Any]],
) -> SemanticPlan:
    """Merge validated criteria without allowing structural rewrites."""

    if len(criteria_by_phase) != len(structure.phases):
        raise TaskParserContractError("criteria count must equal structure phase count")
    phases = []
    for phase, criteria in zip(structure.phases, criteria_by_phase):
        clean = validate_phase_criteria(criteria, phase)
        phases.append(
            {
                **dict(phase),
                "done_when": clean["done_when"],
                "observe": clean["observe"],
                "continue_when": clean["continue_when"],
                "metrics": clean["metrics"],
            }
        )
    return semantic_plan_from_dict(
        {
            "intent": structure.intent,
            "task_goal": dict(structure.task_goal),
            "phases": phases,
        }
    )


def _validate_task_goal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskParserContractError("task_goal must be an object")
    allowed = {"target", "target_attributes", "final_state"}
    unknown = set(value) - allowed
    if unknown:
        raise TaskParserContractError(f"task_goal contains unsupported fields: {sorted(unknown)}")
    attributes = value.get("target_attributes", {})
    if not isinstance(attributes, dict):
        raise TaskParserContractError("task_goal.target_attributes must be an object")
    return {
        "target": _required_string(value.get("target"), "task_goal.target"),
        "target_attributes": dict(attributes),
        "final_state": _required_string(value.get("final_state"), "task_goal.final_state"),
    }


def _validate_structure_phase(value: Any, *, expected_index: int, expected_total: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskParserContractError("semantic structure phase must be an object")
    unknown = set(value) - STRUCTURE_PHASE_FIELDS
    if unknown:
        raise TaskParserContractError(f"semantic structure phase contains unsupported fields: {sorted(unknown)}")
    if set(value) != STRUCTURE_PHASE_FIELDS:
        missing = sorted(STRUCTURE_PHASE_FIELDS - set(value))
        raise TaskParserContractError(f"semantic structure phase is missing fields: {missing}")
    index = _required_int(value.get("index"), "phase index")
    total = _required_int(value.get("total"), "phase total")
    if index != expected_index or total != expected_total:
        raise TaskParserContractError("semantic structure phase index/total must be consecutive and consistent")
    return {
        "index": index,
        "total": total,
        "name": _required_string(value.get("name"), "phase name"),
        "goal": _required_string(value.get("goal"), "phase goal"),
    }


def _required_string(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise TaskParserContractError(f"{field_name} is required")
    return text


def _required_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TaskParserContractError(f"{field_name} must be an integer") from exc


def _string_list(value: Any, field_name: str, *, require_nonempty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise TaskParserContractError(f"phase criteria {field_name} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if require_nonempty and not result:
        raise TaskParserContractError(f"phase criteria {field_name} must not be empty")
    return result


def _validate_numeric_provenance(
    instruction: str,
    *,
    done_when: list[str],
    continue_when: list[str],
    metrics: list[dict[str, Any]],
    task_goal: dict[str, Any],
) -> None:
    instruction_numbers = _numbers(instruction)
    for condition in [*done_when, *continue_when]:
        unsupported = [value for value in _numbers(condition) if not _contains_abs(instruction_numbers, value)]
        if unsupported:
            raise TaskParserContractError(
                f"phase criteria condition contains numbers absent from the user instruction: {unsupported}"
            )

    target_distance = _optional_float(
        dict(task_goal.get("target_attributes", {})).get("distance_m")
        if isinstance(task_goal.get("target_attributes", {}), dict)
        else None
    )
    for metric in metrics:
        name = str(metric.get("name", "")).strip()
        target = _optional_float(metric.get("target"))
        if target is None:
            raise TaskParserContractError(f"phase criteria metric {name or '<unnamed>'} requires numeric target")
        if name == "planar_distance" and target_distance is not None and _same_abs(target, target_distance):
            raise TaskParserContractError(
                "target-relative task_goal distance_m cannot be used as phase-start planar_distance"
            )
        if _contains_abs(instruction_numbers, target):
            continue
        if name in {"forward_distance", "right_distance"} and len(instruction_numbers) >= 2:
            # A horizontal distance plus bearing may be decomposed into signed
            # forward/right components by the criteria contract.
            continue
        raise TaskParserContractError(
            f"phase criteria metric target {target} for {name or '<unnamed>'} has no numeric source in the user instruction"
        )


def _validate_visual_alignment_precision(instruction: str, done_when: list[str]) -> None:
    instruction_lower = instruction.lower()
    explicit_precision_terms = ("center", "centre", "middle", "exact", "precise", "optical axis")
    if any(term in instruction_lower for term in explicit_precision_terms):
        return
    acceptable_qualifiers = ("roughly centered", "approximately centered", "near the center", "near center")
    for condition in done_when:
        lowered = condition.lower()
        if "centered" in lowered and not any(qualifier in lowered for qualifier in acceptable_qualifiers):
            raise TaskParserContractError(
                "visual alignment criteria require bare centering precision that the user did not request"
            )
        if "exact optical-axis" in lowered or "exact optical axis" in lowered:
            raise TaskParserContractError(
                "visual alignment criteria require exact optical-axis precision that the user did not request"
            )


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER_PATTERN.finditer(str(text))]


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains_abs(values: list[float], target: float) -> bool:
    return any(_same_abs(value, target) for value in values)


def _same_abs(left: float, right: float) -> bool:
    return abs(abs(left) - abs(right)) <= 1e-6
