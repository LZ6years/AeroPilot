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
CRITERIA_FIELDS = {"index", "name", "done_when", "continue_when"}
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
    continue_when = _string_list(data.get("continue_when"), "continue_when")
    if instruction is not None:
        _validate_numeric_provenance(
            instruction,
            done_when=done_when,
            continue_when=continue_when,
        )
    return {
        "index": index,
        "name": name,
        "done_when": done_when,
        "continue_when": continue_when,
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
                "continue_when": clean["continue_when"],
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
) -> None:
    instruction_numbers = _numbers(instruction)
    for condition in [*done_when, *continue_when]:
        unsupported = [value for value in _numbers(condition) if not _contains_abs(instruction_numbers, value)]
        if unsupported:
            raise TaskParserContractError(
                f"phase criteria condition contains numbers absent from the user instruction: {unsupported}"
            )


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in _NUMBER_PATTERN.finditer(str(text))]


def _contains_abs(values: list[float], target: float) -> bool:
    return any(_same_abs(value, target) for value in values)


def _same_abs(left: float, right: float) -> bool:
    return abs(abs(left) - abs(right)) <= 1e-6
