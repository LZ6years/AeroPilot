"""Validate and serialize semantic task plans."""

from __future__ import annotations

import re
from typing import Any

from src.agent.semantic.schemas import SemanticPlan


class SemanticPlanValidationError(ValueError):
    """Raised when a semantic plan is unsafe or structurally invalid."""


FORBIDDEN_ACTION_TOKENS = (
    "env.observe",
    "move_body",
    "turn_relative",
    "get_frames",
    "get_pose",
)

SEMANTIC_PHASE_PATCH_FIELDS = {"goal", "done_when", "continue_when"}
SEMANTIC_PLAN_PATCH_TOP_LEVEL_FIELDS = {"phase", "append_recovery_phase"}


def validate_semantic_plan(plan: SemanticPlan) -> SemanticPlan:
    # Keep M3 semantic-only: no primitive action names or executable tool calls.
    if not plan.intent:
        raise SemanticPlanValidationError("SemanticPlan.intent is required")
    if not plan.task_goal.target:
        raise SemanticPlanValidationError("SemanticPlan.task_goal.target is required")
    if not plan.task_goal.final_state:
        raise SemanticPlanValidationError("SemanticPlan.task_goal.final_state is required")
    if not plan.phases:
        raise SemanticPlanValidationError("SemanticPlan.phases must not be empty")
    for expected_index, phase in enumerate(plan.phases, start=1):
        if phase.index != expected_index:
            raise SemanticPlanValidationError(
                f"SemanticPhase.index must be consecutive and 1-based: {phase.name}"
            )
        if phase.total != len(plan.phases):
            raise SemanticPlanValidationError(f"SemanticPhase.total must equal phases length: {phase.name}")
        if not phase.name:
            raise SemanticPlanValidationError("SemanticPhase.name is required")
        if not phase.goal:
            raise SemanticPlanValidationError(f"SemanticPhase.goal is required: {phase.name}")
        if not phase.done_when:
            raise SemanticPlanValidationError(f"SemanticPhase.done_when is required: {phase.name}")
    _reject_forbidden_tokens(plan.to_dict())
    return plan


def semantic_plan_from_dict(data: dict[str, Any]) -> SemanticPlan:
    # Convert untrusted model/rule data into a validated semantic plan.
    plan = SemanticPlan.from_dict(data)
    return validate_semantic_plan(plan)


def validate_semantic_plan_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate the limited patch shape produced by semantic_plan_update."""

    if not isinstance(patch, dict) or not patch:
        raise SemanticPlanValidationError("Semantic plan patch must be a non-empty object")
    unknown_top_level = set(patch) - SEMANTIC_PLAN_PATCH_TOP_LEVEL_FIELDS
    if unknown_top_level:
        raise SemanticPlanValidationError(
            f"Semantic plan patch contains unsupported top-level field(s): {sorted(unknown_top_level)}"
        )
    if "phase" not in patch and "append_recovery_phase" not in patch:
        raise SemanticPlanValidationError("Semantic plan patch must update phase or append_recovery_phase")
    result: dict[str, Any] = {}
    if "phase" in patch:
        phase_patch = patch.get("phase")
        if not isinstance(phase_patch, dict) or not phase_patch:
            raise SemanticPlanValidationError("Semantic plan patch phase must be a non-empty object")
        unknown_phase_fields = set(phase_patch) - SEMANTIC_PHASE_PATCH_FIELDS
        if unknown_phase_fields:
            raise SemanticPlanValidationError(
                f"Semantic phase patch contains unsupported field(s): {sorted(unknown_phase_fields)}"
            )
        result["phase"] = {key: phase_patch[key] for key in phase_patch}
    if "append_recovery_phase" in patch:
        recovery = patch.get("append_recovery_phase")
        if recovery is not None:
            if not isinstance(recovery, dict):
                raise SemanticPlanValidationError("append_recovery_phase must be an object when present")
            result["append_recovery_phase"] = dict(recovery)
    _reject_forbidden_tokens(result)
    return result


def apply_semantic_plan_patch(
    plan: SemanticPlan,
    current_phase_index: int,
    patch: dict[str, Any],
) -> SemanticPlan:
    """Apply a validated current-phase semantic patch and revalidate the plan."""

    if current_phase_index < 0 or current_phase_index >= len(plan.phases):
        raise SemanticPlanValidationError("current_phase_index is outside semantic plan phases")
    clean_patch = validate_semantic_plan_patch(patch)
    plan_data = plan.to_dict()
    phases = [dict(item) for item in plan_data.get("phases", [])]
    current_phase = dict(phases[current_phase_index])
    for key, value in dict(clean_patch.get("phase", {})).items():
        current_phase[key] = value
    phases[current_phase_index] = current_phase
    recovery = clean_patch.get("append_recovery_phase")
    if isinstance(recovery, dict) and recovery:
        phases.append(recovery)
    plan_data["phases"] = phases
    return semantic_plan_from_dict(plan_data)


def _reject_forbidden_tokens(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_forbidden_tokens(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_forbidden_tokens(item)
        return
    if not isinstance(value, str):
        return
    lower = value.lower()
    namespace_match = re.search(r"(?<![a-z0-9_])uav\.[a-z_][a-z0-9_]*", lower)
    if namespace_match:
        token = namespace_match.group(0)
        raise SemanticPlanValidationError(f"Semantic plan must not contain primitive action token: {token}")
    for token in FORBIDDEN_ACTION_TOKENS:
        if _contains_forbidden_token(lower, token):
            raise SemanticPlanValidationError(f"Semantic plan must not contain primitive action token: {token}")


def _contains_forbidden_token(value: str, token: str) -> bool:
    if "." in token:
        return token in value
    return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", value) is not None
