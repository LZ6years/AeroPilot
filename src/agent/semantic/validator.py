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
    "takeoff",
    "land",
    "get_frames",
    "get_pose",
)


def validate_semantic_plan(plan: SemanticPlan) -> SemanticPlan:
    # Keep M3 semantic-only: no primitive action names or executable tool calls.
    if not plan.intent:
        raise SemanticPlanValidationError("SemanticPlan.intent is required")
    if not plan.goal.target.description:
        raise SemanticPlanValidationError("SemanticPlan.goal.target.description is required")
    if not plan.goal.final_state:
        raise SemanticPlanValidationError("SemanticPlan.goal.final_state is required")
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
        if not phase.meaning:
            raise SemanticPlanValidationError(f"SemanticPhase.meaning is required: {phase.name}")
    _reject_forbidden_tokens(plan.to_dict())
    return plan


def semantic_plan_from_dict(data: dict[str, Any]) -> SemanticPlan:
    # Convert untrusted model/rule data into a validated semantic plan.
    plan = SemanticPlan.from_dict(data)
    return validate_semantic_plan(plan)


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
