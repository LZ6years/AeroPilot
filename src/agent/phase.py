"""Phase completion and transition contracts for FlyAgent missions.

This module is intentionally model- and runtime-neutral: planners and judges
can read these schemas, while TaskRuntime remains the only component that
mutates MissionState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.messages import AgentMessage


CONTINUE_CURRENT_PHASE = "continue_current_phase"
ADVANCE_TO_NEXT_PHASE = "advance_to_next_phase"
COMPLETE_TASK = "complete_task"
VALID_PHASE_TRANSITIONS = {
    CONTINUE_CURRENT_PHASE,
    ADVANCE_TO_NEXT_PHASE,
    COMPLETE_TASK,
}

TRANSLATION_FIELDS = {
    "forward": "forward_displacement_m",
    "backward": "forward_displacement_m",
    "right": "right_displacement_m",
    "left": "right_displacement_m",
    "up": "up_displacement_m",
    "down": "up_displacement_m",
    "planar": "planar_displacement_m",
}


@dataclass(frozen=True)
class PhaseTransition:
    """Model-facing phase progression suggestion for the current phase window."""

    decision: str
    reason: str = ""
    agent_messages: list[AgentMessage] = field(default_factory=list)

    @classmethod
    def default_advance(cls, reason: str = "legacy short-horizon plan advances after one window") -> "PhaseTransition":
        return cls(ADVANCE_TO_NEXT_PHASE, reason)

    @classmethod
    def continue_current_phase(cls, reason: str = "continue collecting evidence for current phase") -> "PhaseTransition":
        return cls(CONTINUE_CURRENT_PHASE, reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseTransition":
        payload = dict(data)
        illegal_targets = sorted(set(payload) & {"phase", "phase_name", "phase_index", "next_phase", "next_phase_index"})
        if illegal_targets:
            raise ValueError(
                "phase_transition must not specify target phase fields: "
                + ", ".join(illegal_targets)
            )
        decision = str(payload.get("decision", "")).strip()
        if decision not in VALID_PHASE_TRANSITIONS:
            raise ValueError(f"Unsupported phase_transition decision: {decision}")
        messages_data = payload.get("agent_messages", [])
        if messages_data is None:
            messages_data = []
        if not isinstance(messages_data, list):
            raise ValueError("phase_transition agent_messages must be a list")
        return cls(
            decision=decision,
            reason=str(payload.get("reason", "")).strip(),
            agent_messages=[AgentMessage.from_dict(dict(item)) for item in messages_data],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "agent_messages": [message.to_dict() for message in self.agent_messages],
        }


def criteria_measurement_field(criteria: dict[str, Any]) -> str | None:
    """Return the model-facing measurement field for a completion criterion."""

    axis = str(criteria.get("axis", "")).strip()
    kind = str(criteria.get("kind", "")).strip()
    if kind == "rotation" or axis == "yaw":
        return "yaw_delta_deg"
    if kind == "hover":
        return "planar_displacement_m"
    return TRANSLATION_FIELDS.get(axis)


def evaluate_phase_completion(
    criteria: list[dict[str, Any]] | dict[str, Any] | None,
    phase_execution_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare criteria targets with current phase measurements."""

    criteria_list = _normalize_criteria(criteria)
    measurements = _phase_measurements(phase_execution_summary)
    evaluations = []
    for item in criteria_list:
        field = criteria_measurement_field(item)
        actual = _actual_for_field(field, measurements, phase_execution_summary, item)
        target = _signed_target(item)
        tolerance = float(item.get("tolerance", 0.0) or 0.0)
        error = None if actual is None else round(float(actual) - target, 3)
        within = actual is not None and abs(float(error)) <= tolerance
        overshoot = actual is not None and _is_overshoot(float(actual), target, tolerance)
        evaluations.append(
            {
                "criteria": dict(item),
                "field": field,
                "target": target,
                "actual": actual,
                "error": error,
                "tolerance": tolerance,
                "within_tolerance": within,
                "overshoot": overshoot,
                "correction_allowed": bool(item.get("correction_allowed", False)),
                "status": _status(actual, target, tolerance, overshoot, bool(item.get("correction_allowed", False))),
            }
        )
    return {
        "criteria_count": len(evaluations),
        "all_within_tolerance": bool(evaluations) and all(item["within_tolerance"] for item in evaluations),
        "any_overshoot": any(item["overshoot"] for item in evaluations),
        "evaluations": evaluations,
    }


def _normalize_criteria(criteria: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(criteria, list):
        return [dict(item) for item in criteria if isinstance(item, dict)]
    if isinstance(criteria, dict):
        return [dict(criteria)]
    return []


def _phase_measurements(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ("phase_local_measurements", "cumulative_measurements", "last_window_measurements"):
        value = summary.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _actual_for_field(
    field: str | None,
    measurements: dict[str, Any],
    summary: dict[str, Any],
    criteria: dict[str, Any],
) -> float | None:
    if not field:
        return None
    prefer_tool_fact = criteria.get("prefer_tool_fact")
    facts = summary.get("cumulative_tool_facts", {}) if isinstance(summary.get("cumulative_tool_facts", {}), dict) else {}
    if prefer_tool_fact and prefer_tool_fact in facts:
        return _safe_float(facts.get(prefer_tool_fact))
    return _safe_float(measurements.get(field))


def _signed_target(criteria: dict[str, Any]) -> float:
    target = float(criteria.get("target", 0.0) or 0.0)
    axis = str(criteria.get("axis", "")).strip()
    if axis in {"backward", "left", "down"} and target > 0:
        return -target
    return target


def _is_overshoot(actual: float, target: float, tolerance: float) -> bool:
    if target == 0:
        return abs(actual) > tolerance
    if target > 0:
        return actual > target + tolerance
    return actual < target - tolerance


def _status(actual: float | None, target: float, tolerance: float, overshoot: bool, correction_allowed: bool) -> str:
    if actual is None:
        return "missing_measurement"
    if abs(float(actual) - target) <= tolerance:
        return "within_tolerance"
    if overshoot and correction_allowed:
        return "overshoot_correction_allowed"
    if overshoot:
        return "overshoot"
    return "below_target"


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
