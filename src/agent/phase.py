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

METRIC_FIELDS = {
    "forward_distance": "forward_m",
    "right_distance": "right_m",
    "up_distance": "up_m",
    "planar_distance": "planar_m",
    "yaw_delta": "yaw_delta_deg",
    "hover_stability": "planar_m",
}

MEASURABLE_METRIC_NAMES = set(METRIC_FIELDS)


@dataclass(frozen=True)
class PhaseTransition:
    """Model-facing phase progression suggestion for the current phase window."""

    decision: str
    reason: str = ""
    agent_messages: list[AgentMessage] = field(default_factory=list)

    @classmethod
    def continue_current_phase(cls, reason: str = "continue collecting evidence for current phase") -> "PhaseTransition":
        return cls(CONTINUE_CURRENT_PHASE, reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseTransition":
        payload = dict(data)
        illegal_targets = sorted(set(payload) & {"phase", "phase_name", "phase_index", "next_phase", "next_phase_index"})
        if illegal_targets:
            raise ValueError(
                "phase_decision must not specify target phase fields: "
                + ", ".join(illegal_targets)
            )
        decision = str(payload.get("decision", "")).strip()
        if decision not in VALID_PHASE_TRANSITIONS:
            raise ValueError(f"Unsupported phase_decision decision: {decision}")
        messages_data = payload.get("agent_messages", [])
        if messages_data is None:
            messages_data = []
        if not isinstance(messages_data, list):
            raise ValueError("phase_decision agent_messages must be a list")
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


def metric_measurement_field(metric: dict[str, Any]) -> str | None:
    """Return the model-facing measurement field for a phase metric."""

    return METRIC_FIELDS.get(str(metric.get("name", "")).strip())


def evaluate_phase_completion(
    metrics: list[dict[str, Any]] | dict[str, Any] | None,
    phase_execution_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare phase metrics with current phase measurements."""

    metric_list = _normalize_metrics(metrics)
    measurements = _phase_measurements(phase_execution_summary)
    if not metric_list:
        return {
            "metrics_count": 0,
            "measurable_count": 0,
            "model_judged_count": 1,
            "all_within_tolerance": False,
            "any_overshoot": False,
            "evaluations": [
                {
                    "metric": None,
                    "field": None,
                    "target": None,
                    "actual": None,
                    "error": None,
                    "tolerance": None,
                    "within_tolerance": False,
                    "overshoot": False,
                    "correction_allowed": False,
                    "status": "requires_model_judgment",
                }
            ],
        }
    evaluations = []
    for item in metric_list:
        if not _is_measurable_metric(item):
            evaluations.append(_model_judged_evaluation(item))
            continue
        field = metric_measurement_field(item)
        actual = _actual_for_field(field, measurements, phase_execution_summary, item)
        target = float(item.get("target", 0.0) or 0.0)
        tolerance = float(item.get("tolerance", 0.0) or 0.0)
        error = None if actual is None else round(float(actual) - target, 3)
        within = actual is not None and abs(float(error)) <= tolerance
        overshoot = actual is not None and _is_overshoot(float(actual), target, tolerance)
        evaluations.append(
            {
                "metric": dict(item),
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
    measurable = [item for item in evaluations if item["status"] != "requires_model_judgment"]
    model_judged = [item for item in evaluations if item["status"] == "requires_model_judgment"]
    return {
        "metrics_count": len(evaluations),
        "measurable_count": len(measurable),
        "model_judged_count": len(model_judged),
        "all_within_tolerance": bool(measurable) and all(item["within_tolerance"] for item in measurable),
        "any_overshoot": any(item["overshoot"] for item in evaluations),
        "evaluations": evaluations,
    }


def _normalize_metrics(metrics: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if isinstance(metrics, list):
        return [dict(item) for item in metrics if isinstance(item, dict)]
    if isinstance(metrics, dict):
        return [dict(metrics)]
    return []


def _phase_measurements(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ("measurements", "last_window"):
        value = summary.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _actual_for_field(
    field: str | None,
    measurements: dict[str, Any],
    summary: dict[str, Any],
    metric: dict[str, Any],
) -> float | None:
    if not field:
        return None
    prefer_tool_fact = metric.get("prefer_tool_fact")
    facts = summary.get("tool_facts", {}) if isinstance(summary.get("tool_facts", {}), dict) else {}
    if prefer_tool_fact and prefer_tool_fact in facts:
        return _safe_float(facts.get(prefer_tool_fact))
    return _safe_float(measurements.get(field))


def _is_measurable_metric(metric: dict[str, Any]) -> bool:
    name = str(metric.get("name", "")).strip()
    return name in MEASURABLE_METRIC_NAMES


def _model_judged_evaluation(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": dict(metric),
        "field": None,
        "target": metric.get("target"),
        "actual": None,
        "error": None,
        "tolerance": None,
        "within_tolerance": False,
        "overshoot": False,
        "correction_allowed": bool(metric.get("correction_allowed", False)),
        "status": "requires_model_judgment",
    }


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
