"""Build the runtime evidence bundle for one phase-window model call.

This module owns phase-window evidence that is broader than perception memory
fragments but still model-facing: deterministic phase metrics, compact motion
progress, and a cleaned summary of the latest observation. It deliberately does
not execute actions or mutate MissionState; TaskRuntime remains responsible for
the flight loop and state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.phase import evaluate_phase_completion
from src.agent.planning.model_inputs import sanitize_context_for_model_text
@dataclass(frozen=True)
class PhaseWindowContext:
    """Model/debug payloads prepared for a single phase window."""

    planner_payload: dict[str, Any]
    model_context: dict[str, Any]
    phase_metrics: list[dict[str, Any]]
    phase_completion_evaluation: dict[str, Any]
    phase_progress_evidence: dict[str, Any]


class PhaseWindowContextBuilder:
    """Compose phase-window evidence on top of memory-derived planner context."""

    def build(
        self,
        state: MissionState,
        base_planner_context: dict[str, Any],
        observation_before: dict[str, Any],
        runtime_control_context: dict[str, Any] | None = None,
    ) -> PhaseWindowContext:
        """Return the complete model-facing context for the current phase window."""

        metrics = self.current_phase_metrics(state)
        completion = evaluate_phase_completion(metrics, state.current_phase_execution_summary)
        progress = self.phase_progress_evidence(state)
        planner_payload = dict(base_planner_context)
        runtime_control = self.compact_runtime_control_context(runtime_control_context)
        if runtime_control:
            planner_payload["runtime_control"] = runtime_control
        planner_payload.update(
            {
                "phase_metrics": metrics,
                "phase_completion_evaluation": completion,
                "phase_progress_evidence": progress,
            }
        )
        return PhaseWindowContext(
            planner_payload=planner_payload,
            model_context=sanitize_context_for_model_text(planner_payload),
            phase_metrics=metrics,
            phase_completion_evaluation=completion,
            phase_progress_evidence=progress,
        )

    @staticmethod
    def current_phase_metrics(state: MissionState) -> list[dict[str, Any]]:
        """Return metric contracts for the currently active semantic phase."""

        if state.current_phase_index >= len(state.semantic_plan.phases):
            return []
        return [dict(item) for item in state.semantic_plan.phases[state.current_phase_index].metrics]

    def phase_progress_evidence(self, state: MissionState) -> dict[str, Any]:
        """Summarize model-facing motion facts for the current phase."""

        summary = state.current_phase_execution_summary
        measurements = summary.get("measurements", {}) if isinstance(summary.get("measurements"), dict) else {}
        tool_facts = summary.get("tool_facts", {}) if isinstance(summary.get("tool_facts"), dict) else {}
        tools_executed = tool_facts.get("tools_executed", {})
        if not isinstance(tools_executed, dict):
            tools_executed = {}
        forward_m = self._safe_model_float(measurements.get("forward_m"), 0.0)
        right_m = self._safe_model_float(measurements.get("right_m"), 0.0)
        up_m = self._safe_model_float(measurements.get("up_m"), 0.0)
        planar_m = self._safe_model_float(measurements.get("planar_m"), 0.0)
        yaw_delta_deg = self._safe_model_float(measurements.get("yaw_delta_deg"), 0.0)
        move_duration = self._safe_model_float(tool_facts.get("move_duration_success_sum_sec"), 0.0)
        turn_yaw = self._safe_model_float(tool_facts.get("turn_yaw_success_sum_deg"), 0.0)
        move_count = int(self._safe_model_float(tools_executed.get("uav.move_body"), 0.0) or 0)
        turn_count = int(self._safe_model_float(tools_executed.get("uav.turn_relative"), 0.0) or 0)
        translation_epsilon = 0.05
        has_translation = (
            abs(forward_m) > translation_epsilon
            or abs(right_m) > translation_epsilon
            or abs(up_m) > translation_epsilon
            or abs(planar_m) > translation_epsilon
            or move_duration > 0.0
            or move_count > 0
        )
        yaw_only = not has_translation and (turn_count > 0 or abs(yaw_delta_deg) > 0.0 or abs(turn_yaw) > 0.0)
        return {
            "motion_progress": {
                "measurement_scope": "phase_start_relative_uav_motion",
                "interpretation_note": (
                    "forward_m/right_m/up_m/planar_m are UAV motion progress from the current "
                    "phase start; they are not distance-to-target values and not distance-to-landmark values."
                ),
                "forward_m": forward_m,
                "right_m": right_m,
                "up_m": up_m,
                "planar_m": planar_m,
                "yaw_delta_deg": yaw_delta_deg,
                "move_duration_success_sum_sec": move_duration,
                "turn_yaw_success_sum_deg": turn_yaw,
                "tools_executed": dict(tools_executed),
                "has_translation": has_translation,
                "yaw_only_so_far": yaw_only,
            },
        }

    @staticmethod
    def compact_runtime_control_context(runtime_control_context: dict[str, Any] | None) -> dict[str, Any]:
        """Return the short model-facing runtime control capability summary."""

        if not isinstance(runtime_control_context, dict):
            return {}
        available = runtime_control_context.get("available", [])
        status = runtime_control_context.get("status", {})
        contracts = runtime_control_context.get("contracts", {})
        recommendations = runtime_control_context.get("recommendations", {})
        result: dict[str, Any] = {}
        clean_available: list[str] = []
        if isinstance(available, list):
            clean_available = [str(item).strip() for item in available if str(item).strip()]
            if clean_available:
                result["available"] = clean_available
        if isinstance(contracts, dict):
            clean_contracts = {
                str(name).strip(): str(value).strip()
                for name, value in contracts.items()
                if str(name).strip() in clean_available and str(value).strip()
            }
            if clean_contracts:
                result["contracts"] = clean_contracts
        if isinstance(status, dict):
            clean_status = {
                str(name).strip(): str(value).strip()
                for name, value in status.items()
                if str(name).strip() in clean_available and str(value).strip()
            }
            if clean_status:
                result["status"] = clean_status
        if isinstance(recommendations, dict):
            clean_recommendations = {
                str(name).strip(): str(value).strip()
                for name, value in recommendations.items()
                if str(name).strip() in clean_available and str(value).strip()
            }
            if clean_recommendations:
                result["recommendations"] = clean_recommendations
        return result

    @staticmethod
    def _safe_model_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return round(float(value), 3)
        except (TypeError, ValueError):
            return default
