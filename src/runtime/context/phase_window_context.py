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
from src.uav import clean_pose_for_model


@dataclass(frozen=True)
class PhaseWindowContext:
    """Model/debug payloads prepared for a single phase window."""

    planner_payload: dict[str, Any]
    model_context: dict[str, Any]
    phase_metrics: list[dict[str, Any]]
    phase_completion_evaluation: dict[str, Any]
    phase_progress_evidence: dict[str, Any]
    latest_observation_summary: dict[str, Any]


class PhaseWindowContextBuilder:
    """Compose phase-window evidence on top of memory-derived planner context."""

    def build(
        self,
        state: MissionState,
        base_planner_context: dict[str, Any],
        observation_before: dict[str, Any],
    ) -> PhaseWindowContext:
        """Return the complete model-facing context for the current phase window."""

        metrics = self.current_phase_metrics(state)
        completion = evaluate_phase_completion(metrics, state.current_phase_execution_summary)
        progress = self.phase_progress_evidence(state)
        observation_summary = self.clean_observation_for_model(observation_before)
        planner_payload = dict(base_planner_context)
        planner_payload.update(
            {
                "phase_metrics": metrics,
                "phase_completion_evaluation": completion,
                "phase_progress_evidence": progress,
                "latest_observation_summary": observation_summary,
            }
        )
        return PhaseWindowContext(
            planner_payload=planner_payload,
            model_context=sanitize_context_for_model_text(planner_payload),
            phase_metrics=metrics,
            phase_completion_evaluation=completion,
            phase_progress_evidence=progress,
            latest_observation_summary=observation_summary,
        )

    @staticmethod
    def current_phase_metrics(state: MissionState) -> list[dict[str, Any]]:
        """Return metric contracts for the currently active semantic phase."""

        if state.current_phase_index >= len(state.semantic_plan.phases):
            return []
        return [dict(item) for item in state.semantic_plan.phases[state.current_phase_index].metrics]

    def phase_progress_evidence(self, state: MissionState) -> dict[str, Any]:
        """Summarize physical phase progress separately from visual comparison.

        This is model-facing evidence, not a transition validator. It helps the
        coupled planner avoid treating image scale or centering changes as proof
        of physical approach when the runtime has only recorded yaw corrections.
        """

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
        phase = (
            state.semantic_plan.phases[state.current_phase_index]
            if state.current_phase_index < len(state.semantic_plan.phases)
            else None
        )
        done_when = list(phase.done_when) if phase is not None else []
        support = [self._done_when_motion_support(condition, has_translation) for condition in done_when]
        return {
            "motion_progress": {
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
            "done_when_support": support,
            "visual_progress": {
                "image_comparison_is_auxiliary": True,
                "note": "Image scale, centering, or occlusion changes can support visual evidence but do not prove physical translation by themselves.",
            },
            "model_guidance": (
                "Use image comparison as visual evidence, but do not claim physical approach progress when "
                "motion_progress shows yaw-only behavior."
            ),
        }

    @staticmethod
    def clean_observation_for_model(observation: dict[str, Any]) -> dict[str, Any]:
        """Return a model-facing observation summary with normalized pose fields."""

        data = dict(observation) if isinstance(observation, dict) else {}
        pose = data.get("pose")
        if isinstance(pose, dict):
            data["pose"] = clean_pose_for_model(pose)
        return data

    @staticmethod
    def _safe_model_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return round(float(value), 3)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _done_when_requires_translation(condition: str) -> bool:
        text = str(condition).lower()
        translation_terms = (
            "move",
            "moved",
            "moving",
            "approach",
            "approached",
            "closer",
            "near",
            "arrive",
            "arrival",
            "reach",
            "position",
            "distance",
            "follow",
            "standoff",
            "proximity",
            "travel",
            "移动",
            "前进",
            "靠近",
            "接近",
            "到达",
            "抵达",
            "距离",
            "保持距离",
            "跟随",
            "位置",
        )
        return any(term in text for term in translation_terms)

    def _done_when_motion_support(self, condition: str, has_translation: bool) -> dict[str, Any]:
        requires_translation = self._done_when_requires_translation(condition)
        support: dict[str, Any] = {
            "condition": str(condition),
            "requires_translation": requires_translation,
            "supported_by_motion": has_translation if requires_translation else None,
        }
        if requires_translation and not has_translation:
            support["note"] = "No successful move_body or translation progress has been recorded in this phase."
        elif requires_translation:
            support["note"] = "Runtime motion summary contains translation evidence for this condition."
        else:
            support["note"] = "This condition is not primarily a physical-translation condition."
        return support
