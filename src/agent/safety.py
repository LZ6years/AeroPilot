"""Safety checks and clipping for short-horizon primitive steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.runtime.configs.safety_config import UavSafetyConfig


class SafetyViolation(ValueError):
    """Raised when a short-horizon step is not allowed."""


@dataclass(frozen=True)
class SafetyDecision:
    plan: ShortHorizonPlan
    interventions: list[dict] = field(default_factory=list)


class SafetySupervisor:
    ALLOWED_TOOLS = {
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
    }

    def __init__(self, safety_config: UavSafetyConfig | None = None) -> None:
        self.safety_config = safety_config or UavSafetyConfig()

    def review(self, plan: ShortHorizonPlan, state: Any | None = None) -> SafetyDecision:
        safe_steps: list[PlannedStep] = []
        interventions: list[dict] = []
        yaw_scan_accum = float(getattr(state, "yaw_scan_accum_deg", 0.0) or 0.0)
        for step in plan.steps:
            if step.kind == "tool" and step.tool == "uav.turn_relative":
                safe_step, intervention = self._review_step(step)
                safe_yaw = abs(float(safe_step.args.get("yaw_deg", 0.0)))
                if yaw_scan_accum + safe_yaw > self.safety_config.max_yaw_scan_accum_deg:
                    safe_step = PlannedStep.tool_step(
                        "uav.hover",
                        {},
                        "hold position because yaw scan budget is exhausted",
                    )
                    safe_steps.append(safe_step)
                    interventions.append(
                        {
                            "tool": step.tool,
                            "replacement": "uav.hover",
                            "reason": "yaw_scan_budget_exceeded",
                            "yaw_scan_accum_deg": yaw_scan_accum,
                            "requested_yaw_deg": abs(float(step.args.get("yaw_deg", self.safety_config.max_yaw_step_deg))),
                            "safe_yaw_deg": safe_yaw,
                            "max_yaw_scan_accum_deg": self.safety_config.max_yaw_scan_accum_deg,
                        }
                    )
                    continue
                yaw_scan_accum += abs(float(safe_step.args.get("yaw_deg", 0.0)))
                safe_steps.append(safe_step)
                if intervention:
                    interventions.append(intervention)
                continue
            safe_step, intervention = self._review_step(step)
            safe_steps.append(safe_step)
            if intervention:
                interventions.append(intervention)
        return SafetyDecision(
            ShortHorizonPlan(
                plan.phase,
                safe_steps,
                phase_transition=plan.phase_transition,
                action_window=plan.action_window,
                agent_messages=plan.agent_messages,
            ),
            interventions,
        )

    def _review_step(self, step: PlannedStep) -> tuple[PlannedStep, dict | None]:
        if step.kind == "observe":
            return step, None
        if step.kind != "tool" or not step.tool:
            raise SafetyViolation(f"Unsupported planned step kind: {step.kind}")
        if step.tool not in self.ALLOWED_TOOLS:
            raise SafetyViolation(f"Tool is not allowed in short-horizon plan: {step.tool}")
        if step.tool == "uav.turn_relative":
            return self._clip_turn(step)
        if step.tool == "uav.move_body":
            return self._clip_move_body(step)
        return step, None

    def _clip_turn(self, step: PlannedStep) -> tuple[PlannedStep, dict | None]:
        args = dict(step.args)
        original = float(args.get("yaw_deg", self.safety_config.max_yaw_step_deg))
        limit = abs(self.safety_config.max_yaw_step_deg)
        clipped = max(-limit, min(limit, original))
        args["yaw_deg"] = clipped
        if clipped == original:
            return PlannedStep.tool_step(step.tool or "", args, step.reason), None
        return (
            PlannedStep.tool_step(step.tool or "", args, step.reason),
            {"tool": step.tool, "field": "yaw_deg", "original": original, "clipped": clipped},
        )

    def _clip_move_body(self, step: PlannedStep) -> tuple[PlannedStep, dict | None]:
        args = dict(step.args)
        interventions: dict[str, dict] = {}
        if "duration" in args:
            original_duration = float(args["duration"])
            clipped_duration = min(original_duration, self.safety_config.max_move_duration_sec)
            args["duration"] = clipped_duration
            if clipped_duration != original_duration:
                interventions["duration"] = {"original": original_duration, "clipped": clipped_duration}
        for name in ("vx", "vy"):
            if name in args:
                original = float(args[name])
                limit = abs(self.safety_config.max_linear_velocity_mps)
                clipped = max(-limit, min(limit, original))
                args[name] = clipped
                if clipped != original:
                    interventions[name] = {"original": original, "clipped": clipped}
        if "vz" in args:
            original = float(args["vz"])
            limit = abs(self.safety_config.max_vertical_velocity_mps)
            clipped = max(-limit, min(limit, original))
            args["vz"] = clipped
            if clipped != original:
                interventions["vz"] = {"original": original, "clipped": clipped}
        if interventions:
            return (
                PlannedStep.tool_step(step.tool or "", args, step.reason),
                {"tool": step.tool, "fields": interventions},
            )
        return PlannedStep.tool_step(step.tool or "", args, step.reason), None
