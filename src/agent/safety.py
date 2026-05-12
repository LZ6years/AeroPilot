"""Safety checks and clipping for short-horizon primitive steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.schemas import PlannedStep, ShortHorizonPlan
from src.runtime.configs.action_config import UavActionConfig


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

    def __init__(self, action_config: UavActionConfig | None = None) -> None:
        self.action_config = action_config or UavActionConfig()

    def review(self, plan: ShortHorizonPlan, state: Any | None = None) -> SafetyDecision:
        safe_steps: list[PlannedStep] = []
        interventions: list[dict] = []
        turn_scan_already_used = self._last_successful_tool(state) == "uav.turn_relative"
        for step in plan.steps:
            if step.kind == "tool" and step.tool == "uav.turn_relative" and turn_scan_already_used:
                safe_step = PlannedStep.tool_step(
                    "uav.hover",
                    {},
                    "hold position because consecutive yaw scans require fresh visual confirmation",
                )
                safe_steps.append(safe_step)
                interventions.append(
                    {
                        "tool": step.tool,
                        "replacement": "uav.hover",
                        "reason": "consecutive_yaw_scan_without_visual_confirmation",
                    }
                )
                continue
            safe_step, intervention = self._review_step(step)
            safe_steps.append(safe_step)
            if intervention:
                interventions.append(intervention)
            if safe_step.kind == "tool" and safe_step.tool == "uav.turn_relative":
                turn_scan_already_used = True
        return SafetyDecision(ShortHorizonPlan(plan.phase, safe_steps), interventions)

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
        original = float(args.get("yaw_deg", self.action_config.yaw_step_deg))
        limit = abs(self.action_config.yaw_step_deg)
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
            clipped_duration = min(original_duration, self.action_config.move_duration_sec)
            args["duration"] = clipped_duration
            if clipped_duration != original_duration:
                interventions["duration"] = {"original": original_duration, "clipped": clipped_duration}
        for name in ("vx", "vy"):
            if name in args:
                original = float(args[name])
                limit = abs(self.action_config.linear_velocity_mps)
                clipped = max(-limit, min(limit, original))
                args[name] = clipped
                if clipped != original:
                    interventions[name] = {"original": original, "clipped": clipped}
        if "vz" in args:
            original = float(args["vz"])
            limit = abs(self.action_config.vertical_velocity_mps)
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

    @staticmethod
    def _last_successful_tool(state: Any | None) -> str | None:
        if state is None:
            return None
        for action in reversed(getattr(state, "recent_actions", [])):
            if action.get("result", {}).get("ok") is True:
                return action.get("tool")
        return None
