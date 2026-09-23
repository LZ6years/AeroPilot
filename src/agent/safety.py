"""Safety checks and clipping for short-horizon primitive steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.planning.schemas import PlannedStep
from src.runtime.configs.safety_config import UavSafetyConfig

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class SafetyViolation(ValueError):
    """Raised when a short-horizon step is not allowed."""


@dataclass(frozen=True)
class PhysicalSafetyDecision:
    """Reviewed physical steps, independent of any semantic plan or Phase."""

    steps: list[PlannedStep]
    interventions: list[dict] = field(default_factory=list)


SAFETY_OPERATING_PRINCIPLES = (
    "Prefer conservative action sizes when obstacles, people, occlusions, narrow spaces, or uncertain free space are active.",
    "Use safety to shape or clip motion, not to replace the primary navigation objective.",
    "Do not assume a path is clear only because no obstacle metadata exists.",
)


class SafetySupervisor:
    """Review planned primitive steps against UAV safety limits.

    Obstacle and proximity awareness belongs to this safety boundary rather than
    planner skill selection. The current implementation enforces hard primitive
    limits only; perception-driven obstacle rules can build on
    SAFETY_OPERATING_PRINCIPLES later without exposing safety prose as a flight
    strategy skill.
    """

    ALLOWED_TOOLS = {
        "uav.hover",
        "uav.turn_relative",
        "uav.move_body",
    }

    def __init__(self, safety_config: UavSafetyConfig | None = None) -> None:
        self.safety_config = safety_config or UavSafetyConfig()

    def review_steps(
        self,
        steps: list[PlannedStep],
        state: Any | None = None,
        *,
        registry: "ToolRegistry | None" = None,
    ) -> PhysicalSafetyDecision:
        """Review one ordered batch using the existing physical limits.

        Neither a mutable plan nor a phase is required. The legacy plan entry
        delegates here during migration so there is only one safety algorithm.
        """
        safe_steps: list[PlannedStep] = []
        interventions: list[dict] = []
        yaw_scan_offset = float(getattr(state, "yaw_scan_offset_deg", 0.0) or 0.0)
        yaw_scan_extent = float(getattr(state, "yaw_scan_extent_deg", 0.0) or 0.0)
        for step_index, step in enumerate(steps):
            if step.kind == "tool" and step.tool == "uav.turn_relative":
                safe_step, intervention = self._review_step(step, registry)
                safe_yaw = float(safe_step.args.get("yaw_deg", 0.0))
                projected_offset = yaw_scan_offset + safe_yaw
                projected_extent = max(yaw_scan_extent, abs(projected_offset))
                if projected_extent > self.safety_config.max_yaw_scan_extent_deg:
                    safe_step = PlannedStep.tool_step(
                        "uav.hover",
                        {},
                        "hold position because yaw scan budget is exhausted",
                    )
                    safe_steps.append(safe_step)
                    interventions.append(
                        {
                            "step_index": step_index,
                            "tool": step.tool,
                            "replacement": "uav.hover",
                            "reason": "yaw_scan_budget_exceeded",
                            "yaw_scan_offset_deg": yaw_scan_offset,
                            "yaw_scan_extent_deg": yaw_scan_extent,
                            "requested_yaw_deg": float(
                                step.args.get("yaw_deg", self.safety_config.max_yaw_step_deg)
                            ),
                            "safe_yaw_deg": safe_yaw,
                            "projected_yaw_scan_extent_deg": projected_extent,
                            "max_yaw_scan_extent_deg": self.safety_config.max_yaw_scan_extent_deg,
                        }
                    )
                    continue
                yaw_scan_offset = projected_offset
                yaw_scan_extent = projected_extent
                safe_steps.append(safe_step)
                if intervention:
                    interventions.append({"step_index": step_index, **intervention})
                continue
            safe_step, intervention = self._review_step(step, registry)
            safe_steps.append(safe_step)
            if intervention:
                interventions.append({"step_index": step_index, **intervention})
        return PhysicalSafetyDecision(safe_steps, interventions)

    def _review_step(
        self,
        step: PlannedStep,
        registry: "ToolRegistry | None",
    ) -> tuple[PlannedStep, dict | None]:
        if step.kind == "observe":
            return step, None
        if step.kind != "tool" or not step.tool:
            raise SafetyViolation(f"Unsupported planned step kind: {step.kind}")
        if registry is not None:
            spec = registry.get(step.tool)
            if spec.runtime.effect != "physical":
                raise SafetyViolation(
                    f"Non-physical tool reached physical safety review: {step.tool}"
                )
        elif step.tool not in self.ALLOWED_TOOLS:
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
