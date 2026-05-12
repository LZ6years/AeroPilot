"""Create short-horizon primitive windows from semantic phases."""

from __future__ import annotations

from src.agent.mission_state import MissionState
from src.agent.schemas import PlannedStep, ShortHorizonPlan
from src.runtime.configs.action_config import UavActionConfig


class ShortHorizonPlanner:
    planner_mode = "rule_based"

    def __init__(self, action_config: UavActionConfig | None = None, auto_takeoff: bool | None = None) -> None:
        self.action_config = action_config or UavActionConfig()
        self.auto_takeoff = auto_takeoff

    def plan_next(self, state: MissionState, context: dict | None = None) -> ShortHorizonPlan:
        phase = state.current_phase_name
        steps: list[PlannedStep] = []
        if phase == "orient_right":
            steps.extend(
                [
                    PlannedStep.tool_step(
                        "uav.turn_relative",
                        {"yaw_deg": self.action_config.yaw_step_deg},
                        "orient toward the right-side clue",
                    ),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase == "orient_left":
            steps.extend(
                [
                    PlannedStep.tool_step(
                        "uav.turn_relative",
                        {"yaw_deg": -self.action_config.yaw_step_deg},
                        "orient toward the left-side clue",
                    ),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase == "search_landmark":
            steps.extend(
                [
                    PlannedStep.tool_step("uav.hover", {}, "stabilize before searching"),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase == "align_to_landmark":
            steps.extend(
                [
                    PlannedStep.tool_step(
                        "uav.turn_relative",
                        {"yaw_deg": self.action_config.yaw_step_deg / 3.0},
                        "small alignment scan without visual confirmation",
                    ),
                    PlannedStep.tool_step("uav.hover", {}, "stabilize after alignment scan"),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase == "approach_landmark":
            steps.extend(
                [
                    PlannedStep.tool_step("uav.hover", {}, "do not approach before visual/depth confirmation"),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase in {"stop", "clarify_if_uncertain"}:
            steps.append(PlannedStep.tool_step("uav.hover", {}, "hold position"))
        elif phase == "observe_area":
            steps.extend(
                [
                    PlannedStep.tool_step("uav.hover", {}, "stabilize for area observation"),
                    PlannedStep.observe_step(),
                ]
            )
        elif phase == "completed":
            steps.append(PlannedStep.tool_step("uav.hover", {}, "mission already completed"))
        else:
            steps.extend(
                [
                    PlannedStep.tool_step("uav.hover", {}, f"unknown phase {phase}; hold safely"),
                    PlannedStep.observe_step(),
                ]
            )
        return ShortHorizonPlan(phase, steps)
