"""Build model-ready mission context from raw observations and state."""

from __future__ import annotations

from typing import Any

from src.agent.mission_state import MissionState
from src.runtime.configs.action_config import UavActionConfig


class ObservationContextBuilder:
    def __init__(self, action_config: UavActionConfig | None = None) -> None:
        self.action_config = action_config or UavActionConfig()

    def build(self, state: MissionState) -> dict[str, Any]:
        latest_observation = state.recent_observations[-1] if state.recent_observations else None
        return {
            "instruction": state.user_instruction,
            "current_phase": self._current_phase(state),
            "goal": state.semantic_plan.goal.to_dict(),
            "latest_observation": latest_observation,
            "recent_actions": list(state.recent_actions),
            "safety": {"hover_required": state.hover_required},
            "allowed_action_constraints": self._allowed_action_constraints(),
        }

    def _current_phase(self, state: MissionState) -> dict[str, Any] | None:
        if state.current_phase_index >= len(state.semantic_plan.phases):
            return None
        return state.semantic_plan.phases[state.current_phase_index].to_dict()

    def _allowed_action_constraints(self) -> dict[str, Any]:
        return {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body", "env.observe"],
            "lifecycle_tools_excluded": ["uav.connect", "uav.takeoff", "uav.land"],
            "yaw_step_deg": self.action_config.yaw_step_deg,
            "linear_velocity_mps": self.action_config.linear_velocity_mps,
            "vertical_velocity_mps": self.action_config.vertical_velocity_mps,
            "move_duration_sec": self.action_config.move_duration_sec,
        }
