"""Build model-ready mission context from memory fragments and state."""

from __future__ import annotations

from typing import Any

from src.agent.mission_state import MissionState
from src.memory.context_composer import ContextComposer
from src.memory.flight_memory import FlightMemory
from src.memory.registry import ContextModuleRegistry
from src.memory.schemas import MemorySourceBundle
from src.memory.stores import MapMemoryStore, RunMemoryStore
from src.runtime.configs.action_config import UavActionConfig


class ObservationContextBuilder:
    def __init__(
        self,
        action_config: UavActionConfig | None = None,
        registry: ContextModuleRegistry | None = None,
        map_memory: MapMemoryStore | None = None,
        run_memory: RunMemoryStore | None = None,
    ) -> None:
        self.action_config = action_config or UavActionConfig()
        self.composer = ContextComposer(registry or ContextModuleRegistry.from_config())
        self.map_memory = map_memory or MapMemoryStore()
        self.run_memory = run_memory or RunMemoryStore()

    def build(self, state: MissionState, flight_memory: FlightMemory | None = None) -> dict[str, Any]:
        memory = flight_memory or self._memory_from_state(state)
        task_context = {
            "current_phase": self._current_phase(state),
            "semantic_goal": state.semantic_plan.goal.to_dict(),
            "semantic_plan": state.semantic_plan.to_dict(),
        }
        return self.composer.compose(
            MemorySourceBundle(memory, self.map_memory, self.run_memory),
            task_context,
            safety_constraints={"hover_required": state.hover_required},
            allowed_action_constraints=self._allowed_action_constraints(),
        )

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

    def _memory_from_state(self, state: MissionState) -> FlightMemory:
        memory = FlightMemory(
            run_id=state.run_id,
            map_id="",
            scene_instance_id="",
            agent_id="",
            active_instruction=state.user_instruction,
        )
        memory.update_instruction(state.user_instruction, state.semantic_plan.to_dict())
        for action in state.recent_actions:
            memory.append_action(action.get("tool", ""), action.get("args", {}), action.get("result", {}))
        for observation in state.recent_observations:
            memory.append_observation(observation)
        return memory
