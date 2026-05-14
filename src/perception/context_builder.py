"""Build model-ready mission context from memory fragments and state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.memory.context.composer import ContextComposer
from src.memory.flight_memory import FlightMemory
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.stores import MapMemoryStore, RunMemoryStore
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.safety_config import UavSafetyConfig


DEFAULT_MAP_MEMORY_PATH = Path("memory") / "map" / "map_memory.json"


class ObservationContextBuilder:
    def __init__(
        self,
        action_config: UavActionConfig | None = None,
        safety_config: UavSafetyConfig | None = None,
        registry: ContextModuleRegistry | None = None,
        map_memory: MapMemoryStore | None = None,
        map_memory_path: str | Path = DEFAULT_MAP_MEMORY_PATH,
        run_memory: RunMemoryStore | None = None,
    ) -> None:
        self.action_config = action_config or UavActionConfig()
        self.safety_config = safety_config or UavSafetyConfig()
        self.composer = ContextComposer(registry or ContextModuleRegistry.from_config())
        self.map_memory = map_memory if map_memory is not None else MapMemoryStore.from_file(map_memory_path)
        self.run_memory = run_memory or RunMemoryStore()

    def build(self, state: MissionState, flight_memory: FlightMemory | None = None) -> dict[str, Any]:
        if flight_memory is None:
            raise ValueError("ObservationContextBuilder requires a FlightMemory manifest; MissionState is not a memory source")
        memory = flight_memory
        stage_context = self._stage_context_from_state(state, memory)
        return self.composer.compose(
            MemorySourceBundle(memory, self.map_memory, self.run_memory),
            stage_context,
        )

    def _stage_context_from_state(self, state: MissionState, memory: FlightMemory) -> StageContextInput:
        return StageContextInput(
            active_instruction=memory.active_instruction,
            semantic_plan=state.semantic_plan.to_dict(),
            current_phase=self._current_phase(state),
            semantic_goal=state.semantic_plan.goal.to_dict(),
            safety_constraints=self._safety_constraints(state),
            allowed_action_constraints=self._allowed_action_constraints(),
            observation_policy=self._observation_policy(),
            phase_transition_policy=self._phase_transition_policy(state),
            metadata={
                "run_id": state.run_id,
                "current_phase_index": state.current_phase_index,
            },
        )

    def _current_phase(self, state: MissionState) -> dict[str, Any] | None:
        if state.current_phase_index >= len(state.semantic_plan.phases):
            return None
        return state.semantic_plan.phases[state.current_phase_index].to_dict()

    def _safety_constraints(self, state: MissionState) -> dict[str, Any]:
        max_yaw_scan = self.safety_config.max_yaw_scan_accum_deg
        yaw_scan_accum = float(state.yaw_scan_accum_deg)
        remaining = max(0.0, max_yaw_scan - yaw_scan_accum)
        return {
            "hover_required": state.hover_required,
            "yaw_scan_accum_deg": yaw_scan_accum,
            "yaw_scan_remaining_deg": remaining,
            "max_yaw_scan_accum_deg": max_yaw_scan,
            "yaw_scan_budget_exceeded": remaining <= 0.0,
        }

    def _allowed_action_constraints(self) -> dict[str, Any]:
        action_config = self.action_config
        safety_config = self.safety_config
        return {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body"],
            "action_schemas": {
                "uav.hover": {
                    "args": {},
                },
                "uav.turn_relative": {
                    "args": {"yaw_deg": "number"},
                    "limits": {
                        "yaw_deg_min": -abs(safety_config.max_yaw_step_deg),
                        "yaw_deg_max": abs(safety_config.max_yaw_step_deg),
                    },
                },
                "uav.move_body": {
                    "args_one_of": [
                        {
                            "direction": "forward|backward|left|right|up|down",
                            "duration": "number",
                        },
                        {
                            "vx": "number",
                            "vy": "number",
                            "vz": "number",
                            "duration": "number",
                        },
                    ],
                    "limits": {
                        "abs_vx_vy_max": abs(safety_config.max_linear_velocity_mps),
                        "abs_vz_max": abs(safety_config.max_vertical_velocity_mps),
                        "duration_max": safety_config.max_move_duration_sec,
                    },
                },
            },
            "forbidden_tools": [
                "uav.connect",
                "uav.takeoff",
                "uav.land",
                "uav.set_pose",
                "uav.get_pose",
                "uav.get_frames",
                "env.observe",
            ],
            "lifecycle_tools_excluded": ["uav.connect", "uav.takeoff", "uav.land"],
            "yaw_step_deg": safety_config.max_yaw_step_deg,
            "linear_velocity_mps": safety_config.max_linear_velocity_mps,
            "vertical_velocity_mps": safety_config.max_vertical_velocity_mps,
            "move_duration_sec": safety_config.max_move_duration_sec,
            "safety_limits": safety_config.to_dict(),
            "nominal_action_scale": action_config.to_dict(),
        }

    def _observation_policy(self) -> dict[str, Any]:
        return {
            "mode": "automatic",
            "modalities": ["rgb", "depth", "pose"],
            "sample_timing": "after_each_phase_window",
            "note": "Do not output observe steps. The runtime samples observations automatically.",
        }

    def _phase_transition_policy(self, state: MissionState) -> dict[str, Any]:
        return {
            "mode": "model_suggested_runtime_applied",
            "decisions": ["continue_current_phase", "advance_to_next_phase", "complete_task"],
            "current_phase_window_count": state.current_phase_window_count,
            "max_phase_window_count": state.max_phase_window_count,
            "note": "Output phase_transition in the action plan. Do not output target phase names or indexes.",
        }
