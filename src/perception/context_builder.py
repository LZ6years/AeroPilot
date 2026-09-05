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
from src.plugins import PluginRegistry
from src.plugins.skills import ModelSkillSelector
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.safety_config import UavSafetyConfig
from src.uav.commands import MOVE_BODY_ARGUMENT_SEMANTICS
from src.uav.frames import SIGNED_YAW_SEMANTICS


DEFAULT_MAP_MEMORY_PATH = Path("memory") / "map" / "map_memory.json"


def _compact_phase_neighbor(phase: Any | None, *, status: str) -> dict[str, Any] | None:
    """Return only the adjacent phase identity needed for route continuity."""

    if phase is None:
        return None
    return {
        "index": int(phase.index),
        "name": str(phase.name),
        "goal": str(phase.goal),
        "status": status,
    }


class ObservationContextBuilder:
    def __init__(
        self,
        action_config: UavActionConfig | None = None,
        safety_config: UavSafetyConfig | None = None,
        registry: ContextModuleRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
        skill_selector: ModelSkillSelector | None = None,
        map_memory: MapMemoryStore | None = None,
        map_memory_path: str | Path = DEFAULT_MAP_MEMORY_PATH,
        run_memory: RunMemoryStore | None = None,
    ) -> None:
        self.action_config = action_config or UavActionConfig()
        self.safety_config = safety_config or UavSafetyConfig()
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.composer = ContextComposer(
            registry
            or ContextModuleRegistry.from_config(
                plugin_registry=self.plugin_registry,
                skill_selector=skill_selector,
            )
        )
        self.map_memory = map_memory if map_memory is not None else MapMemoryStore.from_file(map_memory_path)
        self.run_memory = run_memory or RunMemoryStore()

    def semantic_evidence_capabilities(self) -> list[str]:
        """Describe configured evidence without building an observation context."""

        return self.composer.registry.semantic_evidence_capabilities()

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
            phase_neighborhood=self._phase_neighborhood(state),
            semantic_goal=state.semantic_plan.task_goal.to_dict(),
            safety_constraints=self._safety_constraints(state),
            allowed_action_constraints=self._allowed_action_constraints(),
            observation_policy=self._observation_policy(state),
            phase_execution_summary=state.current_phase_execution_summary,
            phase_strategy=state.current_phase_strategy,
            metadata={
                "run_id": state.run_id,
                "current_phase_index": state.current_phase_index,
                "current_phase_key": state.current_phase_key,
            },
        )

    def _current_phase(self, state: MissionState) -> dict[str, Any] | None:
        if state.current_phase_index >= len(state.semantic_plan.phases):
            return None
        return state.semantic_plan.phases[state.current_phase_index].to_dict()

    def _phase_neighborhood(self, state: MissionState) -> dict[str, Any]:
        """Expose adjacent intent without duplicating actionable contracts.

        The current phase remains authoritative. Neighbor goals provide route
        continuity, while their criteria and execution evidence stay hidden so
        an upcoming phase cannot be mistaken for the active phase.
        """

        phases = state.semantic_plan.phases
        current_index = state.current_phase_index
        previous = phases[current_index - 1] if 0 < current_index <= len(phases) else None
        next_phase = phases[current_index + 1] if 0 <= current_index < len(phases) - 1 else None
        return {
            "previous": _compact_phase_neighbor(previous, status="completed"),
            "next": _compact_phase_neighbor(next_phase, status="upcoming"),
        }

    def _safety_constraints(self, state: MissionState) -> dict[str, Any]:
        max_yaw_scan = self.safety_config.max_yaw_scan_extent_deg
        yaw_scan_extent = float(state.yaw_scan_extent_deg)
        remaining = max(0.0, max_yaw_scan - yaw_scan_extent)
        return {
            "hover_required": state.hover_required,
            "yaw_scan_offset_deg": float(state.yaw_scan_offset_deg),
            "yaw_scan_extent_deg": yaw_scan_extent,
            "yaw_scan_remaining_deg": remaining,
            "max_yaw_scan_extent_deg": max_yaw_scan,
            "yaw_scan_budget_exceeded": remaining <= 0.0,
            "yaw_scan_turn_count": int(state.yaw_scan_turn_count),
            "yaw_scan_recharge_count": int(state.yaw_scan_recharge_count),
            "yaw_scan_recharge_distance_m": float(
                self.safety_config.yaw_scan_recharge_distance_m
            ),
        }

    def _allowed_action_constraints(self) -> dict[str, Any]:
        action_config = self.action_config
        safety_config = self.safety_config
        builtin_tools = ["uav.hover", "uav.turn_relative", "uav.move_body"]
        skill_tools = self.plugin_registry.skill_capability_catalog.planner_tools()
        action_schemas = {
            "uav.hover": {
                "args": {},
            },
            "uav.turn_relative": {
                "args": {"yaw_deg": "number"},
                "argument_semantics": {
                    "yaw_deg": SIGNED_YAW_SEMANTICS,
                },
                "limits": {
                    "yaw_deg_min": -abs(safety_config.max_yaw_step_deg),
                    "yaw_deg_max": abs(safety_config.max_yaw_step_deg),
                },
            },
            "uav.move_body": {
                "args_one_of": [
                    {
                        "vx": "number",
                        "vy": "number",
                        "vz": "number",
                        "duration": "number",
                    },
                ],
                "argument_semantics": dict(MOVE_BODY_ARGUMENT_SEMANTICS),
                "limits": {
                    "abs_vx_vy_max": abs(safety_config.max_linear_velocity_mps),
                    "abs_vz_max": abs(safety_config.max_vertical_velocity_mps),
                    "duration_max": safety_config.max_move_duration_sec,
                },
            },
            **self.plugin_registry.skill_capability_catalog.action_schemas(),
        }
        return {
            "planner_tools": [*builtin_tools, *skill_tools],
            "action_schemas": action_schemas,
            "nominal_action_scale": {
                "linear_velocity_mps": action_config.linear_velocity_mps,
                "vertical_velocity_mps": action_config.vertical_velocity_mps,
            },
        }

    def _observation_policy(self, state: MissionState) -> dict[str, Any]:
        return {
            "mode": "automatic",
            "modalities": list(state.observation_modalities),
            "sample_timing": "after_each_phase_window",
            "note": "Do not output observe steps. The runtime samples observations automatically.",
        }
