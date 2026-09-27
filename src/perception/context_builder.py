"""Build model-ready mission context from memory fragments and state."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
from typing import Any

from src.agent.mission_state import MissionState
from src.memory.context.composer import ContextComposer
from src.memory.flight_memory import FlightMemory
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import MemorySourceBundle, TaskContextInput
from src.memory.stores import MapMemoryStore, RunMemoryStore
from src.plugins import PluginRegistry
from src.runtime.configs.safety_config import UavSafetyConfig


DEFAULT_MAP_MEMORY_PATH = Path("memory") / "map" / "map_memory.json"


class ObservationContextBuilder:
    def __init__(
        self,
        safety_config: UavSafetyConfig | None = None,
        registry: ContextModuleRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
        map_memory: MapMemoryStore | None = None,
        map_memory_path: str | Path = DEFAULT_MAP_MEMORY_PATH,
        run_memory: RunMemoryStore | None = None,
    ) -> None:
        self.safety_config = safety_config or UavSafetyConfig()
        self.plugin_registry = plugin_registry or PluginRegistry.from_config()
        self.composer = ContextComposer(
            registry
            or ContextModuleRegistry.from_config(
                plugin_registry=self.plugin_registry,
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
        # Custom readers retain their existing structural interface. The scope
        # covers only this build, never later observations or state changes.
        scope = getattr(memory.memory_reader, "read_scope", None)
        with scope() if callable(scope) else nullcontext():
            stage_context = self._stage_context_from_state(state, memory)
            return self.composer.compose(
                MemorySourceBundle(memory, self.map_memory, self.run_memory),
                stage_context,
            )

    def _stage_context_from_state(self, state: MissionState, memory: FlightMemory) -> TaskContextInput:
        return TaskContextInput(
            active_instruction=state.user_instruction, protocol_context=state.protocol_context,
            task_plan=state.task_plan.to_dict() if state.task_plan else None,
            loaded_knowledge=state.loaded_knowledge, motion_summary=state.motion_summary,
            safety_constraints=self._safety_constraints(state),
            observation_policy={"mode": "automatic", "modalities": list(state.observation_modalities),
                                "sample_timing": "after_each_physical_window"},
            metadata={"run_id": state.run_id, "physical_window_count": state.physical_window_count},
        )

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
