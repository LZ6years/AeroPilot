"""Build binary-free Model Context V2 envelopes and state checkpoints."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from src.memory.context.contracts import (
    ContextModuleResult,
    InstructionEnvelope,
    MemorySourceBundle,
    MissionStateView,
    StateCheckpoint,
    TaskContextInput,
    assert_binary_free,
    checkpoint_fingerprint,
)
from src.memory.context.documents import ContextDocuments
from src.memory.context.registry import ContextModuleRegistry
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore
from src.uav.config import UavSafetyConfig


DEFAULT_MAP_MEMORY_PATH = Path("memory") / "map" / "map_memory.json"


class ModelContextBuilder:
    """Translate mission state into the two V2 context lifetimes.

    Plugin discovery and request projection are intentionally absent.  The
    composition root supplies a fully configured registry and static catalog.
    """

    def __init__(
        self,
        *,
        safety_config: UavSafetyConfig | None = None,
        registry: ContextModuleRegistry,
        documents: ContextDocuments | None = None,
        knowledge_catalog: list[dict[str, Any]] | None = None,
        map_memory: MapMemoryStore | None = None,
        map_memory_path: str | Path = DEFAULT_MAP_MEMORY_PATH,
        run_memory: Any = None,
    ) -> None:
        self.safety_config = safety_config or UavSafetyConfig()
        self.registry = registry
        self.documents = documents or ContextDocuments()
        self.knowledge_catalog = tuple(copy.deepcopy(knowledge_catalog or []))
        self.map_memory = (
            map_memory
            if map_memory is not None
            else MapMemoryStore.from_file(map_memory_path)
        )
        # Kept as an optional source field until third-party ContextModules no
        # longer accept MemorySourceBundle.run_memory. Core Runtime has no run
        # memory store and never creates an implicit placeholder instance.
        self.run_memory = run_memory

    def semantic_evidence_capabilities(self) -> list[str]:
        return self.registry.semantic_evidence_capabilities()

    def build_instruction_envelope(
        self, state: MissionStateView, memory: FlightMemory
    ) -> InstructionEnvelope:
        return InstructionEnvelope(
            instruction_id=memory.active_instruction_id,
            protocol_context=state.protocol_context,
            identity=copy.deepcopy(self.documents.identity),
            user_preferences=copy.deepcopy(self.documents.user_preferences),
            knowledge_catalog=tuple(copy.deepcopy(self.knowledge_catalog)),
            static_module_guidance=tuple(
                {
                    "name": name,
                    "summary": description.get("summary", ""),
                    "model_fields": copy.deepcopy(
                        description.get("model_fields", {})
                    ),
                    "semantic_evidence": copy.deepcopy(
                        description.get("semantic_evidence", [])
                    ),
                }
                for name, description in self.registry.describe_modules().items()
            ),
        )

    def build_checkpoint(
        self,
        state: MissionStateView,
        memory: FlightMemory,
        *,
        checkpoint_id: str,
        reason: str,
    ) -> tuple[StateCheckpoint, tuple[dict[str, Any], ...]]:
        scope = getattr(memory.memory_reader, "read_scope", None)
        with scope() if callable(scope) else nullcontext():
            stage = self._stage_context_from_state(state)
            results = self.registry.build_fragments(
                MemorySourceBundle(memory, self.map_memory, self.run_memory), stage
            )
        fragments = tuple(result.fragment for result in results)
        loaded = tuple(copy.deepcopy(state.loaded_knowledge))
        task_plan = state.task_plan.to_dict() if state.task_plan else None
        observation_id = str(state.latest_observation.get("observation_id") or "") or None
        fingerprint = checkpoint_fingerprint(
            observation_id, task_plan, loaded, fragments
        )
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            reason=reason,  # type: ignore[arg-type]
            observation_id=observation_id,
            task_plan=task_plan,
            loaded_knowledge=loaded,
            fragments=fragments,
            fingerprint=fingerprint,
        )
        assert_binary_free(checkpoint.to_dict())
        return checkpoint, tuple(result.audit.to_dict() for result in results)

    def _stage_context_from_state(self, state: MissionStateView) -> TaskContextInput:
        return TaskContextInput(
            active_instruction=state.user_instruction,
            protocol_context=state.protocol_context,
            task_plan=state.task_plan.to_dict() if state.task_plan else None,
            loaded_knowledge=copy.deepcopy(state.loaded_knowledge),
            motion_summary=copy.deepcopy(state.motion_summary),
            safety_constraints=self._safety_constraints(state),
            observation_policy={
                "mode": "automatic",
                "modalities": list(state.observation_modalities),
                "sample_timing": "after_each_physical_window",
            },
            metadata={
                "run_id": state.run_id,
                "physical_window_count": state.physical_window_count,
            },
        )

    def _safety_constraints(self, state: MissionStateView) -> dict[str, Any]:
        maximum = self.safety_config.max_yaw_scan_extent_deg
        extent = float(state.yaw_scan_extent_deg)
        remaining = max(0.0, maximum - extent)
        return {
            "hover_required": state.hover_required,
            "yaw_scan_offset_deg": float(state.yaw_scan_offset_deg),
            "yaw_scan_extent_deg": extent,
            "yaw_scan_remaining_deg": remaining,
            "max_yaw_scan_extent_deg": maximum,
            "yaw_scan_budget_exceeded": remaining <= 0.0,
            "yaw_scan_turn_count": int(state.yaw_scan_turn_count),
            "yaw_scan_recharge_count": int(state.yaw_scan_recharge_count),
            "yaw_scan_recharge_distance_m": float(
                self.safety_config.yaw_scan_recharge_distance_m
            ),
        }
