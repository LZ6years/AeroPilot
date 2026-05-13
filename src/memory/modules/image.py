"""RGB/depth metadata context module."""

from __future__ import annotations

from typing import Any

from src.memory.modules.base import ContextModule
from src.memory.schemas import BudgetSpec, ContextFragment, MemorySourceBundle


class DefaultImageContextModule(ContextModule):
    name = "image"
    priority = 80
    max_items = 3

    def build(self, sources: MemorySourceBundle, task_context: dict[str, Any]) -> ContextFragment:
        observations = sources.flight_memory.recent_observations[-self.max_items :]
        latest = observations[-1] if observations else {}
        frames = latest.get("frames", {}) if isinstance(latest, dict) else {}
        facts = []
        warnings = []
        if frames:
            facts.append(
                {
                    "type": "available_modalities",
                    "value": sorted(frames.keys()),
                    "source": self.__class__.__name__,
                }
            )
        else:
            warnings.append("no_recent_image_frames")
        for name, frame in frames.items() if isinstance(frames, dict) else []:
            if isinstance(frame, dict) and frame.get("mock") is True:
                warnings.append(f"{name}_frame_is_mock")
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "latest_frame_metadata": frames,
                "recent_observation_sources": [item.get("source") for item in observations if isinstance(item, dict)],
            },
            facts=facts,
            risks=[{"type": "visual_evidence_missing", "source": self.__class__.__name__}] if not frames else [],
            confidence="medium" if frames else "low",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=300, max_items=self.max_items),
            warnings=warnings,
            metadata={"observation_count": len(observations)},
        )
