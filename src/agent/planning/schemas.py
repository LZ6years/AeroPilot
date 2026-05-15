"""Schemas for short-horizon action planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.planning.phase_transition import PhaseTransition


@dataclass(frozen=True)
class PlannedStep:
    kind: str
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    modalities: list[str] = field(default_factory=lambda: ["rgb", "depth", "pose"])
    reason: str = ""

    @classmethod
    def tool_step(cls, tool: str, args: dict[str, Any] | None = None, reason: str = "") -> "PlannedStep":
        return cls(kind="tool", tool=tool, args=dict(args or {}), reason=reason)

    @classmethod
    def observe_step(
        cls, modalities: list[str] | None = None, reason: str = "refresh raw observation"
    ) -> "PlannedStep":
        return cls(kind="observe", modalities=list(modalities or ["rgb", "depth", "pose"]), reason=reason)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannedStep":
        kind = str(data.get("kind", "")).strip()
        if kind == "observe":
            step = cls.observe_step(
                modalities=[str(item) for item in data.get("modalities", ["rgb", "depth", "pose"])],
                reason=str(data.get("reason", "refresh raw observation")),
            )
            if "tool" in data:
                return cls(
                    kind="observe",
                    tool=str(data.get("tool", "")).strip(),
                    modalities=step.modalities,
                    reason=step.reason,
                )
            return step
        if kind == "tool":
            return cls.tool_step(
                str(data.get("tool", "")).strip(),
                dict(data.get("args", {})),
                str(data.get("reason", "")),
            )
        raise ValueError(f"Unsupported planned step kind: {kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "args": dict(self.args),
            "modalities": list(self.modalities),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ShortHorizonPlan:
    phase: str
    steps: list[PlannedStep]
    phase_transition: PhaseTransition = field(default_factory=PhaseTransition.default_advance)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, require_phase_transition: bool = False) -> "ShortHorizonPlan":
        if require_phase_transition and "phase_transition" not in data:
            raise ValueError("ShortHorizonPlan response requires phase_transition")
        transition_data = data.get("phase_transition")
        transition = (
            PhaseTransition.from_dict(dict(transition_data))
            if transition_data is not None
            else PhaseTransition.default_advance()
        )
        return cls(
            phase=str(data.get("phase", "")).strip(),
            steps=[PlannedStep.from_dict(dict(item)) for item in data.get("steps", [])],
            phase_transition=transition,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "steps": [step.to_dict() for step in self.steps],
            "phase_transition": self.phase_transition.to_dict(),
        }
