"""Schemas for short-horizon action planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.messages import AgentMessage
from src.agent.phase import PhaseTransition

ACTION_WINDOW_SCOPE_CURRENT_PHASE = "current_phase_only"


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
class ActionWindow:
    """Model declaration for why a group of steps may run before the next observation."""

    intent: str
    scope: str = ACTION_WINDOW_SCOPE_CURRENT_PHASE
    requires_intermediate_observation: bool = False
    expected_effect: str = ""

    @classmethod
    def default(cls) -> "ActionWindow":
        return cls(
            intent="execute one short action window for the current phase",
            scope=ACTION_WINDOW_SCOPE_CURRENT_PHASE,
            requires_intermediate_observation=False,
            expected_effect="produce measurable progress for the current phase",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionWindow":
        return cls(
            intent=str(data.get("intent", "")).strip(),
            scope=str(data.get("scope", ACTION_WINDOW_SCOPE_CURRENT_PHASE)).strip()
            or ACTION_WINDOW_SCOPE_CURRENT_PHASE,
            requires_intermediate_observation=bool(data.get("requires_intermediate_observation", False)),
            expected_effect=str(data.get("expected_effect", "")).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "scope": self.scope,
            "requires_intermediate_observation": self.requires_intermediate_observation,
            "expected_effect": self.expected_effect,
        }


@dataclass(frozen=True)
class ShortHorizonPlan:
    phase: str
    steps: list[PlannedStep]
    phase_transition: PhaseTransition = field(default_factory=PhaseTransition.default_advance)
    action_window: ActionWindow = field(default_factory=ActionWindow.default)
    agent_messages: list[AgentMessage] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        require_phase_transition: bool = False,
        require_action_window: bool = False,
        require_agent_messages: bool = False,
    ) -> "ShortHorizonPlan":
        if require_phase_transition and "phase_transition" not in data:
            raise ValueError("ShortHorizonPlan response requires phase_transition")
        if require_action_window and "action_window" not in data:
            raise ValueError("ShortHorizonPlan response requires action_window")
        if require_agent_messages and "agent_messages" not in data:
            raise ValueError("ShortHorizonPlan response requires agent_messages")
        transition_data = data.get("phase_transition")
        transition = (
            PhaseTransition.from_dict(dict(transition_data))
            if transition_data is not None
            else PhaseTransition.default_advance()
        )
        action_window_data = data.get("action_window")
        action_window = (
            ActionWindow.from_dict(dict(action_window_data))
            if action_window_data is not None
            else ActionWindow.default()
        )
        messages_data = data.get("agent_messages", [])
        if messages_data is None:
            messages_data = []
        if not isinstance(messages_data, list):
            raise ValueError("ShortHorizonPlan agent_messages must be a list")
        return cls(
            phase=str(data.get("phase", "")).strip(),
            steps=[PlannedStep.from_dict(dict(item)) for item in data.get("steps", [])],
            phase_transition=transition,
            action_window=action_window,
            agent_messages=[AgentMessage.from_dict(dict(item)) for item in messages_data],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "steps": [step.to_dict() for step in self.steps],
            "phase_transition": self.phase_transition.to_dict(),
            "action_window": self.action_window.to_dict(),
            "agent_messages": [message.to_dict() for message in self.agent_messages],
        }
