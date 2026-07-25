"""Schemas for short-horizon action planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.messages import AgentMessage
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition

ACTION_WINDOW_SCOPE_CURRENT_PHASE = "current_phase_only"
VALID_DONE_WHEN_ASSESSMENT_STATUSES = {"supported", "unknown", "contradicted"}


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
class RuntimeControlAction:
    """Planner request for runtime orchestration, separate from UAV tool steps."""

    name: str
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeControlAction":
        name = str(data.get("name", "")).strip()
        payload = data.get("payload", {})
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("RuntimeControlAction payload must be a dict")
        return cls(
            name=name,
            reason=str(data.get("reason", "")).strip(),
            payload=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reason": self.reason,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class DoneWhenAssessment:
    """Model judgment for one zero-based ``done_when`` condition."""

    index: int
    status: str
    evidence: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoneWhenAssessment":
        allowed_fields = {"index", "status", "evidence"}
        extra_fields = sorted(set(data) - allowed_fields)
        if extra_fields:
            raise ValueError(
                "DoneWhenAssessment contains unsupported fields: "
                + ", ".join(extra_fields)
            )
        index = data.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("DoneWhenAssessment index must be a non-negative integer")
        status = str(data.get("status", "")).strip()
        if status not in VALID_DONE_WHEN_ASSESSMENT_STATUSES:
            raise ValueError(
                "DoneWhenAssessment status must be one of: "
                + ", ".join(sorted(VALID_DONE_WHEN_ASSESSMENT_STATUSES))
            )
        evidence = str(data.get("evidence", "")).strip()
        if not evidence:
            raise ValueError("DoneWhenAssessment evidence must not be empty")
        return cls(index=index, status=status, evidence=evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CompletionAssessment:
    """Structured semantic evidence judgments returned by the coupled model."""

    done_when: list[DoneWhenAssessment]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionAssessment":
        extra_fields = sorted(set(data) - {"done_when"})
        if extra_fields:
            raise ValueError(
                "CompletionAssessment contains unsupported fields: "
                + ", ".join(extra_fields)
            )
        raw_items = data.get("done_when")
        if not isinstance(raw_items, list):
            raise ValueError("CompletionAssessment done_when must be a list")
        if any(not isinstance(item, dict) for item in raw_items):
            raise ValueError("CompletionAssessment done_when items must be objects")
        return cls(done_when=[DoneWhenAssessment.from_dict(dict(item)) for item in raw_items])

    def validate_coverage(self, expected_count: int) -> None:
        """Require exactly one assessment for every semantic condition."""

        indices = [item.index for item in self.done_when]
        expected = list(range(expected_count))
        if sorted(indices) != expected or len(indices) != len(set(indices)):
            raise ValueError(
                "CompletionAssessment must cover each done_when index exactly once; "
                f"expected={expected}, actual={indices}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"done_when": [item.to_dict() for item in self.done_when]}


@dataclass(frozen=True)
class ShortHorizonPlan:
    phase: str
    steps: list[PlannedStep]
    action_window: ActionWindow = field(default_factory=ActionWindow.default)
    agent_messages: list[AgentMessage] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        require_action_window: bool = False,
        require_agent_messages: bool = False,
    ) -> "ShortHorizonPlan":
        if require_action_window and "action_window" not in data:
            raise ValueError("ShortHorizonPlan response requires action_window")
        if require_agent_messages and "agent_messages" not in data:
            raise ValueError("ShortHorizonPlan response requires agent_messages")
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
            action_window=action_window,
            agent_messages=[AgentMessage.from_dict(dict(item)) for item in messages_data],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "steps": [step.to_dict() for step in self.steps],
            "action_window": self.action_window.to_dict(),
            "agent_messages": [message.to_dict() for message in self.agent_messages],
        }


@dataclass(frozen=True)
class CoupledActionPhasePlan:
    """Experimental one-call contract that decides phase progress and action.

    This schema deliberately keeps the phase decision and runtime control
    requests outside ``steps``. A coupled planner may either request a phase
    decision with no UAV tool actions, or continue the current phase with a
    normal tool-only ``ShortHorizonPlan``. Runtime control requests live in
    ``control_actions`` so they never become executable UAV pseudo-tools.
    """

    phase: str
    phase_decision: PhaseTransition
    steps: list[PlannedStep] = field(default_factory=list)
    action_window: ActionWindow = field(default_factory=ActionWindow.default)
    agent_messages: list[AgentMessage] = field(default_factory=list)
    control_actions: list[RuntimeControlAction] = field(default_factory=list)
    completion_assessment: CompletionAssessment | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        require_agent_messages: bool = False,
    ) -> "CoupledActionPhasePlan":
        if "phase_decision" not in data:
            raise ValueError("CoupledActionPhasePlan response requires phase_decision")
        if require_agent_messages and "agent_messages" not in data:
            raise ValueError("CoupledActionPhasePlan response requires agent_messages")
        phase_decision = PhaseTransition.from_dict(dict(data.get("phase_decision") or {}))
        raw_steps = data.get("steps", [])
        if raw_steps is None:
            raw_steps = []
        if not isinstance(raw_steps, list):
            raise ValueError("CoupledActionPhasePlan steps must be a list")
        steps = [PlannedStep.from_dict(dict(item)) for item in raw_steps]
        messages_data = data.get("agent_messages", [])
        if messages_data is None:
            messages_data = []
        if not isinstance(messages_data, list):
            raise ValueError("CoupledActionPhasePlan agent_messages must be a list")
        control_actions_data = data.get("control_actions", [])
        if control_actions_data is None:
            control_actions_data = []
        if not isinstance(control_actions_data, list):
            raise ValueError("CoupledActionPhasePlan control_actions must be a list")
        completion_assessment_data = data.get("completion_assessment")
        completion_assessment = (
            CompletionAssessment.from_dict(dict(completion_assessment_data))
            if isinstance(completion_assessment_data, dict)
            else None
        )

        action_window_data = data.get("action_window")
        action_window = (
            ActionWindow.from_dict(dict(action_window_data))
            if action_window_data is not None
            else ActionWindow.default()
        )
        decision = phase_decision.decision
        if decision == CONTINUE_CURRENT_PHASE:
            if "action_window" not in data:
                raise ValueError("CoupledActionPhasePlan continue decision requires action_window")
            if not 1 <= len(steps) <= 3:
                raise ValueError("CoupledActionPhasePlan continue decision requires 1-3 tool steps")
        elif decision in {ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK}:
            if steps:
                raise ValueError("CoupledActionPhasePlan transition decisions require empty steps")
        else:  # PhaseTransition.from_dict should already protect this.
            raise ValueError(f"Unsupported coupled phase decision: {decision}")

        return cls(
            phase=str(data.get("phase", "")).strip(),
            phase_decision=phase_decision,
            steps=steps,
            action_window=action_window,
            agent_messages=[AgentMessage.from_dict(dict(item)) for item in messages_data],
            control_actions=[RuntimeControlAction.from_dict(dict(item)) for item in control_actions_data],
            completion_assessment=completion_assessment,
        )

    def to_short_horizon_plan(self) -> ShortHorizonPlan:
        return ShortHorizonPlan(
            phase=self.phase,
            steps=list(self.steps),
            action_window=self.action_window,
            agent_messages=list(self.agent_messages),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "phase": self.phase,
            "phase_decision": self.phase_decision.to_dict(),
            "agent_messages": [message.to_dict() for message in self.agent_messages],
            "control_actions": [action.to_dict() for action in self.control_actions],
            "action_window": self.action_window.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.completion_assessment is not None:
            data["completion_assessment"] = self.completion_assessment.to_dict()
        return data
