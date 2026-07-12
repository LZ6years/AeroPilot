"""Validate runtime control requests without dispatching UAV tools.

Phase 1 only accepts or rejects requests and records the outcome. Real behavior
for ask_user and semantic_plan_update is intentionally left to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.planning.schemas import RuntimeControlAction
from src.runtime.configs.runtime_control_config import RuntimeControlConfig
from src.runtime.control.actions.semantic_plan_update import (
    SemanticPlanUpdateJob,
    SemanticPlanUpdateManager,
    SemanticPlanUpdateResult,
)
from src.runtime.control.recommendations import runtime_control_recommendations
from src.runtime.control.registry import RuntimeControlRegistry


@dataclass(frozen=True)
class RuntimeControlResult:
    """Validation result for a runtime control request."""

    name: str
    status: str
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "payload": dict(self.payload),
        }


class RuntimeControlExecutor:
    """Runtime-side validator for control actions requested by the planner."""

    def __init__(
        self,
        config: RuntimeControlConfig | None = None,
        registry: RuntimeControlRegistry | None = None,
        semantic_plan_update_manager: SemanticPlanUpdateManager | None = None,
        semantic_plan_updater: Any | None = None,
    ) -> None:
        self.config = config or RuntimeControlConfig()
        self.registry = registry or RuntimeControlRegistry.default()
        self.semantic_plan_update_manager = semantic_plan_update_manager or SemanticPlanUpdateManager(
            semantic_plan_updater
        )
        self._status: dict[str, str] = {}

    def model_context(self, recommendation_context: dict[str, Any] | None = None) -> dict[str, Any]:
        available = self.config.exposed_actions(self.registry.names())
        self._status.update(self.semantic_plan_update_manager.model_status())
        if not available and not self._status:
            return {}
        context: dict[str, Any] = {"available": available}
        contracts = {
            name: contract.model_contract
            for name in available
            for contract in [self.registry.get(name)]
            if contract is not None and contract.model_contract
        }
        if contracts:
            context["contracts"] = contracts
        visible_status = {name: status for name, status in sorted(self._status.items()) if name in available}
        if visible_status:
            context["status"] = visible_status
        recommendation_context = recommendation_context or {}
        recommendations = runtime_control_recommendations(
            available=available,
            status=visible_status,
            phase_execution_summary=recommendation_context.get("phase_execution_summary"),
            completed=bool(recommendation_context.get("completed", False)),
            stopped=bool(recommendation_context.get("stopped", False)),
        )
        if recommendations:
            context["recommendations"] = recommendations
        return context

    def validate(self, action: RuntimeControlAction) -> RuntimeControlResult:
        contract = self.registry.get(action.name)
        if contract is None:
            return RuntimeControlResult(action.name, "rejected", "unknown_runtime_control_action", action.payload)
        if not self.config.is_enabled(action.name):
            return RuntimeControlResult(action.name, "rejected", "runtime_control_action_disabled", action.payload)
        if not self.config.is_exposed(action.name):
            return RuntimeControlResult(action.name, "rejected", "runtime_control_action_not_exposed", action.payload)
        try:
            contract.validate_payload(action.payload)
        except ValueError as exc:
            return RuntimeControlResult(action.name, "rejected", f"invalid_payload: {exc}", action.payload)
        if action.name == "semantic_plan_update":
            return RuntimeControlResult(action.name, "accepted", "validated_for_submission", action.payload)
        return RuntimeControlResult(action.name, "not_implemented", "phase_1_validation_only", action.payload)

    def validate_all(self, actions: list[RuntimeControlAction]) -> list[RuntimeControlResult]:
        return [self.validate(action) for action in actions]

    def submit_semantic_plan_update(self, job: SemanticPlanUpdateJob) -> RuntimeControlResult:
        if not self.config.is_enabled("semantic_plan_update"):
            return RuntimeControlResult("semantic_plan_update", "rejected", "runtime_control_action_disabled", job.request_payload)
        if not self.config.is_exposed("semantic_plan_update"):
            return RuntimeControlResult("semantic_plan_update", "rejected", "runtime_control_action_not_exposed", job.request_payload)
        ok, reason = self.semantic_plan_update_manager.submit(job)
        if not ok:
            return RuntimeControlResult("semantic_plan_update", "rejected", reason, job.request_payload)
        self._status["semantic_plan_update"] = "pending"
        return RuntimeControlResult("semantic_plan_update", "pending", reason, job.request_payload)

    def take_completed_semantic_plan_update(self) -> SemanticPlanUpdateResult | None:
        result = self.semantic_plan_update_manager.take_completed()
        if result is not None:
            self._status.pop("semantic_plan_update", None)
        return result
