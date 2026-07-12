"""Registry for runtime control action contracts.

The registry is intentionally small and static in Phase 1. It names the control
actions that the planner may request when runtime configuration exposes them;
the registry does not execute those requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeControlContract:
    """Stable contract for a runtime control action."""

    name: str
    description: str
    model_contract: str
    required_payload_fields: tuple[str, ...] = ()
    optional_payload_fields: tuple[str, ...] = ()

    def validate_payload(self, payload: dict[str, Any]) -> None:
        missing = [field for field in self.required_payload_fields if field not in payload]
        if missing:
            raise ValueError(f"missing required payload field(s): {', '.join(missing)}")


@dataclass(frozen=True)
class RuntimeControlRegistry:
    """Lookup table for registered runtime control actions."""

    contracts: dict[str, RuntimeControlContract] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "RuntimeControlRegistry":
        return cls(
            contracts={
                "ask_user": RuntimeControlContract(
                    name="ask_user",
                    description="request user clarification when safe progress requires user input",
                    model_contract=(
                        "Request when user clarification is needed to avoid a wrong target, unsafe approach, "
                        "or changing the task meaning."
                    ),
                    required_payload_fields=("question",),
                    optional_payload_fields=("context", "choices"),
                ),
                "semantic_plan_update": RuntimeControlContract(
                    name="semantic_plan_update",
                    description="request a non-blocking semantic contract update for the current phase",
                    model_contract=(
                        "After multiple action-observation cycles, request when confidence is low that the current "
                        "goal, done_when, continue_when, or metrics still fit observed progress and user intent. "
                        "Name the doubtful fields and evidence; keep normal UAV steps if continuing."
                    ),
                    required_payload_fields=("reason",),
                    optional_payload_fields=("focus",),
                ),
            }
        )

    def names(self) -> list[str]:
        return sorted(self.contracts)

    def get(self, name: str) -> RuntimeControlContract | None:
        return self.contracts.get(name)
