"""Repair model action plans that violate the tool contract.

The repair planner is deliberately scoped to contract failures only.  It does
not repair execution errors, safety interventions, or visual uncertainty.  Its
input is a compact, structured RepairContext built from the invalid plan,
ToolContractViolation history, and the relevant PlannerContext contract fields.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.planning.tool_contract_validator import ToolContractViolation
from src.agent.semantic.task_parser import _loads_json_object
from src.runtime.configs.model_config import ModelConfig, load_model_config


@dataclass(frozen=True)
class RepairContext:
    """JSON-serializable input for one contract-repair attempt."""

    attempt: int
    max_attempts: int
    phase: str
    invalid_plan: dict[str, Any]
    violation: dict[str, Any]
    previous_violations: list[dict[str, Any]] = field(default_factory=list)
    allowed_action_constraints: dict[str, Any] = field(default_factory=dict)
    observation_policy: dict[str, Any] = field(default_factory=dict)
    context_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(
        cls,
        *,
        attempt: int,
        max_attempts: int,
        phase: str,
        invalid_plan: dict[str, Any],
        violation: ToolContractViolation,
        previous_violations: list[dict[str, Any]],
        planner_context: dict[str, Any],
    ) -> "RepairContext":
        """Build a small repair payload without exposing raw memory history."""

        return cls(
            attempt=attempt,
            max_attempts=max_attempts,
            phase=phase,
            invalid_plan=dict(invalid_plan),
            violation=violation.to_dict(),
            previous_violations=list(previous_violations),
            allowed_action_constraints=dict(planner_context.get("allowed_action_constraints", {})),
            observation_policy=dict(planner_context.get("observation_policy", {})),
            context_summary=_context_summary(planner_context),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "phase": self.phase,
            "invalid_plan": dict(self.invalid_plan),
            "violation": dict(self.violation),
            "previous_violations": list(self.previous_violations),
            "allowed_action_constraints": dict(self.allowed_action_constraints),
            "observation_policy": dict(self.observation_policy),
            "context_summary": dict(self.context_summary),
        }


@dataclass(frozen=True)
class RepairAttemptRecord:
    """One repair attempt for runtime logging and demo artifacts."""

    attempt: int
    repair_context: dict[str, Any]
    repaired_plan: dict[str, Any] | None = None
    violation: dict[str, Any] | None = None
    status: str = "requested"

    def to_dict(self) -> dict[str, Any]:
        data = {
            "attempt": self.attempt,
            "status": self.status,
            "repair_context": dict(self.repair_context),
        }
        if self.repaired_plan is not None:
            data["repaired_plan"] = dict(self.repaired_plan)
        if self.violation is not None:
            data["violation"] = dict(self.violation)
        return data


class ModelRepairPlannerError(ValueError):
    """Raised after contract repair attempts are exhausted or stuck."""

    def __init__(
        self,
        message: str,
        *,
        initial_plan: dict[str, Any],
        final_plan: dict[str, Any] | None,
        repair_attempts: list[dict[str, Any]],
        violations: list[dict[str, Any]],
        max_repair_attempts: int,
        reason: str,
    ) -> None:
        self.initial_plan = dict(initial_plan)
        self.final_plan = dict(final_plan or {})
        self.repair_attempts = list(repair_attempts)
        self.violations = list(violations)
        self.max_repair_attempts = max_repair_attempts
        self.reason = reason
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "initial_plan": dict(self.initial_plan),
            "final_plan": dict(self.final_plan),
            "repair_attempts": list(self.repair_attempts),
            "violations": list(self.violations),
            "max_repair_attempts": self.max_repair_attempts,
        }


class ModelRepairPlanner:
    """Call the configured model to repair one invalid action plan."""

    def __init__(self, model_config: ModelConfig | None = None, prompt_path: str | None = None) -> None:
        self.model_config = model_config or load_model_config()
        self.prompt_path = prompt_path
        self.system_prompt = self._load_system_prompt()

    def repair(self, repair_context: RepairContext) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(repair_context.to_dict(), ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model repair planner request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model repair planner request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _load_system_prompt(self) -> str:
        path = (
            Path(self.prompt_path)
            if self.prompt_path
            else Path(__file__).resolve().parent / "prompts" / "model_repair_system.txt"
        )
        return path.read_text(encoding="utf-8").strip()


def _context_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Keep repair prompts focused on contract repair, not raw mission history."""

    return {
        "active_instruction": context.get("active_instruction"),
        "current_phase": context.get("current_phase"),
        "semantic_goal": context.get("semantic_goal"),
        "safety_constraints": context.get("safety_constraints"),
        "fragment_names": [
            fragment.get("name")
            for fragment in context.get("fragments", [])
            if isinstance(fragment, dict)
        ],
    }
