"""Plan short-horizon actions with a configured model provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.planning.model_repair_planner import (
    ModelRepairPlanner,
    ModelRepairPlannerError,
    RepairAttemptRecord,
    RepairContext,
)
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.planning.tool_contract_validator import (
    ToolContractValidationError,
    ToolContractValidator,
    ToolContractViolation,
)
from src.agent.semantic.task_parser import _loads_json_object
from src.runtime.configs.model_config import ModelConfig, load_model_config


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when model action planning is requested without usable config."""


class ModelActionPlanner:
    planner_mode = "model"
    ALLOWED_TOOL_STEPS = {"uav.hover", "uav.turn_relative", "uav.move_body"}

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        prompt_path: str | None = None,
        *,
        max_repair_attempts: int = 3,
        repair_planner: ModelRepairPlanner | None = None,
    ) -> None:
        self.model_config = model_config or load_model_config()
        self.prompt_path = prompt_path or self.model_config.action_planner_prompt_path
        self.system_prompt = self._load_system_prompt()
        self.tool_contract_validator = ToolContractValidator(self.ALLOWED_TOOL_STEPS)
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.repair_planner = repair_planner or ModelRepairPlanner(self.model_config)
        self.last_repair_attempts: list[dict[str, Any]] = []
        self.last_repair_result: dict[str, Any] = {"repair_triggered": False}

    def plan_next(self, state: MissionState, context: dict | None = None) -> ShortHorizonPlan:
        self._reset_repair_state()
        if context is None:
            raise ValueError("ModelActionPlanner requires M5 context")
        if not self.model_config.api_key:
            raise ModelActionPlannerConfigError("Model action planner requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise ModelActionPlannerConfigError(f"Unsupported model provider: {self.model_config.provider}")
        response = self._call_openai_compatible(context)
        plan = ShortHorizonPlan.from_dict(response, require_phase_transition=True)
        try:
            self.tool_contract_validator.validate(plan, state.current_phase_name, context)
            return plan
        except ToolContractValidationError as exc:
            return self._repair_contract_violation(plan, response, exc, state.current_phase_name, context)

    def _call_openai_compatible(self, context: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
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
                f"Model action planner request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model action planner request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _repair_contract_violation(
        self,
        initial_plan: ShortHorizonPlan,
        initial_plan_data: dict[str, Any],
        initial_error: ToolContractValidationError,
        expected_phase: str,
        context: dict[str, Any],
    ) -> ShortHorizonPlan:
        if self.max_repair_attempts <= 0:
            raise initial_error

        self.last_repair_result = {"repair_triggered": True, "status": "started"}
        previous_invalid_plan = dict(initial_plan_data)
        previous_invalid_signature = _canonical_plan_signature(previous_invalid_plan)
        violations = [initial_error.violation.to_dict()]
        final_plan_data = dict(initial_plan_data)

        for attempt in range(1, self.max_repair_attempts + 1):
            repair_context = RepairContext.from_context(
                attempt=attempt,
                max_attempts=self.max_repair_attempts,
                phase=expected_phase,
                invalid_plan=previous_invalid_plan,
                violation=_violation_from_dict(violations[-1]),
                previous_violations=violations,
                planner_context=context,
            )
            repaired_plan_data = self.repair_planner.repair(repair_context)
            final_plan_data = dict(repaired_plan_data)

            if _canonical_plan_signature(repaired_plan_data) == previous_invalid_signature:
                violation = ToolContractViolation(
                    code="repair_repeated_same_invalid_plan",
                    message="Model repair planner repeated the same invalid plan",
                    field="plan",
                    expected="a different contract-valid ShortHorizonPlan",
                    actual=repaired_plan_data,
                ).to_dict()
                violations.append(violation)
                self._record_repair_attempt(attempt, repair_context, repaired_plan_data, violation, "failed")
                raise self._repair_error(
                    "Model repair planner repeated the same invalid plan",
                    "repair_repeated_same_invalid_plan",
                    initial_plan_data,
                    final_plan_data,
                    violations,
                )

            try:
                repaired_plan = ShortHorizonPlan.from_dict(repaired_plan_data, require_phase_transition=True)
                self.tool_contract_validator.validate(repaired_plan, expected_phase, context)
            except ToolContractValidationError as exc:
                violation = exc.violation.to_dict()
                violations.append(violation)
                self._record_repair_attempt(attempt, repair_context, repaired_plan_data, violation, "failed")
                previous_invalid_plan = dict(repaired_plan_data)
                previous_invalid_signature = _canonical_plan_signature(previous_invalid_plan)
                continue
            except ValueError as exc:
                violation = ToolContractViolation(
                    code="repair_plan_parse_error",
                    message=str(exc),
                    field="plan",
                    expected="valid ShortHorizonPlan with phase_transition",
                    actual=repaired_plan_data,
                ).to_dict()
                violations.append(violation)
                self._record_repair_attempt(attempt, repair_context, repaired_plan_data, violation, "failed")
                previous_invalid_plan = dict(repaired_plan_data)
                previous_invalid_signature = _canonical_plan_signature(previous_invalid_plan)
                continue

            self._record_repair_attempt(attempt, repair_context, repaired_plan_data, None, "repaired")
            self.last_repair_result = {
                "repair_triggered": True,
                "status": "repaired",
                "successful_attempt": attempt,
                "attempts": list(self.last_repair_attempts),
            }
            return repaired_plan

        raise self._repair_error(
            "Model repair planner exhausted contract repair attempts",
            "max_repair_attempts_exhausted",
            initial_plan_data,
            final_plan_data,
            violations,
        )

    def _record_repair_attempt(
        self,
        attempt: int,
        repair_context: RepairContext,
        repaired_plan: dict[str, Any] | None,
        violation: dict[str, Any] | None,
        status: str,
    ) -> None:
        record = RepairAttemptRecord(
            attempt=attempt,
            repair_context=repair_context.to_dict(),
            repaired_plan=repaired_plan,
            violation=violation,
            status=status,
        ).to_dict()
        self.last_repair_attempts.append(record)

    def _repair_error(
        self,
        message: str,
        reason: str,
        initial_plan: dict[str, Any],
        final_plan: dict[str, Any],
        violations: list[dict[str, Any]],
    ) -> ModelRepairPlannerError:
        self.last_repair_result = {
            "repair_triggered": True,
            "status": "failed",
            "reason": reason,
            "attempts": list(self.last_repair_attempts),
            "violations": list(violations),
        }
        return ModelRepairPlannerError(
            message,
            initial_plan=initial_plan,
            final_plan=final_plan,
            repair_attempts=list(self.last_repair_attempts),
            violations=list(violations),
            max_repair_attempts=self.max_repair_attempts,
            reason=reason,
        )

    def _reset_repair_state(self) -> None:
        self.last_repair_attempts = []
        self.last_repair_result = {"repair_triggered": False}

    def _load_system_prompt(self) -> str:
        path = Path(self.prompt_path) if self.prompt_path else Path(__file__).resolve().parent / "prompts" / "action_planner_system.txt"
        return path.read_text(encoding="utf-8").strip()


def _canonical_plan_signature(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _violation_from_dict(data: dict[str, Any]) -> ToolContractViolation:
    return ToolContractViolation(
        code=str(data.get("code", "")),
        message=str(data.get("message", "")),
        step_index=data.get("step_index"),
        tool=data.get("tool"),
        field=data.get("field"),
        expected=data.get("expected"),
        actual=data.get("actual"),
    )
