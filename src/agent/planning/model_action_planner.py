"""Plan short-horizon actions with a configured model provider."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import json
import urllib.error
import urllib.request
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.agent.mission_state import MissionState
from src.agent.planning.model_inputs import build_model_user_content
from src.agent.planning.model_repair_planner import (
    ModelRepairPlanner,
    ModelRepairPlannerError,
    RepairAttemptRecord,
    RepairContext,
)
from src.agent.phase import (
    ADVANCE_TO_NEXT_PHASE,
    COMPLETE_TASK,
    CONTINUE_CURRENT_PHASE,
    PhaseTransition,
)
from src.agent.planning.schemas import (
    ActionWindow,
    CompletionAssessment,
    CoupledActionPhasePlan,
    PlannedStep,
    ShortHorizonPlan,
)
from src.agent.planning.tool_contract_validator import (
    ToolContractValidationError,
    ToolContractValidator,
    ToolContractViolation,
)
from src.agent.semantic.task_parser import _loads_json_object
from src.runtime.configs.model_config import ModelConfig, load_model_config


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when model action planning is requested without usable config."""


class _ModelActionRequestError(RuntimeError):
    """Internal structured failure for one provider request attempt."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        http_status: int | None = None,
        body: str = "",
    ) -> None:
        self.kind = kind
        self.http_status = http_status
        self.body = body
        super().__init__(message)


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
        if prompt_path:
            raise ModelActionPlannerConfigError(
                "ModelActionPlanner now uses the coupled action/phase prompt only; "
                "action-only planner prompt overrides are no longer supported."
            )
        self.coupled_action_phase_system_prompt = self._load_coupled_action_phase_system_prompt()
        self.tool_contract_validator = ToolContractValidator(self.ALLOWED_TOOL_STEPS)
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.repair_planner = repair_planner or ModelRepairPlanner(self.model_config)
        self.last_repair_attempts: list[dict[str, Any]] = []
        self.last_repair_result: dict[str, Any] = {"repair_triggered": False}
        self.last_completion_result: dict[str, Any] = {"status": "not_run"}
        self.last_model_request_recovery: dict[str, Any] = {"status": "not_run"}

    def plan_next_with_phase_decision(
        self,
        state: MissionState,
        context: dict | None = None,
    ) -> CoupledActionPhasePlan:
        """Run the experimental one-call planner/judge prompt.

        Continue decisions still pass through the same tool contract validator
        and repair loop as normal action planning. The repair is intentionally
        scoped to the tool-only ``ShortHorizonPlan`` part; it never changes the
        coupled phase decision itself.
        """

        self._reset_repair_state()
        if context is None:
            raise ValueError("ModelActionPlanner requires M5 context")
        if not self.model_config.api_key:
            raise ModelActionPlannerConfigError("Model action planner requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise ModelActionPlannerConfigError(f"Unsupported model provider: {self.model_config.provider}")
        response = self._call_coupled_action_phase_openai_compatible(context)
        response_phase = str(response.get("phase", "")).strip()
        if response_phase != state.current_phase_name:
            raise ValueError(
                f"CoupledActionPhasePlan phase mismatch: expected {state.current_phase_name}, got {response_phase}"
            )
        coupled_plan = self._build_coupled_plan_from_model_response(response, state, context)
        if coupled_plan.phase_decision.decision == CONTINUE_CURRENT_PHASE:
            short_plan = coupled_plan.to_short_horizon_plan()
            try:
                self.tool_contract_validator.validate(short_plan, state.current_phase_name, context)
            except ToolContractValidationError as exc:
                repaired_plan = self._repair_contract_violation(
                    short_plan,
                    short_plan.to_dict(),
                    exc,
                    state.current_phase_name,
                    context,
                )
                return CoupledActionPhasePlan(
                    phase=coupled_plan.phase,
                    phase_decision=coupled_plan.phase_decision,
                    steps=list(repaired_plan.steps),
                    action_window=repaired_plan.action_window,
                    agent_messages=list(repaired_plan.agent_messages),
                    control_actions=list(coupled_plan.control_actions),
                    completion_assessment=coupled_plan.completion_assessment,
                )
        return coupled_plan

    def _build_coupled_plan_from_model_response(
        self,
        response: dict[str, Any],
        state: MissionState,
        context: dict[str, Any],
    ) -> CoupledActionPhasePlan:
        """Derive the phase transition from model evidence assessments.

        The model owns semantic evidence classification. This method owns the
        deterministic mapping from those classifications and runtime metrics to
        a phase transition, so free-text explanations cannot override a
        supported completion contract.
        """

        current_phase = context.get("current_phase")
        if not isinstance(current_phase, dict):
            return self._completion_fallback_plan(
                state,
                expected_count=0,
                reason="current_phase_missing",
            )
        done_when = current_phase.get("done_when")
        if not isinstance(done_when, list) or not done_when:
            return self._completion_fallback_plan(
                state,
                expected_count=0,
                reason="current_phase_done_when_missing",
            )

        try:
            raw_assessment = response.get("completion_assessment")
            if not isinstance(raw_assessment, dict):
                raise ValueError("completion_assessment must be an object")
            assessment = CompletionAssessment.from_dict(dict(raw_assessment))
            assessment.validate_coverage(len(done_when))
        except (TypeError, ValueError) as exc:
            return self._completion_fallback_plan(
                state,
                expected_count=len(done_when),
                reason=f"invalid_completion_assessment:{str(exc)[:240]}",
            )

        metric_complete = self._deterministic_metrics_complete(current_phase, context)
        semantic_complete = all(item.status == "supported" for item in assessment.done_when)
        transition = self._derived_phase_transition(
            state,
            assessment,
            semantic_complete=semantic_complete,
            metric_complete=metric_complete,
        )
        normalized = dict(response)
        corrections: list[str] = []
        if "phase_decision" in normalized:
            corrections.append("model_phase_decision_ignored")
        normalized["phase_decision"] = transition.to_dict()
        normalized["completion_assessment"] = assessment.to_dict()
        normalized.setdefault("agent_messages", [])
        normalized.setdefault("control_actions", [])

        if transition.decision != CONTINUE_CURRENT_PHASE:
            if normalized.get("steps"):
                corrections.append("transition_steps_discarded")
                normalized["agent_messages"] = []
            if normalized.get("control_actions"):
                corrections.append("transition_control_actions_discarded")
            normalized["steps"] = []
            normalized["control_actions"] = []
        else:
            raw_steps = normalized.get("steps")
            if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 3:
                return self._completion_fallback_plan(
                    state,
                    expected_count=len(done_when),
                    reason="continue_steps_missing_or_out_of_range",
                    assessment=assessment,
                    metric_complete=metric_complete,
                )
            if "action_window" not in normalized:
                return self._completion_fallback_plan(
                    state,
                    expected_count=len(done_when),
                    reason="continue_action_window_missing",
                    assessment=assessment,
                    metric_complete=metric_complete,
                )

        try:
            plan = CoupledActionPhasePlan.from_dict(
                normalized,
                require_agent_messages=True,
            )
        except (TypeError, ValueError) as exc:
            if transition.decision == CONTINUE_CURRENT_PHASE:
                return self._completion_fallback_plan(
                    state,
                    expected_count=len(done_when),
                    reason=f"continue_plan_parse_error:{str(exc)[:240]}",
                    assessment=assessment,
                    metric_complete=metric_complete,
                )
            raise

        self.last_completion_result = {
            "status": "assessed",
            "assessment": assessment.to_dict(),
            "semantic_complete": semantic_complete,
            "metric_complete": metric_complete,
            "derived_decision": transition.decision,
            "fallback": bool(corrections),
            "fallback_reasons": corrections,
        }
        return plan

    @staticmethod
    def _deterministic_metrics_complete(
        current_phase: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        metrics = current_phase.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            return True
        evaluation = context.get("phase_completion_evaluation")
        return isinstance(evaluation, dict) and evaluation.get("all_within_tolerance") is True

    @staticmethod
    def _derived_phase_transition(
        state: MissionState,
        assessment: CompletionAssessment,
        *,
        semantic_complete: bool,
        metric_complete: bool,
    ) -> PhaseTransition:
        if semantic_complete and metric_complete:
            decision = COMPLETE_TASK if state.is_final_phase else ADVANCE_TO_NEXT_PHASE
            return PhaseTransition(
                decision,
                "all done_when conditions are supported and deterministic metrics are satisfied",
            )
        pending = ", ".join(
            f"{item.index}={item.status}"
            for item in assessment.done_when
            if item.status != "supported"
        )
        if not metric_complete:
            pending = f"{pending}, metrics=outside_tolerance".strip(", ")
        return PhaseTransition(
            CONTINUE_CURRENT_PHASE,
            f"completion evidence still requires progress: {pending or 'unknown'}",
        )

    def _completion_fallback_plan(
        self,
        state: MissionState,
        *,
        expected_count: int,
        reason: str,
        assessment: CompletionAssessment | None = None,
        metric_complete: bool | None = None,
    ) -> CoupledActionPhasePlan:
        """Return one safe observation-producing window for an invalid contract."""

        fallback_assessment = assessment or CompletionAssessment.from_dict(
            {
                "done_when": [
                    {
                        "index": index,
                        "status": "unknown",
                        "evidence": "model completion assessment was missing or invalid",
                    }
                    for index in range(expected_count)
                ]
            }
        )
        transition = PhaseTransition(
            CONTINUE_CURRENT_PHASE,
            "completion assessment contract was invalid; hover and collect a fresh observation",
        )
        plan = CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=transition,
            steps=[
                PlannedStep.tool_step(
                    "uav.hover",
                    {},
                    "hold safely after an invalid completion assessment",
                )
            ],
            action_window=ActionWindow(
                intent="hold safely and refresh evidence",
                requires_intermediate_observation=False,
                expected_effect="collect a fresh observation without changing task semantics",
            ),
            agent_messages=[],
            control_actions=[],
            completion_assessment=fallback_assessment,
        )
        self.last_completion_result = {
            "status": "fallback",
            "assessment": fallback_assessment.to_dict(),
            "semantic_complete": False,
            "metric_complete": metric_complete,
            "derived_decision": CONTINUE_CURRENT_PHASE,
            "fallback": True,
            "fallback_reasons": [reason],
        }
        return plan

    def _call_coupled_action_phase_openai_compatible(self, context: dict[str, Any]) -> dict[str, Any]:
        return self._call_openai_compatible_json(self.coupled_action_phase_system_prompt, context)

    def _call_openai_compatible_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        full_content = build_model_user_content(user_payload)
        attempts: list[dict[str, Any]] = []

        try:
            result = self._request_openai_compatible_json(
                system_prompt,
                full_content,
                attempt=1,
                image_mode="all",
                attempts=attempts,
            )
        except _ModelActionRequestError as first_error:
            if _is_transient_model_request_error(first_error):
                result = self._retry_same_model_request(
                    system_prompt,
                    full_content,
                    attempts,
                    first_error,
                )
            elif _is_provider_image_url_error(first_error) and _valid_model_image_data_urls(
                full_content
            ):
                result = self._retry_image_url_request(
                    system_prompt,
                    user_payload,
                    full_content,
                    attempts,
                    first_error,
                )
            else:
                self._raise_model_request_failure(first_error, attempts)

        recovered = len(attempts) > 1
        self.last_model_request_recovery = {
            "status": "recovered" if recovered else "not_needed",
            "attempt_count": len(attempts),
            "degraded_to_now_only": any(
                item.get("image_mode") == "current_only" for item in attempts
            ),
            "attempts": attempts,
        }
        return result

    def _retry_same_model_request(
        self,
        system_prompt: str,
        content: str | list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        first_error: _ModelActionRequestError,
    ) -> dict[str, Any]:
        try:
            return self._request_openai_compatible_json(
                system_prompt,
                content,
                attempt=2,
                image_mode="all",
                attempts=attempts,
            )
        except _ModelActionRequestError as second_error:
            self._raise_model_request_failure(second_error, attempts, cause=first_error)

    def _retry_image_url_request(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        full_content: str | list[dict[str, Any]],
        attempts: list[dict[str, Any]],
        first_error: _ModelActionRequestError,
    ) -> dict[str, Any]:
        second_error: _ModelActionRequestError | None = None
        try:
            return self._request_openai_compatible_json(
                system_prompt,
                full_content,
                attempt=2,
                image_mode="all",
                attempts=attempts,
            )
        except _ModelActionRequestError as exc:
            second_error = exc
            if not _is_provider_image_url_error(exc):
                self._raise_model_request_failure(exc, attempts, cause=first_error)

        current_only_content = build_model_user_content(
            user_payload,
            image_mode="current_only",
        )
        if _model_image_count(current_only_content) == 0:
            self._raise_model_request_failure(
                second_error or first_error,
                attempts,
                cause=first_error,
            )
        try:
            return self._request_openai_compatible_json(
                system_prompt,
                current_only_content,
                attempt=3,
                image_mode="current_only",
                attempts=attempts,
            )
        except _ModelActionRequestError as third_error:
            self._raise_model_request_failure(third_error, attempts, cause=first_error)

    def _request_openai_compatible_json(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        *,
        attempt: int,
        image_mode: str,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
            error = _ModelActionRequestError(
                f"Model action planner request failed: HTTP {exc.code} {exc.reason}; body={body}",
                kind="http_error",
                http_status=int(exc.code),
                body=body,
            )
            attempts.append(_model_request_attempt(attempt, image_mode, user_content, error))
            raise error from exc
        except urllib.error.URLError as exc:
            error = _ModelActionRequestError(
                f"Model action planner request failed: {exc}",
                kind="network_error",
            )
            attempts.append(_model_request_attempt(attempt, image_mode, user_content, error))
            raise error from exc
        except TimeoutError as exc:
            error = _ModelActionRequestError(
                f"Model action planner request failed: {exc}",
                kind="timeout",
            )
            attempts.append(_model_request_attempt(attempt, image_mode, user_content, error))
            raise error from exc
        attempts.append(_model_request_attempt(attempt, image_mode, user_content, None))
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _raise_model_request_failure(
        self,
        error: _ModelActionRequestError,
        attempts: list[dict[str, Any]],
        *,
        cause: BaseException | None = None,
    ) -> None:
        self.last_model_request_recovery = {
            "status": "failed",
            "attempt_count": len(attempts),
            "degraded_to_now_only": any(
                item.get("image_mode") == "current_only" for item in attempts
            ),
            "attempts": list(attempts),
        }
        summary = json.dumps(self.last_model_request_recovery, ensure_ascii=True)
        raise RuntimeError(f"{error}; recovery={summary}") from (cause or error)

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
                repaired_plan = ShortHorizonPlan.from_dict(
                    repaired_plan_data,
                    require_action_window=True,
                    require_agent_messages=True,
                )
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
                    expected="valid tool-only ShortHorizonPlan",
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
        self.last_completion_result = {"status": "not_run"}
        self.last_model_request_recovery = {"status": "not_run"}

    def _load_coupled_action_phase_system_prompt(self) -> str:
        from pathlib import Path

        prompt_dir = Path(__file__).resolve().parent / "prompts"
        overview = (
            "You are FlyAgent's experimental coupled action-and-phase planner.\n\n"
            "In one JSON response, assess every current done_when condition and plan "
            "UAV tool actions only when the structured evidence assessment still "
            "requires progress. Runtime derives the phase transition deterministically."
        )
        phase_prompt = (prompt_dir / "coupled_phase_decision_system.txt").read_text(encoding="utf-8").strip()
        action_prompt = (prompt_dir / "coupled_action_planner_system.txt").read_text(encoding="utf-8").strip()
        return "\n\n--- Phase Decision Contract ---\n\n".join([overview, phase_prompt]) + (
            "\n\n--- Action Planning Contract ---\n\n" + action_prompt
        )


def _canonical_plan_signature(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _model_request_attempt(
    attempt: int,
    image_mode: str,
    content: str | list[dict[str, Any]],
    error: _ModelActionRequestError | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempt": attempt,
        "status": "failed" if error else "ok",
        "image_mode": image_mode,
        "image_count": _model_image_count(content),
    }
    if error is not None:
        result["error_kind"] = error.kind
        if error.http_status is not None:
            result["http_status"] = error.http_status
        if _is_provider_image_url_error(error):
            result["provider_error"] = "image_url_invalid"
    return result


def _model_image_count(content: str | list[dict[str, Any]]) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if part.get("type") == "image_url")


def _is_transient_model_request_error(error: _ModelActionRequestError) -> bool:
    if error.kind in {"network_error", "timeout"}:
        return True
    status = error.http_status
    return status in {408, 429} or bool(status is not None and status >= 500)


def _is_provider_image_url_error(error: _ModelActionRequestError) -> bool:
    if error.http_status != 400:
        return False
    body = error.body.lower()
    return (
        "provided url does not appear to be valid" in body
        and "invalid_parameter_error" in body
    )


def _valid_model_image_data_urls(content: str | list[dict[str, Any]]) -> bool:
    """Validate provider-bound image data only after an image-URL rejection."""

    if not isinstance(content, list):
        return False
    urls = [
        part.get("image_url", {}).get("url")
        for part in content
        if part.get("type") == "image_url" and isinstance(part.get("image_url"), dict)
    ]
    if not urls:
        return False
    for value in urls:
        if not isinstance(value, str) or not value.startswith("data:image/"):
            return False
        try:
            header, encoded = value.split(",", 1)
            if ";base64" not in header.lower():
                return False
            decoded = base64.b64decode(encoded, validate=True)
            with Image.open(BytesIO(decoded)) as image:
                image.verify()
        except (ValueError, binascii.Error, UnidentifiedImageError, OSError, SyntaxError):
            return False
    return True


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
