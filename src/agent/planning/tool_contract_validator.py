"""Validate model-produced action plans against the planner tool contract.

This module is intentionally narrower than the model planner.  The planner
talks to the model and parses JSON; this validator checks whether the parsed
plan obeys the action contract carried in PlannerContext.  Keeping the contract
logic here makes it reusable by future repair-planning code without giving that
future layer direct access to model-call details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn, TYPE_CHECKING

from src.agent.transcript import AssistantTurn
from src.tools.diagnostics import Diagnostic
from src.tools.validation import ToolArgumentError, validate_tool_args, validate_runtime_bound_tool_args

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class NativeTurnValidationError(ValueError):
    """Pre-execution rejection with attribution, not a semantic judgment."""

    def __init__(self, message: str, *, code: str, call_id: str | None = None,
                 path: str | None = None) -> None:
        super().__init__(message)
        self.diagnostic = Diagnostic(code, "validation", reason=message,
            scope="call" if call_id else "turn", source_tool_call_id=call_id,
            parameter_path=path, execution="not_started").to_dict()


def validate_native_turn(
    turn: AssistantTurn,
    registry: ToolRegistry,
    *,
    allowed_names: set[str],
    latest_observation: dict[str, Any],
    prior_call_ids: set[str] | frozenset[str] = frozenset(),
    executed_read_counts: dict[tuple[str, str], int] | None = None,
) -> str:
    """Validate an entire turn before any mutation, returning its effect.

    No commit, plan, or Phase prerequisite exists. Exposure and scene support
    are supplied by Runtime, policies by ToolRegistry. The caller records read
    attempts only when executed; rejected turns never consume an execution quota.
    """
    if not turn.tool_calls:
        raise NativeTurnValidationError("submit at least one tool call", code="empty_turn")
    seen = set(prior_call_ids)
    specs = []
    for call in turn.tool_calls:
        if not call.tool_call_id or call.tool_call_id in seen:
            raise NativeTurnValidationError("tool call IDs must be non-empty and unique within the instruction",
                                            code="duplicate_call_id", call_id=call.tool_call_id)
        seen.add(call.tool_call_id)
        try:
            spec = registry.get(call.tool_name)
        except KeyError as exc:
            raise NativeTurnValidationError("unknown tool", code="unknown_tool", call_id=call.tool_call_id) from exc
        if (call.tool_name not in allowed_names or not spec.runtime.model_exposed
                or spec.runtime.effect == "lifecycle"):
            raise NativeTurnValidationError("tool is not exposed in this scene", code="tool_not_exposed",
                                            call_id=call.tool_call_id)
        try:
            validate_tool_args(call.args, spec.input_schema)
            validate_runtime_bound_tool_args(call.args, spec.input_schema, latest_observation)
        except ToolArgumentError as exc:
            raise NativeTurnValidationError(str(exc), code=exc.code,
                                            call_id=call.tool_call_id, path=exc.path) from exc
        specs.append(spec)
    effects = {spec.runtime.effect for spec in specs}
    if effects == {"runtime_control", "read_only"}:
        controls = [
            (call, spec)
            for call, spec in zip(turn.tool_calls, specs)
            if spec.runtime.effect == "runtime_control"
        ]
        reads = [
            (call, spec)
            for call, spec in zip(turn.tool_calls, specs)
            if spec.runtime.effect == "read_only"
        ]
        if (
            len(controls) != 1
            or controls[0][0].tool_name != "runtime.update_plan"
            or controls[0][1].runtime.batching != "sidecar"
            or any(spec.runtime.batching != "parallel_safe" for _, spec in reads)
        ):
            raise NativeTurnValidationError(
                "only one plan update may accompany parallel-safe read-only calls",
                code="control_not_exclusive",
            )
        _validate_read_calls(
            reads,
            latest_observation=latest_observation,
            executed_read_counts=executed_read_counts,
        )
        return "plan_read_only"
    if len(effects) != 1:
        raise NativeTurnValidationError("read-only, physical and terminal calls require separate turns",
                                        code="mixed_effects")
    effect = specs[0].runtime.effect
    if effect == "physical":
        if len(specs) != 1 or specs[0].runtime.batching != "single":
            raise NativeTurnValidationError("physical turns require exactly one single physical call; observe before the next action",
                                            code="physical_batch_invalid")
        if any(spec.runtime.result_timing != "after_batch_observation" for spec in specs):
            raise NativeTurnValidationError("unsupported physical result timing", code="unsupported_timing")
    elif effect == "runtime_control":
        expected_batching = (
            "sidecar" if turn.tool_calls[0].tool_name == "runtime.update_plan" else "single"
        )
        if len(specs) != 1 or specs[0].runtime.batching != expected_batching:
            raise NativeTurnValidationError("terminal controls are exclusive; plan updates use the sidecar contract",
                                            code="control_not_exclusive")
        if specs[0].runtime.result_timing != "immediate":
            raise NativeTurnValidationError("unsupported control result timing", code="unsupported_timing")
    elif effect == "read_only":
        _validate_read_calls(
            list(zip(turn.tool_calls, specs)),
            latest_observation=latest_observation,
            executed_read_counts=executed_read_counts,
        )
    else:
        raise NativeTurnValidationError("non-executable tool effect", code="unsupported_effect")
    return effect


def _validate_read_calls(
    calls_and_specs: list[tuple[Any, Any]],
    *,
    latest_observation: dict[str, Any],
    executed_read_counts: dict[tuple[str, str], int] | None,
) -> None:
    """Validate one read subset without consuming its observation quota."""

    if any(
        spec.runtime.result_timing != "immediate" for _, spec in calls_and_specs
    ):
        raise NativeTurnValidationError(
            "read-only results must be immediate", code="unsupported_timing"
        )
    if any(
        spec.runtime.batching not in {"single", "parallel_safe"}
        for _, spec in calls_and_specs
    ):
        raise NativeTurnValidationError(
            "unsupported read-only batching", code="read_batch_invalid"
        )
    if len(calls_and_specs) > 1 and any(
        spec.runtime.batching != "parallel_safe" for _, spec in calls_and_specs
    ):
        raise NativeTurnValidationError(
            "independent read-only batching requires parallel_safe",
            code="read_batch_invalid",
        )
    counts = dict(executed_read_counts or {})
    observation_id = str(latest_observation.get("observation_id") or "")
    for call, spec in calls_and_specs:
        key = (observation_id, call.tool_name)
        limit = spec.runtime.max_calls_per_observation
        count = counts.get(key, 0)
        if limit is not None and count >= limit:
            raise NativeTurnValidationError(
                "per-observation call limit reached",
                code="observation_call_limit",
                call_id=call.tool_call_id,
            )
        counts[key] = count + 1
