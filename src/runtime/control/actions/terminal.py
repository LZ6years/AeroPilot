"""Task terminal validation, independent of flight cleanup and model semantics.

This module validates provenance, not whether a claimed arrival is true. The
Runtime must close ToolResults and perform shutdown after applying a terminal.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from src.agent.transcript import AssistantTurn, ToolResult, TranscriptEvent
from src.tools.builtins.terminal import BLOCKING_CONDITIONS
from src.tools.registry import ToolRegistry


class TerminalState(Protocol):
    completed: bool
    stopped: bool
    stopped_reason: str | None
    stop_source: str | None


def environmental_evidence_refs(
    events: Iterable[TranscriptEvent], registry: ToolRegistry,
) -> frozenset[str]:
    """Collect observed evidence from an instruction's isolated transcript.

    A request is not evidence until its successful result exists. Knowledge,
    controls and lifecycle results cannot prove environmental completion.
    Reset at an instruction boundary even if a caller supplies multiple tasks.
    """
    refs: set[str] = set()
    calls: dict[str, str] = {}
    for event in events:
        if event.kind == "user_instruction":
            refs.clear()
            calls.clear()
        elif event.kind == "state_checkpoint":
            observation_id = getattr(event.payload, "observation_id", None)
            if isinstance(observation_id, str) and observation_id:
                refs.add(observation_id)
        elif event.kind == "assistant_turn":
            turn: AssistantTurn = event.payload
            calls.update((call.tool_call_id, call.tool_name) for call in turn.tool_calls)
        elif event.kind == "tool_result":
            result: ToolResult = event.payload
            if result.status != "ok" or calls.get(result.tool_call_id) != result.tool_name:
                continue
            try:
                spec = registry.get(result.tool_name)
            except KeyError:
                continue
            if spec.runtime.effect not in {"physical", "read_only"} or spec.kind == "knowledge":
                continue
            refs.add(result.tool_call_id)
            if result.result_ref:
                refs.add(result.result_ref)
    return frozenset(refs)


def apply_terminal(
    state: TerminalState,
    payload: dict[str, Any],
    *,
    completed: bool,
    available_evidence: frozenset[str],
) -> dict[str, Any]:
    """Apply one already schema-checked terminal request atomically.

    Direct callers also receive basic defensive checks. No environment client
    is accepted here: cleanup is exclusively TaskRuntime's responsibility.
    """
    if state.completed or state.stopped:
        raise ValueError("task is already terminal")
    reason = payload.get("reason")
    refs = payload.get("evidence_refs")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("terminal reason must be non-empty")
    if (not isinstance(refs, list) or not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)):
        raise ValueError("terminal requires non-empty evidence_refs")
    if len(set(refs)) != len(refs) or not set(refs).issubset(available_evidence):
        invalid = sorted(set(refs) - available_evidence)
        duplicates = sorted({ref for ref in refs if refs.count(ref) > 1})
        examples = sorted(available_evidence, key=lambda ref: (not ref.startswith("obs"), ref))[:5]
        raise ValueError(
            "terminal evidence must reference prior environmental evidence in this instruction; "
            f"invalid references: {invalid[:5]!r}; duplicates: {duplicates[:5]!r}; "
            f"accepted examples from this instruction: {examples!r}. "
            "Use observation IDs or successful environmental tool_call_id/result_ref; "
            "image frame IDs, image paths, plans and knowledge are not accepted by themselves."
        )
    condition = payload.get("blocking_condition")
    if not completed and condition not in BLOCKING_CONDITIONS:
        raise ValueError("unsupported blocking_condition")
    source = "runtime.report_completed" if completed else "runtime.report_blocked"
    state.completed = completed
    state.stopped = not completed
    state.stopped_reason = None if completed else f"agent_reported_blocked:{condition}"
    state.stop_source = source
    return {
        "completed": state.completed, "stopped": state.stopped,
        "stopped_reason": state.stopped_reason, "stop_source": source,
        "reason": reason, "evidence_refs": list(refs),
        **({"suggested_user_action": payload["suggested_user_action"]}
           if "suggested_user_action" in payload else {}),
    }
