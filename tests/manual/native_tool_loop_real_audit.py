"""Audit finalized native ToolCall UAV-Flow runs without executing them.

The utility reads ``summary.json``, per-case ``case.json``/``audit.json``, and
``scene_session.json`` from an existing evaluation directory.  It validates
the chronological native tool protocol, observation boundaries, physical
action replay, and portable HTML assets.  It never contacts a model or
simulator and never mutates Runtime run directories.

Example::

    python tests/manual/native_tool_loop_real_audit.py ^
      --eval-dir H:/UAV-Agent-Evaluation/uav_flow/outputs/native_canary ^
      --output H:/UAV-Agent-Evaluation/uav_flow/outputs/native_canary/native_tool_audit.json ^
      --strict

``--strict`` returns exit code 2 when an audit violation is found.  Counts in
the report describe exercised coverage only; protocol validity is determined
from the concrete per-turn violations rather than from aggregate counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tools.registry import ToolRegistry, default_registry


SCHEMA_VERSION = "native_tool_loop_real_audit_v2"
TERMINAL_RESULT_STATUSES = {"ok", "error", "rejected"}
PUBLIC_INTENT_SOURCES = {"provider_content", "tool_arguments_fallback"}


@dataclass(frozen=True)
class Violation:
    """One precise audit failure tied to a case and transcript record."""

    code: str
    message: str
    case_id: str | None = None
    transcript_id: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "case_id": self.case_id,
            "transcript_id": self.transcript_id,
            "tool_call_id": self.tool_call_id,
        }


def audit_evaluation(eval_dir: str | Path) -> dict[str, Any]:
    """Return a deterministic read-only audit report for one evaluation."""

    root = Path(eval_dir).resolve()
    violations: list[Violation] = []
    summary = _read_object(root / "summary.json", violations, code="summary_invalid")
    session = _read_object(
        root / "scene_session.json", violations, code="scene_session_invalid"
    )
    expected = _integer(summary.get("expected_case_count"))
    finalized = _integer(summary.get("finalized_case_count"))
    if expected is None or finalized is None or expected != finalized:
        violations.append(
            Violation(
                "evaluation_not_fully_finalized",
                f"expected_case_count={expected!r}, finalized_case_count={finalized!r}",
            )
        )
    if summary.get("run_status") != "completed":
        violations.append(
            Violation(
                "evaluation_not_completed",
                f"run_status={summary.get('run_status')!r}",
            )
        )
    _audit_scene_session(session, violations)

    cases_root = root / "cases"
    case_dirs = (
        sorted(path for path in cases_root.iterdir() if path.is_dir())
        if cases_root.is_dir()
        else []
    )
    if not case_dirs:
        violations.append(Violation("cases_missing", f"no case directories under {cases_root}"))

    registry = default_registry()
    case_reports: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        case_reports.append(_audit_case(case_dir, registry, violations))

    coverage = {
        "assistant_turns": sum(item["coverage"]["assistant_turns"] for item in case_reports),
        "read_only_turns": sum(item["coverage"]["read_only_turns"] for item in case_reports),
        "committed_turns": sum(item["coverage"]["committed_turns"] for item in case_reports),
        "physical_batches": sum(item["coverage"]["physical_batches"] for item in case_reports),
        "multi_action_batches": sum(
            item["coverage"]["multi_action_batches"] for item in case_reports
        ),
        "protocol_feedback_items": sum(
            item["coverage"]["protocol_feedback_items"] for item in case_reports
        ),
        "provider_rejected_turns": sum(
            item["coverage"]["provider_rejected_turns"] for item in case_reports
        ),
        "rejected_assistant_turns": sum(
            item["coverage"]["rejected_assistant_turns"] for item in case_reports
        ),
        "reported_blocked_turns": sum(
            item["coverage"]["reported_blocked_turns"] for item in case_reports
        ),
        "task_terminal_items": sum(
            item["coverage"]["task_terminal_items"] for item in case_reports
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_dir": str(root),
        "status": "ok" if not violations else "violations_found",
        "run_status": summary.get("run_status"),
        "expected_case_count": expected,
        "finalized_case_count": finalized,
        "scene_session": session,
        "coverage": coverage,
        "cases": case_reports,
        "violations": [item.to_dict() for item in violations],
    }


def _audit_scene_session(session: dict[str, Any], violations: list[Violation]) -> None:
    expected = {"start_count": 1, "restart_count": 0, "stop_count": 1}
    for key, value in expected.items():
        if session.get(key) != value:
            violations.append(
                Violation(
                    "scene_session_count_mismatch",
                    f"{key} must be {value}, received {session.get(key)!r}",
                )
            )


def _audit_case(
    case_dir: Path,
    registry: ToolRegistry,
    violations: list[Violation],
) -> dict[str, Any]:
    case_id = case_dir.name
    before = len(violations)
    case_data = _read_object(
        case_dir / "case.json", violations, code="case_json_invalid", case_id=case_id
    )
    audit = _read_object(
        case_dir / "audit.json", violations, code="audit_json_invalid", case_id=case_id
    )
    if case_data.get("status") == "error":
        error = case_data.get("error")
        violations.append(
            Violation(
                "case_execution_error",
                f"case finalized with an execution error: {error!r}",
                case_id=case_id,
            )
        )
        error_text = json.dumps(error, ensure_ascii=False).lower()
        if "native tool loop exhausted" in error_text:
            violations.append(
                Violation(
                    "native_tool_loop_exhausted",
                    "provider did not produce a valid committed turn within max_tool_rounds",
                    case_id=case_id,
                )
            )
    records = audit.get("agent_transcript")
    if not isinstance(records, list) or not records:
        violations.append(
            Violation("transcript_missing", "agent_transcript is empty", case_id=case_id)
        )
        records = []

    transcript = [item for item in records if isinstance(item, dict)]
    if len(transcript) != len(records):
        violations.append(
            Violation(
                "transcript_record_invalid",
                "every agent_transcript item must be an object",
                case_id=case_id,
            )
        )

    coverage = _audit_transcript(
        transcript,
        audit,
        case_data,
        case_dir,
        registry,
        violations,
        case_id,
    )
    return {
        "case_id": case_id,
        "class_name": _nested(case_data, "case", "class_name"),
        "case_status": case_data.get("status"),
        "completed": bool(case_data.get("completed")),
        "stopped_reason": case_data.get("stopped_reason"),
        "status": "ok" if len(violations) == before else "violations_found",
        "coverage": coverage,
        "violation_count": len(violations) - before,
    }


def _audit_transcript(
    records: list[dict[str, Any]],
    audit: dict[str, Any],
    case_data: dict[str, Any],
    case_dir: Path,
    registry: ToolRegistry,
    violations: list[Violation],
    case_id: str,
) -> dict[str, int]:
    coverage = {
        "assistant_turns": 0,
        "read_only_turns": 0,
        "committed_turns": 0,
        "physical_batches": 0,
        "multi_action_batches": 0,
        "protocol_feedback_items": 0,
        "provider_rejected_turns": 0,
        "rejected_assistant_turns": 0,
        "reported_blocked_turns": 0,
        "task_terminal_items": 0,
    }
    if not records:
        return coverage
    if records[0].get("kind") != "user_instruction":
        violations.append(
            Violation(
                "instruction_not_first",
                "the first transcript item must be user_instruction",
                case_id=case_id,
                transcript_id=_record_id(records[0]),
            )
        )
    if not any(item.get("kind") == "state_checkpoint" for item in records[1:]):
        violations.append(
            Violation(
                "state_checkpoint_missing",
                "no state checkpoint follows the user instruction",
                case_id=case_id,
            )
        )
    if not any(item.get("kind") == "assistant_turn" for item in records):
        violations.append(
            Violation(
                "assistant_turn_missing",
                "transcript contains no accepted assistant tool-call turn",
                case_id=case_id,
            )
        )

    _audit_unique_record_ids(records, violations, case_id)
    observation_ids, rgb_observation_ids = _observation_sets(audit)
    checkpoint_indices: list[tuple[int, str]] = []
    calls: dict[str, tuple[int, dict[str, Any], str]] = {}
    results: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    assistant_indices = [
        index for index, item in enumerate(records) if item.get("kind") == "assistant_turn"
    ]

    for index, item in enumerate(records):
        kind = item.get("kind")
        payload = _payload(item)
        transcript_id = _record_id(item)
        if kind == "state_checkpoint":
            observation_id = payload.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                violations.append(
                    Violation(
                        "checkpoint_observation_missing",
                        "state checkpoint requires observation_id",
                        case_id,
                        transcript_id,
                    )
                )
            else:
                checkpoint_indices.append((index, observation_id))
                if observation_id not in observation_ids:
                    violations.append(
                        Violation(
                            "checkpoint_observation_unknown",
                            f"observation_id {observation_id!r} is absent from audit observations",
                            case_id,
                            transcript_id,
                        )
                    )
                if observation_id not in rgb_observation_ids:
                    violations.append(
                        Violation(
                            "checkpoint_rgb_missing",
                            f"observation_id {observation_id!r} has no portable RGB asset",
                            case_id,
                            transcript_id,
                        )
                    )
            context = payload.get("context")
            if not isinstance(context, dict):
                violations.append(
                    Violation(
                        "checkpoint_context_invalid",
                        "state checkpoint context must be an object",
                        case_id,
                        transcript_id,
                    )
                )
            elif _contains_forbidden_history_key(context):
                violations.append(
                    Violation(
                        "checkpoint_duplicates_transcript",
                        "state context embeds agent transcript/tool-call history",
                        case_id,
                        transcript_id,
                    )
                )
            if isinstance(context, dict):
                _audit_checkpoint_image_contract(
                    context,
                    observation_id if isinstance(observation_id, str) else "",
                    violations,
                    case_id,
                    transcript_id,
                )
        elif kind == "assistant_turn":
            coverage["assistant_turns"] += 1
            intent = payload.get("public_intent")
            if not isinstance(intent, str) or not intent.strip():
                violations.append(
                    Violation(
                        "public_intent_missing",
                        "assistant turn requires non-empty public_intent",
                        case_id,
                        transcript_id,
                    )
                )
            intent_source = payload.get(
                "public_intent_source", "provider_content"
            )
            if intent_source not in PUBLIC_INTENT_SOURCES:
                violations.append(
                    Violation(
                        "public_intent_source_invalid",
                        f"unsupported public_intent_source {intent_source!r}",
                        case_id,
                        transcript_id,
                    )
                )
            raw_calls = payload.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                violations.append(
                    Violation(
                        "assistant_tool_calls_missing",
                        "assistant turn requires at least one ToolCall",
                        case_id,
                        transcript_id,
                    )
                )
                continue
            for call in raw_calls:
                if not isinstance(call, dict):
                    violations.append(
                        Violation(
                            "tool_call_invalid",
                            "ToolCall must be an object",
                            case_id,
                            transcript_id,
                        )
                    )
                    continue
                call_id = call.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    violations.append(
                        Violation(
                            "tool_call_id_missing",
                            "ToolCall requires tool_call_id",
                            case_id,
                            transcript_id,
                        )
                    )
                    continue
                if call_id in calls:
                    violations.append(
                        Violation(
                            "tool_call_id_duplicate",
                            f"duplicate tool_call_id {call_id!r}",
                            case_id,
                            transcript_id,
                            call_id,
                        )
                    )
                else:
                    calls[call_id] = (index, call, transcript_id)
        elif kind == "tool_result":
            call_id = payload.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                violations.append(
                    Violation(
                        "tool_result_call_id_missing",
                        "ToolResult requires tool_call_id",
                        case_id,
                        transcript_id,
                    )
                )
                continue
            results.setdefault(call_id, []).append((index, payload, transcript_id))
            if payload.get("status") not in TERMINAL_RESULT_STATUSES:
                violations.append(
                    Violation(
                        "tool_result_status_invalid",
                        f"unsupported terminal status {payload.get('status')!r}",
                        case_id,
                        transcript_id,
                        call_id,
                    )
                )
        elif kind == "protocol_feedback":
            coverage["protocol_feedback_items"] += 1
        elif kind == "provider_turn_rejected":
            coverage["provider_rejected_turns"] += 1
            if not isinstance(payload.get("error"), str) or not payload["error"].strip():
                violations.append(
                    Violation(
                        "provider_rejection_error_missing",
                        "provider_turn_rejected requires a non-empty error",
                        case_id,
                        transcript_id,
                    )
                )
            raw_calls = payload.get("tool_calls")
            if not isinstance(raw_calls, list):
                violations.append(
                    Violation(
                        "provider_rejection_calls_invalid",
                        "provider_turn_rejected tool_calls must be a list",
                        case_id,
                        transcript_id,
                    )
                )
        elif kind == "task_terminal":
            coverage["task_terminal_items"] += 1
            if payload.get("completed") is not False or payload.get("stopped") is not True:
                violations.append(
                    Violation(
                        "task_terminal_state_invalid",
                        "task_terminal must record completed=false and stopped=true",
                        case_id,
                        transcript_id,
                    )
                )

    _audit_provider_rejection_feedback(records, violations, case_id)

    _audit_call_result_pairs(records, calls, results, violations, case_id)

    accepted_physical_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    physical_batch_turns: list[tuple[int, int, str]] = []
    for assistant_index in assistant_indices:
        item = records[assistant_index]
        transcript_id = _record_id(item)
        raw_calls = _payload(item).get("tool_calls")
        turn_calls = [call for call in raw_calls if isinstance(call, dict)] if isinstance(raw_calls, list) else []
        if _assistant_turn_fully_rejected(turn_calls, results):
            # Runtime rejected the entire canonical turn before execution and
            # already recorded terminal ToolResults plus repair feedback. Such
            # attempts are valuable audit evidence, but they are not committed
            # windows and must not create synthetic observation/action duties.
            coverage["rejected_assistant_turns"] += 1
            continue
        specs = []
        for call in turn_calls:
            name = call.get("tool_name")
            try:
                spec = registry.get(str(name))
            except KeyError:
                violations.append(
                    Violation(
                        "tool_unknown",
                        f"tool {name!r} is not present in the current ToolRegistry",
                        case_id,
                        transcript_id,
                        _string(call.get("tool_call_id")),
                    )
                )
                continue
            if not spec.runtime.model_exposed:
                violations.append(
                    Violation(
                        "tool_not_model_exposed",
                        f"tool {name!r} is not model_exposed",
                        case_id,
                        transcript_id,
                        _string(call.get("tool_call_id")),
                    )
                )
            specs.append((call, spec))

        read_calls = [(call, spec) for call, spec in specs if spec.runtime.effect == "read_only"]
        physical_calls = [(call, spec) for call, spec in specs if spec.runtime.effect == "physical"]
        report_blocked_calls = [
            (call, spec)
            for call, spec in specs
            if call.get("tool_name") == "runtime.report_blocked"
        ]
        commits = [
            (call, spec)
            for call, spec in specs
            if call.get("tool_name") == "runtime.commit_phase_turn"
        ]
        if read_calls:
            coverage["read_only_turns"] += 1
            if len(read_calls) != len(specs):
                violations.append(
                    Violation(
                        "read_only_turn_mixed",
                        "read-only tools cannot be mixed with commit or side-effect tools",
                        case_id,
                        transcript_id,
                    )
                )
            if any(spec.runtime.result_timing != "immediate" for _, spec in read_calls):
                violations.append(
                    Violation(
                        "read_only_result_not_immediate",
                        "read-only tools must declare immediate result timing",
                        case_id,
                        transcript_id,
                    )
                )
            _audit_read_only_observation_ids(
                read_calls,
                assistant_index,
                checkpoint_indices,
                violations,
                case_id,
                transcript_id,
            )
            continue

        coverage["committed_turns"] += 1
        if len(commits) != 1 or not turn_calls or turn_calls[0].get("tool_name") != "runtime.commit_phase_turn":
            violations.append(
                Violation(
                    "commit_contract_invalid",
                    "non-read-only turn requires exactly one leading runtime.commit_phase_turn",
                    case_id,
                    transcript_id,
                )
            )
            continue
        assessment = _nested(commits[0][0], "args", "completion_assessment", "done_when")
        statuses = [item.get("status") for item in assessment if isinstance(item, dict)] if isinstance(assessment, list) else []
        if not statuses:
            violations.append(
                Violation(
                    "completion_assessment_missing",
                    "commit must carry a non-empty completion assessment",
                    case_id,
                    transcript_id,
                    _string(commits[0][0].get("tool_call_id")),
                )
            )
        complete = bool(statuses) and all(status == "supported" for status in statuses)
        side_effect_calls = [
            (call, spec)
            for call, spec in specs
            if spec.runtime.effect in {"physical", "runtime_control"}
            and call.get("tool_name") != "runtime.commit_phase_turn"
        ]
        if complete and side_effect_calls:
            violations.append(
                Violation(
                    "completed_turn_has_side_effects",
                    "a completed phase commit cannot include physical/runtime-control side effects",
                    case_id,
                    transcript_id,
                )
            )
        if report_blocked_calls:
            coverage["reported_blocked_turns"] += 1
            other_side_effects = [
                (call, spec)
                for call, spec in side_effect_calls
                if call.get("tool_name") != "runtime.report_blocked"
            ]
            if (
                complete
                or len(report_blocked_calls) != 1
                or len(turn_calls) != 2
                or turn_calls[1].get("tool_name") != "runtime.report_blocked"
                or physical_calls
                or other_side_effects
            ):
                violations.append(
                    Violation(
                        "report_blocked_turn_invalid",
                        "runtime.report_blocked requires one incomplete leading commit and no other calls",
                        case_id,
                        transcript_id,
                    )
                )
            if not any(status in {"unknown", "contradicted"} for status in statuses):
                violations.append(
                    Violation(
                        "report_blocked_assessment_invalid",
                        "blocked termination requires unknown or contradicted completion evidence",
                        case_id,
                        transcript_id,
                    )
                )
            report_args = _object(report_blocked_calls[0][0].get("args"))
            evidence_refs = report_args.get("evidence_refs")
            known_refs = _prior_evidence_refs(records, assistant_index)
            if not isinstance(evidence_refs, list) or not evidence_refs:
                violations.append(
                    Violation(
                        "report_blocked_evidence_missing",
                        "runtime.report_blocked requires at least one evidence reference",
                        case_id,
                        transcript_id,
                    )
                )
            else:
                unknown_refs = sorted(
                    str(ref)
                    for ref in evidence_refs
                    if not isinstance(ref, str) or not ref or ref not in known_refs
                )
                if unknown_refs:
                    violations.append(
                        Violation(
                            "report_blocked_evidence_unknown",
                            "runtime.report_blocked cites unknown prior evidence: "
                            + ", ".join(unknown_refs),
                            case_id,
                            transcript_id,
                        )
                    )
            later_terminal = any(
                record.get("kind") == "task_terminal"
                for record in records[assistant_index + 1 :]
            )
            if not later_terminal:
                violations.append(
                    Violation(
                        "report_blocked_terminal_missing",
                        "accepted runtime.report_blocked must be followed by task_terminal",
                        case_id,
                        transcript_id,
                    )
                )
        elif not complete and not 1 <= len(physical_calls) <= 3:
            violations.append(
                Violation(
                    "physical_batch_size_invalid",
                    f"incomplete phase requires 1-3 physical calls, received {len(physical_calls)}",
                    case_id,
                    transcript_id,
                )
            )
        if any(spec.runtime.batching != "ordered" for _, spec in physical_calls):
            violations.append(
                Violation(
                    "physical_batching_invalid",
                    "every physical call in a committed window must use ordered batching",
                    case_id,
                    transcript_id,
                )
            )

        if physical_calls:
            coverage["physical_batches"] += 1
            coverage["multi_action_batches"] += int(len(physical_calls) > 1)
            physical_batch_turns.append((assistant_index, len(physical_calls), transcript_id))
            for call, _ in physical_calls:
                call_id = _string(call.get("tool_call_id"))
                paired = results.get(call_id, [])
                if len(paired) != 1:
                    continue
                result_payload = paired[0][1]
                if result_payload.get("status") == "ok":
                    requested = _object(result_payload.get("requested_args"))
                    executed = _object(result_payload.get("executed_args"))
                    interventions = result_payload.get("safety_interventions")
                    if not _json_equal(requested, executed) and not interventions:
                        violations.append(
                            Violation(
                                "silent_safety_argument_change",
                                "requested_args differ from executed_args without a safety intervention",
                                case_id,
                                paired[0][2],
                                call_id,
                            )
                        )
                    if not result_payload.get("result_ref"):
                        violations.append(
                            Violation(
                                "physical_result_ref_missing",
                                "successful physical ToolResult requires result_ref",
                                case_id,
                                paired[0][2],
                                call_id,
                            )
                        )
                    accepted_physical_calls.append(
                        (str(call.get("tool_name")), executed, result_payload)
                    )

    _audit_physical_observations(
        records,
        physical_batch_turns,
        checkpoint_indices,
        audit,
        violations,
        case_id,
    )
    _audit_action_replay(
        accepted_physical_calls, audit, registry, violations, case_id
    )
    _audit_portable_html(case_dir, audit, violations, case_id)
    _audit_scene_capabilities(records, case_data, registry, violations, case_id)
    return coverage


def _prior_evidence_refs(
    records: list[dict[str, Any]], assistant_index: int
) -> set[str]:
    """Collect instruction-local observation and ToolCall/result references."""

    refs: set[str] = set()
    for item in records[:assistant_index]:
        payload = _payload(item)
        if item.get("kind") == "state_checkpoint":
            observation_id = payload.get("observation_id")
            if isinstance(observation_id, str) and observation_id:
                refs.add(observation_id)
        elif item.get("kind") == "assistant_turn":
            calls = payload.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    call_id = _object(call).get("tool_call_id")
                    if isinstance(call_id, str) and call_id:
                        refs.add(call_id)
        elif item.get("kind") == "tool_result":
            call_id = payload.get("tool_call_id")
            result_ref = payload.get("result_ref")
            if isinstance(call_id, str) and call_id:
                refs.add(call_id)
            if isinstance(result_ref, str) and result_ref:
                refs.add(result_ref)
    return refs


def _audit_provider_rejection_feedback(
    records: list[dict[str, Any]],
    violations: list[Violation],
    case_id: str,
) -> None:
    """Require rejected wire responses to be followed by repair guidance."""

    for index, item in enumerate(records):
        if item.get("kind") != "provider_turn_rejected":
            continue
        next_assistant = next(
            (
                later
                for later in range(index + 1, len(records))
                if records[later].get("kind") == "assistant_turn"
            ),
            len(records),
        )
        if not any(
            records[later].get("kind") == "protocol_feedback"
            for later in range(index + 1, next_assistant)
        ):
            violations.append(
                Violation(
                    "provider_rejection_feedback_missing",
                    "provider_turn_rejected must be followed by protocol feedback",
                    case_id,
                    _record_id(item),
                )
            )


def _assistant_turn_fully_rejected(
    calls: list[dict[str, Any]],
    results: dict[str, list[tuple[int, dict[str, Any], str]]],
) -> bool:
    if not calls:
        return False
    paired = []
    for call in calls:
        call_results = results.get(_string(call.get("tool_call_id")), [])
        if len(call_results) != 1:
            return False
        paired.append(call_results[0][1])
    return all(result.get("status") == "rejected" for result in paired)


def _audit_call_result_pairs(
    records: list[dict[str, Any]],
    calls: dict[str, tuple[int, dict[str, Any], str]],
    results: dict[str, list[tuple[int, dict[str, Any], str]]],
    violations: list[Violation],
    case_id: str,
) -> None:
    for call_id, (call_index, call, transcript_id) in calls.items():
        paired = results.get(call_id, [])
        if len(paired) != 1:
            violations.append(
                Violation(
                    "tool_result_cardinality",
                    f"ToolCall requires exactly one ToolResult, received {len(paired)}",
                    case_id,
                    transcript_id,
                    call_id,
                )
            )
            continue
        result_index, result, result_transcript_id = paired[0]
        if result_index <= call_index:
            violations.append(
                Violation(
                    "tool_result_precedes_call",
                    "ToolResult must occur after its ToolCall",
                    case_id,
                    result_transcript_id,
                    call_id,
                )
            )
        if result.get("tool_name") != call.get("tool_name"):
            violations.append(
                Violation(
                    "tool_result_name_mismatch",
                    f"call tool={call.get('tool_name')!r}, result tool={result.get('tool_name')!r}",
                    case_id,
                    result_transcript_id,
                    call_id,
                )
            )
        next_assistant = next(
            (
                index
                for index in range(call_index + 1, len(records))
                if records[index].get("kind") == "assistant_turn"
            ),
            len(records),
        )
        if result_index >= next_assistant:
            violations.append(
                Violation(
                    "tool_result_not_replayed_before_next_turn",
                    "ToolResult must be recorded before the next assistant turn",
                    case_id,
                    result_transcript_id,
                    call_id,
                )
            )
        if result.get("status") == "rejected":
            feedback = any(
                records[index].get("kind") == "protocol_feedback"
                for index in range(result_index + 1, next_assistant)
            )
            if not feedback:
                violations.append(
                    Violation(
                        "rejected_call_feedback_missing",
                        "a rejected ToolCall must be followed by protocol feedback",
                        case_id,
                        result_transcript_id,
                        call_id,
                    )
                )
    for call_id, paired in results.items():
        if call_id not in calls:
            for _, _, transcript_id in paired:
                violations.append(
                    Violation(
                        "orphan_tool_result",
                        "ToolResult has no preceding ToolCall",
                        case_id,
                        transcript_id,
                        call_id,
                    )
                )


def _audit_read_only_observation_ids(
    calls: list[tuple[dict[str, Any], Any]],
    assistant_index: int,
    checkpoints: list[tuple[int, str]],
    violations: list[Violation],
    case_id: str,
    transcript_id: str,
) -> None:
    latest = next(
        (observation_id for index, observation_id in reversed(checkpoints) if index < assistant_index),
        None,
    )
    for call, _ in calls:
        args = _object(call.get("args"))
        if "observation_id" in args and args.get("observation_id") != latest:
            violations.append(
                Violation(
                    "read_only_stale_observation",
                    f"read-only call observation_id={args.get('observation_id')!r}, latest={latest!r}",
                    case_id,
                    transcript_id,
                    _string(call.get("tool_call_id")),
                )
            )


def _audit_checkpoint_image_contract(
    context: dict[str, Any],
    observation_id: str,
    violations: list[Violation],
    case_id: str,
    transcript_id: str,
) -> None:
    """Require one current image reference per chronological checkpoint."""

    fragments = context.get("fragments")
    if not isinstance(fragments, list):
        return
    image_fragment = next(
        (
            item
            for item in fragments
            if isinstance(item, dict) and item.get("name") == "image"
        ),
        None,
    )
    if not isinstance(image_fragment, dict):
        return
    selected = _object(image_fragment.get("content")).get("selected_frames")
    selected = [item for item in selected if isinstance(item, dict)] if isinstance(selected, list) else []
    rgb = [item for item in selected if item.get("role") != "current_depth"]
    if len(rgb) != 1 or rgb[0].get("role") != "current_observation":
        violations.append(
            Violation(
                "checkpoint_image_contract_invalid",
                f"checkpoint must contain one current_observation image reference, received roles={[item.get('role') for item in rgb]}",
                case_id,
                transcript_id,
            )
        )
        return
    if str(rgb[0].get("observation_id") or "") != observation_id:
        violations.append(
            Violation(
                "checkpoint_image_observation_mismatch",
                "checkpoint image observation_id does not match the checkpoint",
                case_id,
                transcript_id,
            )
        )


def _audit_physical_observations(
    records: list[dict[str, Any]],
    batches: list[tuple[int, int, str]],
    checkpoints: list[tuple[int, str]],
    audit: dict[str, Any],
    violations: list[Violation],
    case_id: str,
) -> None:
    automatic = [
        item
        for item in _dict_items(audit.get("observations"))
        if item.get("source") == "automatic_phase_window"
    ]
    if len(automatic) != len(batches):
        violations.append(
            Violation(
                "automatic_observation_batch_mismatch",
                f"physical_batches={len(batches)}, automatic_observations={len(automatic)}",
                case_id,
            )
        )
    for assistant_index, _, transcript_id in batches:
        before = next(
            (obs for index, obs in reversed(checkpoints) if index < assistant_index), None
        )
        next_assistant = next(
            (
                index
                for index in range(assistant_index + 1, len(records))
                if records[index].get("kind") == "assistant_turn"
            ),
            len(records),
        )
        after_checkpoints = [
            obs
            for index, obs in checkpoints
            if assistant_index < index < next_assistant
        ]
        if len(after_checkpoints) != 1:
            violations.append(
                Violation(
                    "post_batch_checkpoint_cardinality",
                    f"physical batch requires one checkpoint before the next assistant turn, received {len(after_checkpoints)}",
                    case_id,
                    transcript_id,
                )
            )
        elif before == after_checkpoints[0]:
            violations.append(
                Violation(
                    "post_batch_observation_not_advanced",
                    f"physical batch reused observation_id {before!r}",
                    case_id,
                    transcript_id,
                )
            )


def _audit_action_replay(
    expected: list[tuple[str, dict[str, Any], dict[str, Any]]],
    audit: dict[str, Any],
    registry: ToolRegistry,
    violations: list[Violation],
    case_id: str,
) -> None:
    actual: list[dict[str, Any]] = []
    for action in _dict_items(audit.get("actions")):
        try:
            spec = registry.get(str(action.get("tool")))
        except KeyError:
            continue
        # A failed episode may have no ``tool_step_completed`` events.  The
        # benchmark reader then retains bounded actions as a compatibility
        # fallback, including shutdown hover.  Only phase-tagged actions are
        # authoritative planner steps for a native transcript replay.
        formal_step = action.get("phase") not in (None, "") or isinstance(
            action.get("step_index"), int
        )
        if spec.runtime.effect == "physical" and formal_step:
            actual.append(action)
    if len(expected) != len(actual):
        violations.append(
            Violation(
                "physical_action_count_mismatch",
                f"successful physical ToolResults={len(expected)}, Action Timeline records={len(actual)}",
                case_id,
            )
        )
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
        tool, executed_args, _ = expected_item
        if actual_item.get("tool") != tool or not _json_equal(
            executed_args, _object(actual_item.get("args"))
        ):
            violations.append(
                Violation(
                    "physical_action_replay_mismatch",
                    f"physical action #{index + 1} does not match executed ToolResult args",
                    case_id,
                )
            )


def _audit_portable_html(
    case_dir: Path,
    audit: dict[str, Any],
    violations: list[Violation],
    case_id: str,
) -> None:
    review = case_dir / "review.html"
    if not review.is_file():
        violations.append(Violation("review_html_missing", str(review), case_id))
    else:
        text = review.read_text(encoding="utf-8", errors="replace")
        if "Agent Tool Transcript" not in text:
            violations.append(
                Violation(
                    "review_transcript_section_missing",
                    "review.html does not contain Agent Tool Transcript",
                    case_id,
                )
            )
    assets_root = (case_dir / "review_assets").resolve()
    for asset in _dict_items(audit.get("frame_assets")):
        if str(asset.get("modality", "")).lower() != "rgb":
            continue
        display = asset.get("display_path")
        if not isinstance(display, str) or not display:
            continue
        path = Path(display).resolve()
        if not path.is_file() or not _is_relative_to(path, assets_root):
            violations.append(
                Violation(
                    "rgb_asset_not_portable",
                    f"RGB display_path must be an existing file under review_assets: {path}",
                    case_id,
                )
            )


def _audit_scene_capabilities(
    records: list[dict[str, Any]],
    case_data: dict[str, Any],
    registry: ToolRegistry,
    violations: list[Violation],
    case_id: str,
) -> None:
    supported = set(_string_items(_nested(case_data, "scene_capabilities", "supported_actions")))
    if not supported:
        return
    for item in records:
        if item.get("kind") != "assistant_turn":
            continue
        for call in _dict_items(_payload(item).get("tool_calls")):
            try:
                spec = registry.get(str(call.get("tool_name")))
            except KeyError:
                continue
            backend = spec.metadata.get("backend_action")
            if spec.runtime.effect == "physical" and backend not in supported:
                violations.append(
                    Violation(
                        "physical_tool_not_supported_by_scene",
                        f"backend action {backend!r} is absent from scene capabilities",
                        case_id,
                        _record_id(item),
                        _string(call.get("tool_call_id")),
                    )
                )


def _observation_sets(audit: dict[str, Any]) -> tuple[set[str], set[str]]:
    observations = {
        str(item.get("id"))
        for item in _dict_items(audit.get("observations"))
        if item.get("id") not in (None, "")
    }
    rgb = {
        str(item.get("observation_id"))
        for item in _dict_items(audit.get("frame_assets"))
        if str(item.get("modality", "")).lower() == "rgb"
        and item.get("observation_id") not in (None, "")
        and item.get("display_path")
    }
    return observations, rgb


def _audit_unique_record_ids(
    records: list[dict[str, Any]], violations: list[Violation], case_id: str
) -> None:
    seen: set[str] = set()
    for item in records:
        record_id = _record_id(item)
        if not record_id:
            violations.append(
                Violation("transcript_id_missing", "transcript item requires id", case_id)
            )
        elif record_id in seen:
            violations.append(
                Violation(
                    "transcript_id_duplicate",
                    f"duplicate transcript id {record_id!r}",
                    case_id,
                    record_id,
                )
            )
        seen.add(record_id)


def _contains_forbidden_history_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in {"agent_transcript", "tool_calls", "tool_results"} for key in value):
            return True
        return any(_contains_forbidden_history_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_history_key(item) for item in value)
    return False


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _read_object(
    path: Path,
    violations: list[Violation],
    *,
    code: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        violations.append(Violation(code, f"{path}: {exc}", case_id=case_id))
        return {}
    if not isinstance(value, dict):
        violations.append(Violation(code, f"{path}: root must be an object", case_id=case_id))
        return {}
    return value


def _payload(item: dict[str, Any]) -> dict[str, Any]:
    return _object(item.get("payload"))


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_items(value: Any) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def _record_id(item: dict[str, Any]) -> str:
    return _string(item.get("id"))


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when any audit violation is found.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_evaluation(args.eval_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "eval_dir": report["eval_dir"],
                "case_count": len(report["cases"]),
                "violation_count": len(report["violations"]),
                "coverage": report["coverage"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 2 if args.strict and report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
