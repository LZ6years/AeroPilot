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
from src.plugins import PluginRegistry
from src.memory.context.contracts import (
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    TRANSCRIPT_SCHEMA,
)


SCHEMA_VERSION = "unified_tool_loop_real_audit_v1"
TERMINAL_RESULT_STATUSES = {"ok", "error", "rejected", "skipped"}
PUBLIC_INTENT_SOURCES = {"provider_content", "tool_arguments_fallback"}


@dataclass(frozen=True)
class Violation:
    """One precise audit failure tied to a case and transcript record."""

    code: str
    message: str
    case_id: str | None = None
    transcript_id: str | None = None
    tool_call_id: str | None = None
    scope: str = "native_transcript"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "case_id": self.case_id,
            "transcript_id": self.transcript_id,
            "tool_call_id": self.tool_call_id,
            "scope": self.scope,
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

    registry = default_registry(PluginRegistry.from_config())
    case_reports: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        case_reports.append(_audit_case(case_dir, registry, violations))

    coverage = {
        "assistant_turns": sum(item["coverage"]["assistant_turns"] for item in case_reports),
        "read_only_turns": sum(item["coverage"]["read_only_turns"] for item in case_reports),
        "control_turns": sum(item["coverage"]["control_turns"] for item in case_reports),
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
        "visual_grounding_successes_requiring_manual_candidate_review": sum(
            item["coverage"][
                "visual_grounding_successes_requiring_manual_candidate_review"
            ]
            for item in case_reports
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
    _audit_physical_trace(
        case_data,
        transcript,
        audit,
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
        "control_turns": 0,
        "physical_batches": 0,
        "multi_action_batches": 0,
        "protocol_feedback_items": 0,
        "provider_rejected_turns": 0,
        "rejected_assistant_turns": 0,
        "reported_blocked_turns": 0,
        "task_terminal_items": 0,
        "metric_depth_successes_requiring_manual_roi_review": 0,
        "visual_grounding_successes_requiring_manual_candidate_review": 0,
    }
    if not records:
        return coverage
    if (
        len(records) < 2
        or records[0].get("kind") != "instruction_envelope"
        or records[1].get("kind") != "user_instruction"
    ):
        violations.append(
            Violation(
                "instruction_not_first",
                "instruction_envelope followed by user_instruction must start the transcript",
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
        if item.get("schema_version") != TRANSCRIPT_SCHEMA:
            violations.append(Violation(
                "transcript_schema_unsupported",
                f"expected {TRANSCRIPT_SCHEMA}",
                case_id,
                transcript_id,
            ))
        if kind == "instruction_envelope" and payload.get("schema_version") != INSTRUCTION_ENVELOPE_SCHEMA:
            violations.append(Violation(
                "instruction_envelope_schema_unsupported",
                f"expected {INSTRUCTION_ENVELOPE_SCHEMA}",
                case_id,
                transcript_id,
            ))
        if kind == "state_checkpoint":
            if payload.get("schema_version") != STATE_CHECKPOINT_SCHEMA:
                violations.append(Violation(
                    "checkpoint_schema_unsupported",
                    f"expected {STATE_CHECKPOINT_SCHEMA}",
                    case_id,
                    transcript_id,
                ))
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
            if _contains_forbidden_history_key(payload):
                violations.append(
                    Violation(
                        "checkpoint_duplicates_transcript",
                        "state checkpoint embeds agent transcript/tool-call history",
                        case_id,
                        transcript_id,
                    )
                )
            _audit_checkpoint_image_contract(
                payload,
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
            if (
                payload.get("tool_name") == "skill.metric_depth.sample_region"
                and payload.get("status") == "ok"
            ):
                coverage[
                    "metric_depth_successes_requiring_manual_roi_review"
                ] += 1
            if (
                payload.get("tool_name") == "skill.visual_grounding.measure_visible_depth"
                and payload.get("status") == "ok"
            ):
                coverage[
                    "visual_grounding_successes_requiring_manual_candidate_review"
                ] += 1
                _audit_visual_grounding_result(
                    payload,
                    violations,
                    case_id,
                    transcript_id,
                    call_id,
                )
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
            if payload.get("completed") is True and payload.get("stopped") is True:
                violations.append(
                    Violation(
                        "task_terminal_state_invalid",
                        "task_terminal cannot be both completed and stopped",
                        case_id,
                        transcript_id,
                    )
                )

    _audit_provider_rejection_feedback(records, violations, case_id)

    _audit_call_result_pairs(records, calls, results, violations, case_id)
    _audit_advisory_state(records, violations, case_id)

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
        controls = [(call, spec) for call, spec in specs if spec.runtime.effect == "runtime_control"]
        plan_read_mix = (
            len(controls) == 1
            and controls[0][0].get("tool_name") == "runtime.update_plan"
            and not physical_calls
            and len(read_calls) + 1 == len(specs)
            and all(spec.runtime.batching == "parallel_safe" for _, spec in read_calls)
        )
        if read_calls:
            coverage["read_only_turns"] += 1
            if len(read_calls) != len(specs) and not plan_read_mix:
                violations.append(
                    Violation(
                        "read_only_turn_mixed",
                        "read-only tools may mix only with one plan update, never terminal or physical tools",
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
            if not controls:
                continue

        if controls:
            coverage["control_turns"] += 1
            if len(controls) != 1 or (len(turn_calls) != 1 and not plan_read_mix):
                violations.append(Violation("control_turn_mixed", "terminal controls are exclusive; plan updates may mix only with parallel-safe reads", case_id, transcript_id))
            for call, _ in controls:
                name = call.get("tool_name")
                if name == "runtime.update_plan":
                    paired = results.get(_string(call.get("tool_call_id")), [])
                    if len(paired) == 1 and paired[0][1].get("status") == "ok":
                        known_refs = _prior_plan_evidence_refs(
                            records, assistant_index, registry
                        )
                        for step in _dict_items(_object(call.get("args")).get("steps")):
                            refs = step.get("evidence_refs")
                            if (
                                not isinstance(refs, list)
                                or any(not isinstance(ref, str) or ref not in known_refs for ref in refs)
                                or (step.get("status") == "completed" and not refs)
                            ):
                                violations.append(Violation(
                                    "plan_evidence_invalid",
                                    "plan evidence must exist before the turn and completed steps require it",
                                    case_id,
                                    transcript_id,
                                    _string(call.get("tool_call_id")),
                                ))
                    continue
                if name not in {"runtime.report_completed", "runtime.report_blocked"}:
                    continue
                paired = results.get(_string(call.get("tool_call_id")), [])
                if len(paired) != 1 or paired[0][1].get("status") != "ok":
                    continue
                if name == "runtime.report_blocked":
                    coverage["reported_blocked_turns"] += 1
                refs = _object(call.get("args")).get("evidence_refs")
                known_refs = _prior_evidence_refs(records, assistant_index, registry)
                if not isinstance(refs, list) or not refs or any(ref not in known_refs for ref in refs if isinstance(ref, str)) or any(not isinstance(ref, str) for ref in refs):
                    violations.append(Violation("terminal_evidence_invalid", "terminal requires prior environmental evidence, not plan or knowledge", case_id, transcript_id))
                if not any(record.get("kind") == "task_terminal" for record in records[assistant_index + 1:]):
                    violations.append(Violation("task_terminal_missing", "accepted terminal must have durable task_terminal", case_id, transcript_id))
                terminals = [_payload(record) for record in records[assistant_index + 1:]
                             if record.get("kind") == "task_terminal"]
                for terminal in terminals:
                    completed = name == "runtime.report_completed"
                    if (terminal.get("completed") is not completed
                            or terminal.get("stopped") is not (not completed)
                            or terminal.get("stop_source") != name):
                        violations.append(Violation("terminal_result_state_mismatch",
                            "durable terminal does not match the accepted terminal call", case_id, transcript_id))
                if any(record.get("kind") == "assistant_turn" for record in records[assistant_index + 1:]):
                    violations.append(Violation("assistant_after_terminal", "model execution must end after accepted terminal", case_id, transcript_id))
            continue
        if len(physical_calls) != len(turn_calls) or not 1 <= len(physical_calls) <= 3:
            violations.append(Violation("physical_batch_size_invalid", "physical turn requires 1-3 physical calls only", case_id, transcript_id))
        if any(spec.runtime.batching != "ordered" for _, spec in physical_calls):
            violations.append(
                Violation(
                    "physical_batching_invalid",
                    "every physical call in a physical window must use ordered batching",
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


def _audit_visual_grounding_result(
    payload: dict[str, Any],
    violations: list[Violation],
    case_id: str,
    transcript_id: str,
    call_id: str,
) -> None:
    """Validate result structure without pretending to judge target identity."""

    requested = payload.get("requested_args")
    requested = requested if isinstance(requested, dict) else {}
    # Absent toggle denotes the measurement-only interface; old requests retain
    # their recorded false/true semantics and are never rewritten.
    include_distance = requested.get("include_distance", True)
    data = payload.get("data")
    data = data if isinstance(data, dict) else {}
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        violations.append(
            Violation(
                "visual_grounding_candidates_invalid",
                "successful visual grounding result requires a candidates list",
                case_id,
                transcript_id,
                call_id,
            )
        )
        return
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            violations.append(
                Violation(
                    "visual_grounding_candidate_invalid",
                    "each grounded candidate must be an object",
                    case_id,
                    transcript_id,
                    call_id,
                )
            )
            continue
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            violations.append(
                Violation(
                    "visual_grounding_candidate_id_invalid",
                    f"candidate_id must be non-empty and unique, received {candidate_id!r}",
                    case_id,
                    transcript_id,
                    call_id,
                )
            )
        else:
            seen.add(candidate_id)
        distance = candidate.get("distance_m")
        distance_status = candidate.get("distance_status")
        if include_distance is False and (
            distance is not None or distance_status != "not_requested"
        ):
            violations.append(
                Violation(
                    "visual_grounding_unrequested_distance",
                    "detection-only result must not contain a measured distance",
                    case_id,
                    transcript_id,
                    call_id,
                )
            )
        if include_distance is True and distance is not None:
            if not isinstance(distance, (int, float)) or not math.isfinite(float(distance)) or float(distance) <= 0:
                violations.append(
                    Violation(
                        "visual_grounding_distance_invalid",
                        f"distance_m must be a finite positive number, received {distance!r}",
                        case_id,
                        transcript_id,
                        call_id,
                    )
                )
            if distance_status != "ok":
                violations.append(
                    Violation(
                        "visual_grounding_distance_status_invalid",
                        "a numeric distance requires distance_status='ok'",
                        case_id,
                        transcript_id,
                        call_id,
                    )
                )


def _audit_physical_trace(
    case_data: dict[str, Any],
    records: list[dict[str, Any]],
    audit: dict[str, Any],
    registry: ToolRegistry,
    violations: list[Violation],
    case_id: str,
) -> None:
    """Validate window-v2 evidence independently from transcript protocol."""

    def trace_violation(code: str, message: str) -> None:
        violations.append(
            Violation(
                code,
                message,
                case_id=case_id,
                scope="physical_trace",
            )
        )

    trace = case_data.get("trace")
    if not isinstance(trace, dict):
        trace_violation("physical_trace_missing", "case trace must be an object")
        return
    if trace.get("trace_schema_version") != "physical_window_v2":
        trace_violation(
            "physical_trace_schema_unsupported",
            "strict audit requires physical_window_v2",
        )
        return
    if trace.get("trace_integrity") != "ok":
        trace_violation(
            "physical_trace_invalid",
            f"trace_integrity={trace.get('trace_integrity')!r}, issues={trace.get('trace_issues')!r}",
        )
    windows = trace.get("physical_windows")
    if not isinstance(windows, list):
        trace_violation("physical_windows_invalid", "physical_windows must be a list")
        windows = []

    observation_by_id = {
        str(item.get("id")): item
        for item in _dict_items(audit.get("observations"))
        if item.get("id") not in (None, "")
    }
    trace_steps: dict[str, dict[str, Any]] = {}
    after_ids: set[str] = set()
    step_count = 0
    motion_count = 0
    collision_count = 0
    collision_window_count = 0
    for window in windows:
        if not isinstance(window, dict):
            trace_violation("physical_window_invalid", "physical window must be an object")
            continue
        if window.get("status") == "collision":
            collision_window_count += 1
        before_id = window.get("observation_before_id")
        after_id = window.get("observation_after_id")
        if not isinstance(before_id, str) or not isinstance(after_id, str):
            trace_violation(
                "physical_window_observation_missing",
                "physical window requires before/after observation IDs",
            )
        elif before_id == after_id:
            trace_violation(
                "physical_window_observation_not_advanced",
                f"physical window reused {before_id!r}",
            )
        else:
            if before_id not in observation_by_id:
                trace_violation(
                    "physical_window_before_observation_unknown",
                    f"before observation {before_id!r} is absent from audit",
                )
            after_observation = observation_by_id.get(after_id)
            if after_observation is None:
                trace_violation(
                    "physical_window_after_observation_unknown",
                    f"after observation {after_id!r} is absent from audit",
                )
            elif after_observation.get("source") != "automatic_physical_window":
                trace_violation(
                    "physical_window_after_observation_source_invalid",
                    f"after observation {after_id!r} is not automatic_physical_window",
                )
            if after_id in after_ids:
                trace_violation(
                    "physical_window_after_observation_duplicate",
                    f"after observation {after_id!r} is reused by multiple windows",
                )
            after_ids.add(after_id)
        steps = window.get("ordered_steps")
        if not isinstance(steps, list) or not steps:
            trace_violation(
                "physical_window_steps_missing",
                "physical window requires ordered physical steps",
            )
            continue
        for expected_index, step in enumerate(steps):
            step_count += 1
            if not isinstance(step, dict):
                trace_violation("physical_step_invalid", "physical step must be an object")
                continue
            if step.get("step_index") != expected_index:
                trace_violation(
                    "physical_step_order_invalid",
                    f"expected step_index {expected_index}, got {step.get('step_index')!r}",
                )
            if step.get("effect") != "physical":
                trace_violation(
                    "non_physical_step_in_trace",
                    f"trace step effect is {step.get('effect')!r}",
                )
            tool = str(step.get("tool") or "")
            if (
                step.get("status") in {"ok", "collision"}
                and tool in {"uav.move_body", "uav.turn_relative"}
            ):
                motion_count += 1
            if step.get("status") == "collision":
                collision_count += 1
            call_id = step.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                trace_violation(
                    "physical_step_tool_call_id_missing",
                    "native physical trace step requires tool_call_id",
                )
            elif call_id in trace_steps:
                trace_violation(
                    "physical_step_tool_call_id_duplicate",
                    f"duplicate physical tool_call_id {call_id!r}",
                )
            else:
                trace_steps[call_id] = step
            if step.get("status") in {"ok", "collision", "error"} and not step.get("action_ref"):
                trace_violation(
                    "physical_step_action_ref_missing",
                    f"executed step {call_id!r} lacks action_ref",
                )

    if trace.get("physical_call_count") != step_count:
        trace_violation(
            "physical_call_count_mismatch",
            f"declared={trace.get('physical_call_count')!r}, actual={step_count}",
        )
    if trace.get("physical_window_count") != len(windows):
        trace_violation(
            "physical_window_count_mismatch",
            f"declared={trace.get('physical_window_count')!r}, actual={len(windows)}",
        )
    if trace.get("motion_call_count") != motion_count:
        trace_violation(
            "motion_call_count_mismatch",
            f"declared={trace.get('motion_call_count')!r}, actual={motion_count}",
        )
    if trace.get("collision_call_count", 0) != collision_count:
        trace_violation(
            "collision_call_count_mismatch",
            f"declared={trace.get('collision_call_count')!r}, actual={collision_count}",
        )
    if trace.get("collision_window_count", 0) != collision_window_count:
        trace_violation(
            "collision_window_count_mismatch",
            f"declared={trace.get('collision_window_count')!r}, actual={collision_window_count}",
        )

    transcript_steps: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    results: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("kind") != "tool_result":
            continue
        payload = _payload(record)
        call_id = payload.get("tool_call_id")
        if isinstance(call_id, str):
            results[call_id] = payload
    for record in records:
        if record.get("kind") != "assistant_turn":
            continue
        calls = _payload(record).get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("tool_call_id")
            if not isinstance(call_id, str):
                continue
            try:
                spec = registry.get(str(call.get("tool_name") or ""))
            except KeyError:
                continue
            result = results.get(call_id)
            if (
                spec.runtime.effect == "physical"
                and result is not None
                and result.get("status") in {"ok", "collision", "error", "skipped"}
            ):
                transcript_steps[call_id] = (call, result)
    if set(trace_steps) != set(transcript_steps):
        trace_violation(
            "physical_trace_transcript_mismatch",
            f"trace IDs={sorted(trace_steps)}, transcript IDs={sorted(transcript_steps)}",
        )
    for call_id in sorted(set(trace_steps) & set(transcript_steps)):
        step = trace_steps[call_id]
        call, result = transcript_steps[call_id]
        if (step.get("requested_tool") or step.get("tool")) != call.get("tool_name"):
            trace_violation(
                "physical_trace_tool_mismatch",
                f"{call_id}: trace requested={step.get('requested_tool')!r}, transcript={call.get('tool_name')!r}",
            )
        if step.get("status") != result.get("status"):
            trace_violation(
                "physical_trace_status_mismatch",
                f"{call_id}: trace={step.get('status')!r}, ToolResult={result.get('status')!r}",
            )
        if not _json_equal(step.get("requested_args"), result.get("requested_args")):
            trace_violation(
                "physical_trace_requested_args_mismatch",
                f"{call_id}: requested args differ from ToolResult",
            )
        if not _json_equal(step.get("executed_args"), result.get("executed_args")):
            trace_violation(
                "physical_trace_executed_args_mismatch",
                f"{call_id}: executed args differ from ToolResult",
            )
        if step.get("action_ref") != result.get("result_ref"):
            trace_violation(
                "physical_trace_action_ref_mismatch",
                f"{call_id}: action_ref differs from ToolResult result_ref",
            )
        if not _json_equal(
            step.get("safety_interventions"),
            result.get("safety_interventions"),
        ):
            trace_violation(
                "physical_trace_safety_mismatch",
                f"{call_id}: safety interventions differ from ToolResult",
            )

    tracking = case_data.get("backend_action_tracking")
    if not isinstance(tracking, dict):
        violations.append(
            Violation(
                "window_tracking_missing",
                "backend_action_tracking must be an object",
                case_id=case_id,
                scope="tracking",
            )
        )
        return
    for key, expected in (
        ("schema_version", "uav_flow_window_tracking_v2"),
        ("association_source", "explicit_runtime_window"),
        ("trace_valid", True),
        ("tracking_valid", True),
    ):
        if tracking.get(key) != expected:
            violations.append(
                Violation(
                    "window_tracking_invalid",
                    f"{key} expected {expected!r}, got {tracking.get(key)!r}",
                    case_id=case_id,
                    scope="tracking",
                )
            )


def _prior_evidence_refs(records, assistant_index, registry) -> set[str]:
    """Only successful environmental results and preceding observations count."""
    refs, pending = set(), {}
    for item in records[:assistant_index]:
        payload = _payload(item)
        if item.get("kind") == "user_instruction":
            refs.clear()
            pending.clear()
        elif item.get("kind") == "state_checkpoint" and payload.get("observation_id"):
            refs.add(payload["observation_id"])
        elif item.get("kind") == "assistant_turn":
            pending.update((call.get("tool_call_id"), call.get("tool_name"))
                           for call in _dict_items(payload.get("tool_calls")))
        elif item.get("kind") == "tool_result" and payload.get("status") == "ok":
            name = pending.get(payload.get("tool_call_id"))
            if name != payload.get("tool_name"):
                continue
            try:
                spec = registry.get(name)
            except KeyError:
                continue
            if spec.runtime.effect not in {"physical", "read_only"} or spec.kind == "knowledge":
                continue
            refs.add(payload["tool_call_id"])
            if payload.get("result_ref"):
                refs.add(payload["result_ref"])
    return refs


def _prior_plan_evidence_refs(records, assistant_index, registry) -> set[str]:
    """Observations and successful environmental/read results before a plan turn."""

    refs = set()
    for item in records[:assistant_index]:
        payload = _payload(item)
        if item.get("kind") == "user_instruction":
            refs.clear()
        elif item.get("kind") == "state_checkpoint" and payload.get("observation_id"):
            refs.add(payload["observation_id"])
        elif item.get("kind") == "tool_result" and payload.get("status") in {"ok", "collision"}:
            try:
                spec = registry.get(payload.get("tool_name"))
            except KeyError:
                continue
            if spec.runtime.effect not in {"physical", "read_only"}:
                continue
            if payload.get("tool_call_id"):
                refs.add(payload["tool_call_id"])
            if payload.get("result_ref"):
                refs.add(payload["result_ref"])
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
    checkpoint: dict[str, Any],
    observation_id: str,
    violations: list[Violation],
    case_id: str,
    transcript_id: str,
) -> None:
    """Require one current image reference per chronological checkpoint."""

    fragments = checkpoint.get("fragments")
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
    inputs = image_fragment.get("media_refs")
    inputs = [item for item in inputs if isinstance(item, dict)] if isinstance(inputs, list) else []
    refs = [item.get("media_ref") for item in inputs if isinstance(item.get("media_ref"), dict)]
    rgb = [item for item in refs if item.get("role") != "current_depth"]
    unavailable = _object(image_fragment.get("content")).get("visual_evidence") == "current_rgb_unavailable"
    if unavailable and not rgb:
        return
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


def _audit_advisory_state(records, violations, case_id):
    """Replay only pure advisory contracts, never execute Runtime or plugins.

    Results are authoritative application receipts; a proposal or rejection
    cannot advance the audited revision. Knowledge text is checked for delivery
    provenance, not interpreted as an instruction or environmental evidence.
    """
    from src.agent.planning.task_plan import TaskPlan

    plan = None
    calls = {}
    knowledge = {}
    for record in records:
        payload = _payload(record)
        kind = record.get("kind")
        def fail(code, message):
            violations.append(Violation(code, message, case_id, _record_id(record)))
        if kind == "assistant_turn":
            for call in _dict_items(payload.get("tool_calls")):
                if isinstance(call.get("tool_call_id"), str):
                    calls[call["tool_call_id"]] = call
        elif kind == "state_checkpoint":
            if "task_plan" in payload and not _json_equal(payload["task_plan"], plan.to_dict() if plan else None):
                fail("checkpoint_plan_mismatch", "checkpoint plan differs from successfully applied updates")
        elif kind == "tool_result":
            call = calls.get(_string(payload.get("tool_call_id")))
            if not call:
                continue  # Pairing errors have their own diagnostics.
            name = call.get("tool_name")
            args, data = _object(call.get("args")), _object(payload.get("data"))
            if payload.get("status") != "ok":
                if name == "runtime.update_plan" and "plan" in data:
                    fail("rejected_plan_applied", "failed plan result must not contain an applied plan")
                continue
            if name == "runtime.update_plan":
                previous = plan or TaskPlan()
                try:
                    expected = previous.updated(args)
                except ValueError as exc:
                    fail("applied_plan_invalid", str(exc))
                    continue
                if (data.get("previous_revision") != previous.revision
                        or not _json_equal(data.get("plan"), expected.to_dict())):
                    fail("plan_application_mismatch", "plan receipt differs from requested update or revision")
                plan = expected
            elif name == "knowledge.load":
                entry = args.get("name")
                version = data.get("version")
                if not isinstance(entry, str):
                    fail("knowledge_provenance_invalid", "knowledge name must be a string")
                    continue
                prior = knowledge.get(entry)
                refs = data.get("references")
                manifest = data.get("reference_manifest")
                if (data.get("name") != entry or not isinstance(version, str) or not version
                        or not isinstance(refs, list) or not isinstance(manifest, list)):
                    fail("knowledge_provenance_invalid", "load receipt lacks name/version/reference provenance")
                    continue
                delivered = set() if prior is None else prior[1]
                requested = args.get("reference_paths", [])
                declared = {item["path"] for item in _dict_items(manifest) if isinstance(item.get("path"), str)}
                paths = [item.get("path") for item in _dict_items(refs)]
                first = prior is None
                if (not isinstance(requested, list) or any(not isinstance(p, str) for p in requested)
                        or any(p not in declared for p in requested)):
                    fail("knowledge_reference_invalid", "load requested undeclared references")
                    continue
                if (data.get("body_included") is not first
                        or (first and not isinstance(data.get("body"), str))
                        or (not first and "body" in data)
                        or (prior and prior[0] != version)
                        or paths != [p for p in requested if p not in delivered]
                        or any(not isinstance(item.get("text"), str) for item in _dict_items(refs))):
                    fail("knowledge_delivery_mismatch", "load does not preserve version or incremental body/reference delivery")
                knowledge[entry] = (version, delivered | set(requested))


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
        if item.get("source") == "automatic_physical_window"
    ]
    # Terminal refreshes and read-only checkpoints are not batch observations.
    # Physical-window before/after and ordered-call references are independently
    # checked by _audit_physical_trace; here verify their transcript placement.
    automatic_ids = {str(item.get("id")) for item in automatic}
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
            if assistant_index < index < next_assistant and obs in automatic_ids
        ]
        if len(after_checkpoints) != 1:
            violations.append(
                Violation(
                    "post_batch_checkpoint_cardinality",
                    f"physical batch requires one automatic after checkpoint before the next assistant turn, received {len(after_checkpoints)}",
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
        # The audit reader joins v2 actions by physical-window references.
        # No phase labels or timestamp fallback may decide membership here.
        if spec.runtime.effect == "physical" and _object(action.get("result")).get("ok") is True:
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
        # The compact report keeps the causal timeline and links the durable
        # transcript instead of embedding a second, full transcript view.
        compact = (
            '<section class="phase-timeline">' in text
            and 'href="audit.json"' in text
            and (case_dir / "audit.json").is_file()
        )
        if "Agent Tool Transcript" not in text and not compact:
            violations.append(
                Violation(
                    "review_transcript_section_missing",
                    "review.html needs a transcript view or a phase timeline with a durable audit link",
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
