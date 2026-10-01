"""Tests for the offline native ToolCall real-run audit utility."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from src.memory.context.contracts import (
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    TRANSCRIPT_SCHEMA,
)


SCRIPT = Path(__file__).parent / "manual" / "native_tool_loop_real_audit.py"
SPEC = importlib.util.spec_from_file_location("native_tool_loop_real_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value) -> None:
    if isinstance(value, dict) and isinstance(value.get("agent_transcript"), list):
        for item in value["agent_transcript"]:
            if isinstance(item, dict):
                item.setdefault("schema_version", TRANSCRIPT_SCHEMA)
                item.setdefault("instruction_id", "inst_1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_batch_observation_audit_does_not_count_terminal_refresh():
    records = [
        {"kind": "state_checkpoint"}, {"kind": "assistant_turn"},
        {"kind": "tool_result"}, {"kind": "state_checkpoint"},
        {"kind": "state_checkpoint"}, {"kind": "task_terminal"},
    ]
    audit = {"observations": [
        {"id": "obs2", "source": "automatic_physical_window"},
        {"id": "obs3", "source": "task_terminal"},
    ]}
    violations = []
    MODULE._audit_physical_observations(records, [(1, 2, "turn1")],
                                        [(0, "obs1"), (3, "obs2"), (4, "obs3")],
                                        audit, violations, "case")
    assert violations == []
    audit["observations"][1]["source"] = "automatic_physical_window"
    MODULE._audit_physical_observations(records, [(1, 2, "turn1")],
                                        [(0, "obs1"), (3, "obs2"), (4, "obs3")],
                                        audit, violations, "case")
    assert "post_batch_checkpoint_cardinality" in {v.code for v in violations}


def test_advisory_audit_checks_applied_revision_and_rejected_updates():
    from src.agent.planning.task_plan import TaskPlan
    args = {"explanation": "Reach endpoint", "steps": [
        {"id": "go", "description": "Move to endpoint", "status": "in_progress"}]}
    plan = TaskPlan().updated(args).to_dict()
    records = [
        {"kind": "assistant_turn", "payload": {"tool_calls": [
            {"tool_call_id": "p", "tool_name": "runtime.update_plan", "args": args}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "p", "status": "ok",
            "data": {"previous_revision": 0, "plan": plan}}},
        {"kind": "state_checkpoint", "payload": {"task_plan": plan}},
    ]
    violations = []
    MODULE._audit_advisory_state(records, violations, "case")
    assert not violations
    records[1]["payload"]["status"] = "rejected"
    MODULE._audit_advisory_state(records, violations, "case")
    assert {v.code for v in violations} == {"rejected_plan_applied", "checkpoint_plan_mismatch"}


def test_advisory_audit_checks_incremental_knowledge_and_version():
    def load(identity, refs, data):
        return [
            {"kind": "assistant_turn", "payload": {"tool_calls": [{"tool_call_id": identity,
                "tool_name": "knowledge.load", "args": {"name": "guide", "reference_paths": refs}}]}},
            {"kind": "tool_result", "payload": {"tool_call_id": identity, "status": "ok", "data": data}},
        ]
    base = {"name": "guide", "version": "sha", "reference_manifest": [{"path": "ref.md"}]}
    records = load("k1", [], {**base, "body_included": True, "body": "guidance", "references": []})
    records += load("k2", ["ref.md"], {**base, "body_included": False,
        "references": [{"path": "ref.md", "text": "detail"}]})
    violations = []
    MODULE._audit_advisory_state(records, violations, "case")
    assert not violations
    records[-1]["payload"]["data"]["version"] = "changed"
    MODULE._audit_advisory_state(records, violations, "case")
    assert {v.code for v in violations} == {"knowledge_delivery_mismatch"}


def test_compact_report_uses_timeline_and_durable_audit(tmp_path):
    page = tmp_path / "review.html"
    page.write_text('<section class="phase-timeline"></section><a href="audit.json">Audit</a>', encoding="utf-8")
    violations = []
    MODULE._audit_portable_html(tmp_path, {}, violations, "case")
    assert [v.code for v in violations] == ["review_transcript_section_missing"]
    _write_json(tmp_path / "audit.json", {"agent_transcript": []})
    violations = []
    MODULE._audit_portable_html(tmp_path, {}, violations, "case")
    assert violations == []
    page.write_text('<a href="audit.json">Audit only</a>', encoding="utf-8")
    MODULE._audit_portable_html(tmp_path, {}, violations, "case")
    assert [v.code for v in violations] == ["review_transcript_section_missing"]


def _valid_eval(tmp_path: Path) -> Path:
    root = tmp_path / "eval"
    case_dir = root / "cases" / "case-1"
    assets = case_dir / "review_assets" / "frames"
    assets.mkdir(parents=True)
    rgb1 = assets / "frame_1_rgb.png"
    rgb2 = assets / "frame_2_rgb.png"
    rgb1.write_bytes(b"png-one")
    rgb2.write_bytes(b"png-two")
    (case_dir / "review.html").write_text(
        "<h2>Agent Tool Transcript</h2>", encoding="utf-8"
    )
    _write_json(
        root / "summary.json",
        {
            "run_status": "completed",
            "expected_case_count": 1,
            "finalized_case_count": 1,
        },
    )
    _write_json(
        root / "scene_session.json",
        {"start_count": 1, "restart_count": 0, "stop_count": 1},
    )
    _write_json(
        case_dir / "case.json",
        {
            "case": {"class_name": "Move"},
            "status": "ok",
            "completed": True,
            "stopped_reason": None,
            "trace_valid": True,
            "tracking_valid": True,
            "infrastructure_clean": True,
            "trace": {
                "trace_schema_version": "physical_window_v2",
                "trace_integrity": "ok",
                "trace_issues": [],
                "action_count": 2,
                "physical_call_count": 2,
                "physical_window_count": 1,
                "motion_call_count": 2,
                "samples": [],
                "coordinate_frames": ["unrealcv_world_m"],
                "physical_windows": [
                    {
                        "decision_id": "decision_0001",
                        "phase": None,
                        "phase_key": None,
                        "window_index": 1,
                        "status": "ok",
                        "observation_before_id": "obs_1",
                        "observation_after_id": "obs_2",
                        "observation_before_pose": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 1.0,
                            "yaw_deg": 0.0,
                            "coordinate_frame": "unrealcv_world_m",
                        },
                        "observation_after_pose": {
                            "x": 0.49,
                            "y": 0.09,
                            "z": 1.0,
                            "yaw_deg": 10.0,
                            "coordinate_frame": "unrealcv_world_m",
                        },
                        "ordered_steps": [
                            {
                                "step_index": 0,
                                "tool_call_id": "turn_1",
                                "tool": "uav.turn_relative",
                                "effect": "physical",
                                "status": "ok",
                                "action_ref": "memory/actions.jsonl#1",
                                "result_pose_ref": "memory/poses.jsonl#1",
                                "requested_args": {"yaw_deg": 10.0},
                                "executed_args": {"yaw_deg": 10.0},
                                "safety_interventions": [],
                            },
                            {
                                "step_index": 1,
                                "tool_call_id": "move_1",
                                "tool": "uav.move_body",
                                "effect": "physical",
                                "status": "ok",
                                "action_ref": "memory/actions.jsonl#2",
                                "result_pose_ref": "memory/poses.jsonl#2",
                                "requested_args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                                "executed_args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                                "safety_interventions": [],
                            },
                        ],
                    }
                ],
            },
            "backend_action_tracking": {
                "schema_version": "uav_flow_window_tracking_v2",
                "association_source": "explicit_runtime_window",
                "trace_valid": True,
                "tracking_valid": True,
                "physical_call_count": 2,
                "motion_call_count": 2,
                "physical_window_count": 1,
                "tracked_window_count": 1,
                "observed_window_count": 1,
                "unavailable_window_count": 0,
                "zero_motion_completion": False,
                "all_observed_within_tolerance": True,
                "windows": [],
            },
            "scene_capabilities": {
                "supported_actions": ["turn_relative", "move_body", "hover"]
            },
        },
    )
    transcript = [
        {
            "id": "turn_0",
            "kind": "instruction_envelope",
            "payload": {
                "schema_version": INSTRUCTION_ENVELOPE_SCHEMA,
                "instruction_id": "inst_1",
                "protocol_context": "",
                "identity": {},
                "user_preferences": {},
                "knowledge_catalog": [],
                "static_module_guidance": [],
            },
        },
        {
            "id": "turn_1",
            "kind": "user_instruction",
            "payload": {"instruction": "turn and move"},
        },
        {
            "id": "turn_2",
            "kind": "state_checkpoint",
            "payload": {
                "schema_version": STATE_CHECKPOINT_SCHEMA,
                "checkpoint_id": "checkpoint_0001",
                "reason": "instruction_start",
                "observation_id": "obs_1",
                "task_plan": None,
                "loaded_knowledge": [],
                "fragments": [],
                "fingerprint": "fixture-1",
            },
        },
        {
            "id": "turn_3",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "Turn and move in one bounded window.",
                "tool_calls": [
                    {
                        "tool_call_id": "turn_1",
                        "tool_name": "uav.turn_relative",
                        "args": {"yaw_deg": 10.0},
                    },
                    {
                        "tool_call_id": "move_1",
                        "tool_name": "uav.move_body",
                        "args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                    },
                ],
            },
        },
        {
            "id": "turn_4",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "turn_1",
                "tool_name": "uav.turn_relative",
                "status": "ok",
                "requested_args": {"yaw_deg": 10.0},
                "executed_args": {"yaw_deg": 10.0},
                "safety_interventions": [],
                "result_ref": "memory/actions.jsonl#1",
            },
        },
        {
            "id": "turn_5",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "move_1",
                "tool_name": "uav.move_body",
                "status": "ok",
                "requested_args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                "executed_args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                "safety_interventions": [],
                "result_ref": "memory/actions.jsonl#2",
            },
        },
        {
            "id": "turn_7",
            "kind": "state_checkpoint",
            "payload": {
                "schema_version": STATE_CHECKPOINT_SCHEMA,
                "checkpoint_id": "checkpoint_0002",
                "reason": "physical_window_completed",
                "observation_id": "obs_2",
                "task_plan": None,
                "loaded_knowledge": [],
                "fragments": [],
                "fingerprint": "fixture-2",
            },
        },
        {
            "id": "turn_8",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "The requested motion is now complete.",
                "tool_calls": [
                    {
                        "tool_call_id": "done",
                        "tool_name": "runtime.report_completed",
                        "args": {"reason": "Observed endpoint", "evidence_refs": ["obs_2"]},
                    }
                ],
            },
        },
        {
            "id": "turn_9",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "done",
                "tool_name": "runtime.report_completed",
                "status": "ok",
                "requested_args": {},
                "executed_args": {},
                "safety_interventions": [],
            },
        },
        {"id": "terminal", "kind": "task_terminal", "payload": {
            "completed": True, "stopped": False, "stop_source": "runtime.report_completed"}},
    ]
    for item in transcript:
        item["schema_version"] = TRANSCRIPT_SCHEMA
        item["instruction_id"] = "inst_1"
    _write_json(
        case_dir / "audit.json",
        {
            "agent_transcript": transcript,
            "observations": [
                {"id": "obs_1", "source": "preflight"},
                {"id": "obs_2", "source": "automatic_physical_window"},
            ],
            "actions": [
                {
                    "tool": "uav.turn_relative",
                    "result": {"ok": True},
                    "args": {"yaw_deg": 10.0},
                    "phase": None,
                    "step_index": 0,
                },
                {
                    "tool": "uav.move_body",
                    "result": {"ok": True},
                    "args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 0.5},
                    "phase": None,
                    "step_index": 1,
                },
            ],
            "frame_assets": [
                {
                    "observation_id": "obs_1",
                    "modality": "rgb",
                    "display_path": str(rgb1),
                },
                {
                    "observation_id": "obs_2",
                    "modality": "rgb",
                    "display_path": str(rgb2),
                },
            ],
        },
    )
    return root


def test_audit_accepts_closed_multi_action_transcript(tmp_path):
    report = MODULE.audit_evaluation(_valid_eval(tmp_path))

    assert report["status"] == "ok", report["violations"]
    assert report["violations"] == []
    assert report["coverage"]["physical_batches"] == 1
    assert report["coverage"]["multi_action_batches"] == 1


def test_audit_accepts_detection_only_visual_grounding_and_marks_manual_review(
    tmp_path,
):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["agent_transcript"][3:3] = [
        {
            "id": "grounding_turn",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "Locate the current visual reference.",
                "tool_calls": [
                    {
                        "tool_call_id": "ground_1",
                        "tool_name": "skill.visual_grounding.measure_visible_depth",
                        "args": {
                            "observation_id": "obs_1",
                            "query": "the person ahead",
                            "include_distance": False,
                        },
                    }
                ],
            },
        },
        {
            "id": "grounding_result",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "ground_1",
                "tool_name": "skill.visual_grounding.measure_visible_depth",
                "status": "ok",
                "requested_args": {
                    "observation_id": "obs_1",
                    "query": "the person ahead",
                    "include_distance": False,
                },
                "executed_args": {
                    "observation_id": "obs_1",
                    "query": "the person ahead",
                    "include_distance": False,
                },
                "safety_interventions": [],
                "data": {
                    "observation_id": "obs_1",
                    "query": "the person ahead",
                    "include_distance": False,
                    "candidates": [
                        {
                            "candidate_id": "candidate_01",
                            "label": "person",
                            "confidence": 0.91,
                            "box_normalized": [0.4, 0.2, 0.6, 0.8],
                            "distance_m": None,
                            "distance_status": "not_requested",
                        }
                    ],
                },
            },
        },
    ]
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)

    assert report["status"] == "ok", report["violations"]
    assert (
        report["coverage"][
            "visual_grounding_successes_requiring_manual_candidate_review"
        ]
        == 1
    )


def test_audit_rejects_legacy_trace_and_unavailable_window_tracking(tmp_path):
    root = _valid_eval(tmp_path)
    case_path = root / "cases" / "case-1" / "case.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["trace"]["trace_schema_version"] = None
    payload["trace"]["trace_integrity"] = "unsupported_legacy_trace_schema"
    payload["backend_action_tracking"]["tracking_valid"] = False
    _write_json(case_path, payload)

    report = MODULE.audit_evaluation(root)

    scopes = {item["scope"] for item in report["violations"]}
    codes = {item["code"] for item in report["violations"]}
    assert "physical_trace" in scopes
    assert "physical_trace_schema_unsupported" in codes


def test_audit_accepts_generic_report_blocked_terminal(tmp_path):
    root = _valid_eval(tmp_path)
    case_dir = root / "cases" / "case-1"
    audit_path = case_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["agent_transcript"][7:] = [
        {
            "id": "turn_8",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "No safe evidence-producing action remains.",
                "tool_calls": [
                    {
                        "tool_call_id": "report_blocked",
                        "tool_name": "runtime.report_blocked",
                        "args": {
                            "blocking_condition": "insufficient_evidence",
                            "reason": "No safe evidence-producing action remains.",
                            "evidence_refs": ["obs_2", "move_1"],
                        },
                    },
                ],
            },
        },
        {
            "id": "turn_9",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "report_blocked",
                "tool_name": "runtime.report_blocked",
                "status": "ok",
            },
        },
        {
            "id": "turn_11",
            "kind": "task_terminal",
            "payload": {
                "completed": False,
                "stopped": True,
                "stopped_reason": "agent_reported_blocked:insufficient_evidence",
                "stop_source": "runtime.report_blocked",
            },
        },
    ]
    _write_json(audit_path, audit)
    case_json = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    case_json["completed"] = False
    case_json["stopped_reason"] = "agent_reported_blocked:insufficient_evidence"
    _write_json(case_dir / "case.json", case_json)

    report = MODULE.audit_evaluation(root)

    assert report["status"] == "ok", report["violations"]
    assert report["coverage"]["reported_blocked_turns"] == 1
    assert report["coverage"]["task_terminal_items"] == 1


def test_audit_rejects_report_blocked_with_unknown_evidence_and_physical_call(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    turn = audit["agent_transcript"][3]["payload"]
    turn["tool_calls"].insert(
        1,
        {
            "tool_call_id": "report_bad",
            "tool_name": "runtime.report_blocked",
            "args": {
                "blocking_condition": "insufficient_evidence",
                "reason": "Cannot continue.",
                "evidence_refs": ["obs_unknown"],
            },
        },
    )
    audit["agent_transcript"].insert(
        4,
        {
            "id": "result_report_bad",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "report_bad",
                "tool_name": "runtime.report_blocked",
                "status": "ok",
            },
        },
    )
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "control_turn_mixed" in codes
    assert "terminal_evidence_invalid" in codes


def test_audit_rejects_duplicate_missing_and_orphan_results(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transcript = audit["agent_transcript"]
    transcript[3]["payload"]["tool_calls"][1]["tool_call_id"] = "turn_1"
    transcript[:] = [
        item
        for item in transcript
        if item.get("payload", {}).get("tool_call_id") != "move_1"
    ]
    transcript.insert(
        5,
        {
            "id": "orphan",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "never_called",
                "tool_name": "uav.hover",
                "status": "ok",
            },
        },
    )
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "tool_call_id_duplicate" in codes
    assert "orphan_tool_result" in codes
    assert report["status"] == "violations_found"


def test_audit_rejects_stale_read_only_observation_and_mixed_turn(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transcript = audit["agent_transcript"]
    transcript[3]["payload"]["tool_calls"].insert(
        0,
        {
            "tool_call_id": "detect_1",
            "tool_name": "uav.get_pose",
            "args": {"observation_id": "obs_old"},
        },
    )
    transcript.insert(
        4,
        {
            "id": "detect_result",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "detect_1",
                "tool_name": "uav.get_pose",
                "status": "ok",
            },
        },
    )
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "read_only_turn_mixed" in codes
    assert "read_only_stale_observation" in codes


def test_audit_rejects_silent_safety_change_and_nonportable_rgb(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    turn_result = next(
        item
        for item in audit["agent_transcript"]
        if item.get("payload", {}).get("tool_call_id") == "turn_1"
    )
    turn_result["payload"]["executed_args"] = {"yaw_deg": 5.0}
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not portable")
    audit["frame_assets"][0]["display_path"] = str(outside)
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "silent_safety_argument_change" in codes
    assert "physical_action_replay_mismatch" in codes
    assert "rgb_asset_not_portable" in codes


def test_audit_reports_exhausted_native_tool_loop(tmp_path):
    root = _valid_eval(tmp_path)
    case_dir = root / "cases" / "case-1"
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    case.update(
        {
            "status": "error",
            "completed": False,
            "error": {
                "error_type": "RuntimeError",
                "error": "Agent native tool loop exhausted 6 rounds without a valid phase commit",
            },
        }
    )
    _write_json(case_dir / "case.json", case)
    audit = json.loads((case_dir / "audit.json").read_text(encoding="utf-8"))
    audit["agent_transcript"] = audit["agent_transcript"][:3] + [
        {
            "id": f"feedback_{index}",
            "kind": "protocol_feedback",
            "payload": {"message": "Return a corrected native tool-call turn."},
        }
        for index in range(6)
    ]
    audit["observations"] = audit["observations"][:1]
    audit["actions"] = [
        {
            "tool": "uav.hover",
            "args": {},
            "result": {"ok": True},
            # No phase/step_index: this is shutdown safety, not a Planner call.
        }
    ]
    audit["frame_assets"] = audit["frame_assets"][:1]
    _write_json(case_dir / "audit.json", audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "case_execution_error" in codes
    assert "native_tool_loop_exhausted" in codes
    assert "assistant_turn_missing" in codes


def test_audit_accepts_fallback_intent_and_closed_provider_rejection(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transcript = audit["agent_transcript"]
    transcript[3]["payload"]["public_intent_source"] = "tool_arguments_fallback"
    transcript[3:3] = [
        {
            "id": "provider_rejected",
            "kind": "provider_turn_rejected",
            "payload": {
                "error": "unknown provider tool",
                "finish_reason": "tool_calls",
                "content_present": False,
                "content": None,
                "tool_calls": [
                    {
                        "tool_call_id": "wire_bad",
                        "wire_name": "unknown_tool",
                        "arguments": {},
                        "arguments_status": "parsed",
                    }
                ],
            },
        },
        {
            "id": "provider_feedback",
            "kind": "protocol_feedback",
            "payload": {"message": "Use an exposed tool."},
        },
    ]
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)

    assert report["status"] == "ok"
    assert report["coverage"]["provider_rejected_turns"] == 1
    assert report["coverage"]["assistant_turns"] == 2


def test_audit_excludes_closed_runtime_rejection_from_accepted_windows(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["agent_transcript"][3:3] = [
        {
            "id": "assistant_rejected",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "Submit an invalid plan update.",
                "public_intent_source": "provider_content",
                "tool_calls": [
                    {
                        "tool_call_id": "plan_rejected",
                        "tool_name": "runtime.update_plan",
                        "args": {
                            "completion_assessment": {
                                "done_when": [
                                    {
                                        "index": 0,
                                        "status": "unknown",
                                        "evidence": "not complete",
                                    }
                                ]
                            }
                        },
                    }
                ],
            },
        },
        {
            "id": "result_rejected",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "plan_rejected",
                "tool_name": "runtime.update_plan",
                "status": "rejected",
                "message": "plan update requires valid steps",
                "data": {"rejection": {"scope": "turn", "code": "invalid_turn_combination", "role": "turn_member", "executed": False}},
            },
        },
        {
            "id": "feedback_rejected",
            "kind": "protocol_feedback",
            "payload": {"message": "Submit a valid plan or act directly."},
        },
    ]
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)

    assert report["status"] == "ok"
    assert report["coverage"]["rejected_assistant_turns"] == 1
    assert report["coverage"]["control_turns"] == 1
    assert report["coverage"]["physical_batches"] == 1


def test_audit_rejects_invalid_intent_source_and_missing_rejection_feedback(tmp_path):
    root = _valid_eval(tmp_path)
    audit_path = root / "cases" / "case-1" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    transcript = audit["agent_transcript"]
    transcript[3]["payload"]["public_intent_source"] = "invented_reasoning"
    transcript.insert(
        3,
        {
            "id": "provider_rejected",
            "kind": "provider_turn_rejected",
            "payload": {
                "error": "invalid arguments",
                "finish_reason": "tool_calls",
                "content_present": False,
                "content": None,
                "tool_calls": [],
            },
        },
    )
    _write_json(audit_path, audit)

    report = MODULE.audit_evaluation(root)
    codes = {item["code"] for item in report["violations"]}

    assert "public_intent_source_invalid" in codes
    assert "provider_rejection_feedback_missing" in codes
