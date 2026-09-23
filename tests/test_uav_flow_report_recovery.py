"""Regression tests for interrupted UAV-Flow report persistence and recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.benchmarks.core import (
    BenchmarkAuditLog,
    BenchmarkCaseReview,
    BenchmarkReviewRenderer,
)
from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.postprocess import (
    recover_existing_evaluation,
)
from evaluation.benchmarks.uav_flow.runner import UavFlowBenchmarkRunner


def test_recover_existing_evaluation_writes_partial_html_idempotently(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "partial"
    complete = eval_dir / "cases" / "case_complete"
    incomplete = eval_dir / "cases" / "case_incomplete"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    image = complete / "review_assets" / "frames" / "rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-a-decoded-image-but-a-stable-report-asset")
    _write_json(complete / "case.json", _case_payload("case_complete"))
    audit_payload = _audit_payload("case_complete", display_path=image)
    audit_payload["agent_transcript"] = [
        {
            "id": "turn_0001",
            "timestamp": "2026-08-17T00:00:00+00:00",
            "instruction_id": "inst_complete",
            "kind": "assistant_turn",
            "payload": {
                "public_intent": "Recover the saved transcript.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_recovered",
                        "tool_name": "runtime.commit_phase_turn",
                        "args": {"phase": "turn"},
                    }
                ],
            },
        },
        {
            "id": "turn_0002",
            "timestamp": "2026-08-17T00:00:01+00:00",
            "instruction_id": "inst_complete",
            "kind": "tool_result",
            "payload": {
                "tool_call_id": "call_recovered",
                "tool_name": "runtime.commit_phase_turn",
                "status": "ok",
                "message": "recovered",
            },
        },
    ]
    _write_json(complete / "audit.json", audit_payload)

    first = recover_existing_evaluation(
        eval_dir,
        run_status="interrupted",
        expected_case_count=2,
        strict=True,
    )
    second = recover_existing_evaluation(
        eval_dir,
        run_status="interrupted",
        expected_case_count=2,
        strict=True,
    )

    assert first["finalized_case_count"] == 1
    assert first["incomplete_case_ids"] == ["case_incomplete"]
    assert second["generated_case_html_count"] == 1
    assert (complete / "review.html").is_file()
    assert not (incomplete / "review.html").exists()
    root_html = (eval_dir / "review.html").read_text(encoding="utf-8")
    case_html = (complete / "review.html").read_text(encoding="utf-8")
    assert "Evaluation State: INTERRUPTED" in root_html
    assert "case_complete" in root_html
    assert "Recover the saved transcript." not in case_html  # raw transcript is opt-in
    assert "audit.json" in case_html
    assert "Recover the saved transcript." in (complete / "audit.json").read_text(encoding="utf-8")
    assert "call_recovered" in case_html
    summary = json.loads((eval_dir / "summary.json").read_text("utf-8"))
    assert summary["run_status"] == "interrupted"
    assert summary["expected_case_count"] == 2
    assert summary["finalized_case_count"] == 1
    assert summary["metrics"]["case_count"] == 1


def test_recovery_accepts_legacy_audit_without_agent_transcript(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "legacy" / "cases" / "legacy_case"
    case_dir.mkdir(parents=True)
    image = case_dir / "review_assets" / "frames" / "rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"legacy-report-asset")
    _write_json(case_dir / "case.json", _case_payload("legacy_case"))
    payload = _audit_payload("legacy_case", display_path=image)
    assert "agent_transcript" not in payload
    _write_json(case_dir / "audit.json", payload)

    result = recover_existing_evaluation(
        tmp_path / "legacy",
        run_status="completed",
        expected_case_count=1,
        strict=True,
    )

    assert result["finalized_case_count"] == 1
    html = (case_dir / "review.html").read_text(encoding="utf-8")
    assert "Initial plan snapshot unavailable" in html and "audit.json" in html


def test_strict_recovery_rejects_missing_declared_asset(tmp_path: Path) -> None:
    case_dir = tmp_path / "eval" / "cases" / "missing_asset"
    case_dir.mkdir(parents=True)
    _write_json(case_dir / "case.json", _case_payload("missing_asset"))
    _write_json(
        case_dir / "audit.json",
        _audit_payload(
            "missing_asset",
            display_path=case_dir / "does_not_exist.png",
        ),
    )

    with pytest.raises(ValueError, match="declared_frame_asset_missing"):
        recover_existing_evaluation(
            tmp_path / "eval",
            expected_case_count=1,
            strict=True,
        )

    assert not (tmp_path / "eval" / "summary.json").exists()


def test_runner_flushes_completed_case_before_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = UavFlowEvalConfig.from_dict(
        {**{"execution_mode": "managed", "save_artifacts": True, "enable_action_planner_debug": True},
            "output_dir": str(tmp_path / "outputs"),
            "run_root": str(tmp_path / "runs"),
            "execution_mode": "managed",
            "generate_html_review": True,
            "enable_sr_judge": False,
            "enable_visual_diagnostics": False,
        }
    )
    runner = UavFlowBenchmarkRunner(config)
    cases = [_case("first"), _case("second")]
    monkeypatch.setattr(runner, "load_cases", lambda: cases)

    def fake_run_case(case: UavFlowCase, eval_dir: str | Path):
        case_dir = Path(eval_dir) / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        if case.case_id == "second":
            raise KeyboardInterrupt()
        review = _case_review(case.case_id)
        runner._case_reviews[case.case_id] = review
        BenchmarkReviewRenderer().write_case(eval_dir, review)
        return {
            "case_id": case.case_id,
            "class_name": case.class_name,
            "status": "ok",
            "completed": True,
            "stopped": False,
            "stopped_reason": None,
            "error": None,
            "ndtw": {"status": "ok", "ndtw": 1.0},
            "turn_diagnostics": {},
            "backend_action_tracking": {},
            "sr_judge": {"status": "skipped"},
        }

    monkeypatch.setattr(runner, "run_case", fake_run_case)

    with pytest.raises(KeyboardInterrupt):
        runner.run(eval_id="interrupted")

    eval_dir = tmp_path / "outputs" / "interrupted"
    assert (eval_dir / "cases" / "first" / "review.html").is_file()
    assert not (eval_dir / "cases" / "second" / "review.html").exists()
    summary = json.loads((eval_dir / "summary.json").read_text("utf-8"))
    assert summary["run_status"] == "interrupted"
    assert summary["finalized_case_count"] == 1
    assert summary["incomplete_case_ids"] == ["second"]


def _case(case_id: str) -> UavFlowCase:
    return UavFlowCase(
        case_id=case_id,
        file_name=f"{case_id}.json",
        source_path=f"{case_id}.json",
        class_name="Turn",
        instruction="Turn toward the target.",
        instruction_unified="Turn toward the target.",
        selected_instruction="Turn toward the target.",
        initial_pos=[0, 0, 100, 0, 0, 0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0]],
    )


def _case_payload(case_id: str) -> dict:
    return {
        "case": {
            "case_id": case_id,
            "class_name": "Turn",
            "instruction": "Turn toward the target.",
            "instruction_unified": "Turn toward the target.",
            "selected_instruction": "Turn toward the target.",
        },
        "planner_safe_case": {"class_name": "Turn"},
        "run_dir": "unused/offline/run",
        "status": "ok",
        "completed": True,
        "stopped": False,
        "stopped_reason": None,
        "stop_source": None,
        "error": None,
        "cleanup_errors": [],
        "audit_summary": {"rgb_count": 1, "action_count": 0},
        "visual_diagnostics": {"status": "skipped"},
        "plot_result": {"status": "skipped"},
        "ndtw": {"status": "ok", "ndtw": 1.0},
        "turn_diagnostics": {},
        "backend_action_tracking": {"zero_motion_completion": False},
        "scenario_diagnostics": {},
        "sr_judge": {"status": "skipped"},
        "stmr": {"mode": "off", "status": "disabled"},
        "metric_error": None,
    }


def _audit_payload(case_id: str, *, display_path: Path) -> dict:
    return {
        "run_dir": "unused/offline/run",
        "started_at": "2026-08-17T00:00:00+00:00",
        "ended_at": "2026-08-17T00:00:01+00:00",
        "task_initial_observation_ref": None,
        "observation_anchors": {},
        "observations": [],
        "frames": [],
        "poses": [],
        "actions": [],
        "agent_messages": [],
        # Historical audits omit agent_transcript. Offline recovery must keep
        # accepting that additive-schema predecessor.
        "events": [],
        "debug_events": [],
        "planner_debug": [],
        "diagnostics": [],
        "frame_assets": [
            {
                "frame_id": f"{case_id}_rgb",
                "observation_id": "obs_1",
                "modality": "rgb",
                "timestamp": "2026-08-17T00:00:00+00:00",
                "display_path": str(display_path),
                "raw_path": str(display_path),
                "source": "artifact_path",
                "unavailable_reason": None,
            }
        ],
    }


def _case_review(case_id: str) -> BenchmarkCaseReview:
    return BenchmarkCaseReview(
        case_id=case_id,
        benchmark_name="UAV-Flow",
        protocol_variant="uav-flow-v1",
        instruction="Turn toward the target.",
        outcome={"status": "ok", "completed": True},
        metrics={"ndtw": 1.0},
        audit=BenchmarkAuditLog(
            run_dir="unused",
            started_at=None,
            ended_at="2026-08-17T00:00:01+00:00",
            task_initial_observation_ref=None,
            observations=(),
            frames=(),
            poses=(),
            actions=(),
            agent_messages=(),
            events=(),
            debug_events=(),
            planner_debug=(),
        ),
        frame_assets=(),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
