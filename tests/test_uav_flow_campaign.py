"""Regression coverage for reproducible multi-segment UAV-Flow campaigns."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.benchmarks.uav_flow.campaign import (
    RUN_MANIFEST_SCHEMA,
    audit_campaign,
    dataset_sha256,
    write_campaign_outputs,
    write_run_manifest,
)
from evaluation.benchmarks.uav_flow.cases import load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    tasks = root / "test_jsons"
    tasks.mkdir(parents=True)
    classification = {
        "Turn": ["case_a.json", "case_b.json"],
        "Move": ["case_c.json"],
    }
    (root / "classified_instr.json").write_text(
        json.dumps(classification), encoding="utf-8"
    )
    for index, file_name in enumerate(
        ["case_a.json", "case_b.json", "case_c.json"]
    ):
        (tasks / file_name).write_text(
            json.dumps(
                {
                    "instruction": f"instruction {index}",
                    "instruction_unified": f"instruction {index}",
                    "initial_pos": [0, 0, 100, 0, 0, 0],
                    "reference_path_preprocessed": [[0, 0, 0, 0, 0, 0]],
                }
            ),
            encoding="utf-8",
        )
    return root


def _config(tmp_path: Path, dataset: Path) -> tuple[Path, UavFlowEvalConfig]:
    path = tmp_path / "full.local"
    data = {
        "uav_flow_repo_path": str(dataset),
        "output_dir": str(tmp_path / "outputs"),
        "run_root": str(tmp_path / "runs"),
        "map_config_path": "map.json",
        "map_id": "map",
        "execution_mode": "managed",
        "save_artifacts": True,
        "enable_action_planner_debug": True,
        "enable_sr_judge": False,
        "enable_visual_diagnostics": False,
        "parser_mode": "model",
        "shutdown_policy": "hover",
        "planner_depth_input": "now",
        "runtime_loop_guard_windows": 80,
        "stmr_mode": "off",
        "limit_per_class": None,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, UavFlowEvalConfig.from_dict(data)


def _manifest(config: UavFlowEvalConfig, selected: list[str]) -> dict:
    resolved = config.to_dict()
    resolved["runtime_loop_guard_windows"] = 50
    resolved["case_ids"] = selected
    resolved["classes"] = None
    resolved["limit_per_class"] = None
    resolved["max_cases"] = None
    cases = load_cases(config.uav_flow_repo_path, case_ids=selected)
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "git_commit": "abc123",
        "git_dirty": False,
        "resolved_config": resolved,
        "model": {
            "provider": "test",
            "name": "model",
            "max_tool_rounds": 10,
            "max_consecutive_protocol_rejections": 3,
        },
        "selected_case_ids": selected,
        "class_counts": {},
        "dataset_sha256": dataset_sha256(config),
        "_cases": cases,
    }


def _segment(
    tmp_path: Path,
    config: UavFlowEvalConfig,
    name: str,
    selected: list[str],
    finalized: list[str],
    *,
    run_status: str = "completed",
    error_case: str | None = None,
) -> Path:
    root = tmp_path / name
    root.mkdir()
    manifest = _manifest(config, selected)
    cases = manifest.pop("_cases")
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps({"run_status": run_status}), encoding="utf-8"
    )
    (root / "scene_session.json").write_text(
        json.dumps(
            {
                "start_count": 1,
                "restart_count": 0,
                "stop_count": 0 if run_status == "running" else 1,
            }
        ),
        encoding="utf-8",
    )
    by_id = {case.case_id: case for case in cases}
    for case_id in finalized:
        case_dir = root / "cases" / case_id
        case_dir.mkdir(parents=True)
        payload = {
            "case": by_id[case_id].to_dict(),
            "status": "error" if case_id == error_case else "ok",
            "completed": case_id != error_case,
            "stopped_reason": None,
            "error": {"error": "provider failed"} if case_id == error_case else None,
            "trace": {
                "action_count": 1,
                "physical_call_count": 1,
                "physical_window_count": 1,
                "motion_call_count": 1,
                "trace_schema_version": "physical_window_v2",
                "trace_integrity": "ok",
            },
            "trace_valid": True,
            "tracking_valid": True,
            "infrastructure_clean": True,
            "ndtw": {"status": "ok", "ndtw": 0.5},
            "sr_judge": {"status": "skipped"},
            "backend_action_tracking": {
                "zero_motion_completion": False,
                "trace_valid": True,
                "tracking_valid": True,
                "all_observed_within_tolerance": True,
            },
        }
        (case_dir / "case.json").write_text(json.dumps(payload), encoding="utf-8")
        (case_dir / "audit.json").write_text("{}", encoding="utf-8")
        (case_dir / "review.html").write_text("ok", encoding="utf-8")
    return root


def _native_stub(root: Path) -> dict:
    case_dirs = list((root / "cases").glob("*")) if (root / "cases").is_dir() else []
    return {
        "cases": [
            {
                "case_id": case_dir.name,
                "coverage": {
                    "assistant_turns": 2,
                    "committed_turns": 1,
                    "physical_batches": 1,
                },
            }
            for case_dir in case_dirs
            if (case_dir / "case.json").is_file() and (case_dir / "audit.json").is_file()
        ],
        "violations": [],
    }


def test_run_manifest_is_redacted_and_hashes_full_dataset(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    _, config = _config(tmp_path, dataset)
    cases = load_cases(dataset)
    monkeypatch.setattr(
        "evaluation.benchmarks.uav_flow.campaign._git_state",
        lambda root: ("deadbeef", False),
    )
    model = SimpleNamespace(
        provider="provider",
        model_name="model",
        max_tool_rounds=10,
        max_consecutive_protocol_rejections=3,
        api_key="secret",
    )

    path = write_run_manifest(
        tmp_path / "eval", config=config, model_config=model, cases=cases
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    assert payload["git_commit"] == "deadbeef"
    assert payload["selected_case_ids"] == ["case_a", "case_b", "case_c"]
    assert payload["class_counts"] == {"Move": 1, "Turn": 2}
    assert payload["dataset_sha256"] == dataset_sha256(config)
    assert payload["model"]["max_tool_rounds"] == 10
    assert payload["model"]["max_consecutive_protocol_rejections"] == 3
    assert "secret" not in Path(path).read_text(encoding="utf-8")


def test_campaign_combines_segments_in_dataset_order(tmp_path):
    dataset = _dataset(tmp_path)
    config_path, config = _config(tmp_path, dataset)
    main = _segment(tmp_path, config, "main", ["case_a", "case_b", "case_c"], ["case_a"], run_status="interrupted")
    continuation = _segment(tmp_path, config, "cont01", ["case_b", "case_c"], ["case_b", "case_c"])

    report = audit_campaign(
        config_path,
        [main, continuation],
        expected_loop_guard_windows=50,
        segment_auditor=_native_stub,
    )

    assert report["status"] == "ok"
    assert report["coverage_complete"] is True
    assert report["finalized_case_count"] == 3
    assert [item["case_id"] for item in report["cases"]] == ["case_a", "case_b", "case_c"]
    assert report["transcript_coverage"]["assistant_turns"] == 6


def test_finalized_error_is_not_returned_as_missing(tmp_path):
    dataset = _dataset(tmp_path)
    config_path, config = _config(tmp_path, dataset)
    main = _segment(
        tmp_path,
        config,
        "main",
        ["case_a", "case_b", "case_c"],
        ["case_a", "case_b"],
        run_status="interrupted",
        error_case="case_b",
    )

    report = audit_campaign(
        config_path,
        [main],
        expected_loop_guard_windows=50,
        segment_auditor=_native_stub,
    )

    assert report["missing_case_ids"] == ["case_c"]
    assert report["status_counts"] == {"error": 1, "ok": 1}


def test_campaign_separates_trace_tracking_and_infrastructure_gates(tmp_path):
    dataset = _dataset(tmp_path)
    config_path, config = _config(tmp_path, dataset)
    main = _segment(
        tmp_path,
        config,
        "main",
        ["case_a", "case_b", "case_c"],
        ["case_a", "case_b", "case_c"],
    )
    case_path = main / "cases" / "case_b" / "case.json"
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["tracking_valid"] = False
    payload["infrastructure_clean"] = False
    payload["backend_action_tracking"]["tracking_valid"] = False
    case_path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_campaign(
        config_path,
        [main],
        expected_loop_guard_windows=50,
        segment_auditor=_native_stub,
    )

    assert report["coverage_complete"] is True
    assert report["protocol_valid"] is True
    assert report["trace_valid"] is True
    assert report["tracking_valid"] is False
    assert report["infrastructure_clean"] is False
    assert report["status"] == "incomplete_or_invalid"


def test_campaign_rejects_duplicate_config_drift_and_dataset_drift(tmp_path):
    dataset = _dataset(tmp_path)
    config_path, config = _config(tmp_path, dataset)
    main = _segment(tmp_path, config, "main", ["case_a", "case_b"], ["case_a", "case_b"])
    continuation = _segment(tmp_path, config, "cont01", ["case_b", "case_c"], ["case_b", "case_c"])
    manifest_path = continuation / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_config"]["planner_depth_input"] = "off"
    manifest["dataset_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_campaign(
        config_path,
        [main, continuation],
        expected_loop_guard_windows=50,
        segment_auditor=_native_stub,
    )

    codes = {item["code"] for item in report["violations"]}
    assert {"duplicate_finalized_case", "resolved_config_mismatch", "dataset_hash_mismatch"}.issubset(codes)
    assert report["coverage_complete"] is False
    assert report["status"] == "incomplete_or_invalid"


def test_campaign_outputs_escape_html_and_write_ordered_remaining_ids(tmp_path):
    report = {
        "status": '<script>alert("x")</script>',
        "coverage_complete": False,
        "protocol_valid": True,
        "infrastructure_clean": True,
        "finalized_case_count": 1,
        "expected_case_count": 3,
        "missing_case_ids": ["case_b", "case_c"],
        "violations": [],
        "metrics": {"overall_mean_ndtw": 0.5, "by_class": {}},
        "segments": [
            {
                "segment": '<img src=x onerror=alert(1)>',
                "run_status": "completed",
                "finalized_case_count": 1,
                "selected_case_count": 1,
                "review_html": str(tmp_path / "review.html"),
            }
        ],
        "cases": [],
    }

    paths = write_campaign_outputs(report, tmp_path / "report")
    page = Path(paths["campaign_review"]).read_text(encoding="utf-8")

    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert Path(paths["remaining_case_ids_txt"]).read_text(encoding="utf-8") == "case_b,case_c\n"
