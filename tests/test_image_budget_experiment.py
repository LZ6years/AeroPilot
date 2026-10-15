"""Fixed-cohort identity, accounting, threshold audit and completion boundary."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from evaluation.benchmarks.aerial_vln.campaign import _attempt_dir
from evaluation.benchmarks.aerial_vln.parallel_campaign import campaign_tasks
from evaluation.benchmarks.aerial_vln.parallel_watchdog import resume_policy
from evaluation.benchmarks.aerial_vln.recovery import RecoveryQueue
from evaluation.experiments.compression_cost import effective_planner_calls, effective_totals
from evaluation.experiments.image_budget import quotas, select_samples
from evaluation.experiments.image_budget_report import aggregate_group


def test_paired_case_keys_and_directories_do_not_collide(tmp_path):
    a = {"dataset": "aerial_vln", "split": "val_seen", "episode_id": "same", "image_budget": 10}
    b = {**a, "image_budget": 20}
    assert RecoveryQueue.key(a) != RecoveryQueue.key(b)
    assert _attempt_dir(tmp_path, a, 1) != _attempt_dir(tmp_path, b, 1)
    assert campaign_tasks({"experiment": "image_budget", "tasks": [a, {**b, "split": "test_seen"}]}) == [a, {**b, "split": "test_seen"}]
    assert RecoveryQueue.key({"split": "val_seen", "episode_id": "same"}) == "val_seen/same"


def test_cohort_exact_quota_and_anchors_are_reproducible():
    population, successes = [], {}
    for split in ("val_seen", "val_unseen"):
        for difficulty in ("high", "medium"):
            for i in range(60):
                population.append({"dataset": "aerial_vln", "split": split, "difficulty": difficulty,
                                   "scene_id": str(i % 4), "episode_id": f"{split}_{difficulty}_{i:03}"})
    for i in range(8):
        successes[("aerial_vln", "val_seen", f"val_seen_high_{i:03}")] = [{}]
    for split, n in (("test_seen", 1418), ("test_unseen_map", 958), ("test_unseen_object", 629)):
        for i in range(n):
            population.append({"dataset": "travel_uav", "split": split, "difficulty": "hard" if i % 2 else "easy",
                               "scene_id": str(i % 3), "episode_id": f"{split}_{i:04}"})
    for i in range(6):
        successes[("travel_uav", "test_seen", f"test_seen_{i:04}")] = [{}]
    first = select_samples(population, successes)
    assert first == select_samples(population, successes)
    assert len(first) == 200
    assert sum(t["historical_success_anchor"] for t in first) == 14
    for split in ("val_seen", "val_unseen"):
        assert sum(t["split"] == split and t["difficulty"] == "high" for t in first) == 35
        assert sum(t["split"] == split and t["difficulty"] == "medium" for t in first) == 15
    assert quotas({"seen": 1418, "map": 958, "object": 629}, 100) == {"seen": 47, "map": 32, "object": 21}


def test_final_successful_attempt_latency_excludes_failure_and_backoff(tmp_path):
    path = tmp_path / "debug/tool_loop/instruction/decision_0001.json"
    path.parent.mkdir(parents=True)
    def exchange(attempts, elapsed, *, error=False):
        return {"response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
                "canonical_turn": {} if not error else None, "error_type": "ProtocolError" if error else None,
                "transport": {"final_status": "success", "elapsed_seconds": elapsed,
                              "attempts": attempts, "attempt_count": len(attempts)},
                "history_projection": {"estimated_input_tokens": 25000}}
    path.write_text(json.dumps({"model_rounds": [
        exchange([{"status": "failed", "elapsed_seconds": 3, "backoff_seconds": 1}, {"status": "success", "elapsed_seconds": 5}], 9),
        exchange([{"status": "success", "elapsed_seconds": 7}], 7),
        exchange([{"status": "success", "elapsed_seconds": 100}], 100, error=True),
    ]}), encoding="utf-8")
    values = effective_totals(effective_planner_calls(tmp_path))
    assert values["calls"] == 2
    assert values["decision_latency_seconds"] == 6
    assert values["input_tokens"] == 200
    assert values["failed_attempt_seconds"] == 3
    assert values["backoff_seconds"] == 1
    assert values["transport_attempts"] == 4


def test_missing_attempt_timing_is_not_replaced_by_logical_elapsed(tmp_path):
    path = tmp_path / "debug/tool_loop/instruction/decision_0001.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model_rounds": [{"response": {"usage": {}}, "canonical_turn": {},
        "transport": {"final_status": "success", "elapsed_seconds": 123}}]}), encoding="utf-8")
    values = effective_totals(effective_planner_calls(tmp_path))
    assert values["calls"] == 1
    assert values["decision_latency_seconds"] is None
    assert values["input_tokens"] is None


def test_failed_legacy_integer_attempts_are_audited_without_crashing_or_fabricating_time(tmp_path):
    path = tmp_path / "debug/tool_loop/instruction/decision_0001.json"
    path.parent.mkdir(parents=True)
    payload = {"model_rounds": [{"error_type": "ProviderRequestError", "response": None,
                "transport": {"final_status": "failed", "attempts": 2, "attempt_count": 2,
                              "elapsed_seconds": 182.516}}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    values = effective_totals(effective_planner_calls(tmp_path))
    assert values["calls"] == 0 and values["excluded_responses"] == 1
    assert values["transport_attempts"] == values["failed_attempts"] == 2
    assert values["missing_attempt_detail_count"] == 2
    assert values["failed_attempt_seconds"] is None and values["backoff_seconds"] is None
    assert path.read_bytes() == before


def test_incomplete_cohort_never_fills_paper_table_and_completion_stops_watchdog():
    result = aggregate_group([], [], 10)
    assert not result["complete"] and all(v is None for v in result["paper"].values())
    assert resume_policy({"status": "awaiting_image_budget_choice", "tasks": {"case": {}}}) == "awaiting_image_budget_choice"


def test_navigation_failures_remain_in_all_fixed_cohort_metrics():
    rows = []
    for dataset in ("aerial_vln", "travel_uav"):
        for index in range(100):
            rows.append({"image_budget": 10, "status": "scored", "dataset": dataset,
                         "success": int(index < 50), "ndtw": 0.5 if index < 50 else 0.25,
                         "ne": 10 if index < 50 else 30, "peak_estimate_coverage_complete": True,
                         "peak_estimated_input_tokens": 23000 if index < 50 else 45000})
    result = aggregate_group(rows, [], 10)
    assert result["complete"]
    assert result["paper"]["aerial_sr"] == result["paper"]["travel_sr"] == 50
    assert result["paper"]["aerial_ndtw"] == 37.5
    assert result["paper"]["travel_ne"] == 20
    assert result["future_additional_executions_upper_bound"] == 200


def test_endpoint_proximity_uses_dataset_distance_and_inclusive_boundary():
    from evaluation.experiments.image_budget_report import final_proximity
    for dataset, field in (("aerial_vln", "distance_to_goal"), ("travel_uav", "navigation_error")):
        assert final_proximity(dataset, {field: 25, "success": False}) == {"final_distance_m": 25, "within_25m": True}
        assert final_proximity(dataset, {field: 25.0001})["within_25m"] is False
        for missing in (None, float("nan"), float("inf"), -1, True):
            assert final_proximity(dataset, {field: missing})["within_25m"] is None
        assert final_proximity(dataset, {})["within_25m"] is None


def test_ndtw_missing_is_not_zero_and_failed_navigation_keeps_path_quality():
    row = {"dataset": "aerial_vln", "image_budget": 10, "status": "scored", "success": 0,
           "ndtw": 0.5, "peak_estimate_coverage_complete": False}
    result = aggregate_group([row], [], 10)
    assert result["provisional"]["aerial_sr"] == 0
    assert result["provisional"]["aerial_ndtw"] == 50
    assert aggregate_group([row, {**row, "ndtw": None}], [], 10)["provisional"]["aerial_ndtw"] is None


def test_report_marks_only_retained_valid_endpoints(tmp_path, monkeypatch):
    from evaluation.experiments import image_budget_report as reporting
    monkeypatch.setattr(reporting, "analyze_attempt", lambda *args: {
        "requests": [], "totals": effective_totals([]), "summary_audit_files": 0})
    (tmp_path / "manifest.json").write_text(json.dumps({"image_budgets": [10], "counts": {}, "fingerprint": "test"}), encoding="utf-8")
    tasks = {}
    for dataset in ("aerial_vln", "travel_uav"):
        for valid in (True, False):
            key = dataset + str(valid)
            result = {"metrics": {"success": False, "ndtw": 0.5, "distance_to_goal": 25, "navigation_error": 24}}
            tasks[key] = {"task": {"dataset": dataset, "image_budget": 10, "split": "test", "episode_id": key,
                                   "scene_id": "1", "difficulty": "high", "historical_success_anchor": False},
                          "status": "scored" if valid else "retry_pending", "result": result,
                          "attempts": [{"number": 1, "status": "scored" if valid else "provider_failed",
                                        "valid_score": valid, "cleanup_verified": True, "result": result}]}
    report = reporting.write_report(tmp_path, {"status": "running", "tasks": tasks})
    assert report["valid_executions"] == 2
    assert all(p["within_25m_count"] == 1 for p in report["phases"])
    import csv
    with (tmp_path / "per_case.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["within_25m"] for r in rows] == ["True", "", "True", ""]
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'nDTW ↑ (%)' in page and 'id="near25"' in page and 'SDTW' not in page
    assert page.count('data-near25="1" data-dataset=') == 2


def test_disabled_compression_measures_without_scheduling_or_changing_messages(tmp_path):
    from tests.test_history_compression import _transcript, StubCompactor
    from src.agent.context.compression import HistoryCompressionConfig, HistoryProjectionState
    from src.agent.context.projection import HistoryProjector
    from src.agent.context.window_manager import ContextWindowManager
    from src.tools.registry import default_registry
    transcript = _transcript(tmp_path, 8)
    projector, compactor = HistoryProjector(), StubCompactor()
    state = HistoryProjectionState()
    manager = ContextWindowManager(projector, replace(HistoryCompressionConfig(), enabled=False), compactor=compactor)
    result = manager.project(transcript, default_registry(), state=state,
                             fixed_messages=[], tools=[], parameters={})
    assert result.messages == projector.project(transcript, default_registry()).messages
    assert result.audit["estimated_input_tokens"] == state.last_estimated_tokens > 0
    assert result.audit["compaction_status"] == "disabled" and not compactor.requests
    initial = state.calibration_ratio
    manager.observe_usage(state, {"prompt_tokens": 2 * state.last_estimated_tokens})
    assert state.calibration_ratio > initial
