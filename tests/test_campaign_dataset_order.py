"""Regression checks for dataset-ordered continuation and audited exclusions."""
from copy import deepcopy
import json
import pytest

from evaluation.benchmarks.core.campaign import parallel, recovery
from evaluation.experiments.image_budget_report import aggregate_group


def test_operator_retry_preserves_history_and_grants_only_one_attempt(tmp_path):
    tasks = [{"dataset": "aerial_vln", "split": "seen", "episode_id": str(i)} for i in range(2)]
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    q.state.update(status="paused", reason="episode_attempts_exhausted")
    for item in q.state["tasks"].values():
        item.update(status="retry_exhausted", attempts=[{"number": n, "status": "provider_failed",
                    "valid_score": False, "cleanup_verified": True, "result": {"error": "timeout"}} for n in range(1, 4)])
    before = deepcopy(q.state["tasks"][q.key(tasks[0])]["attempts"])
    q.authorize_extra_attempt(tasks[0], reason="User requested continuation", authorized_at="2026-10-14")
    assert q.state["tasks"][q.key(tasks[0])]["attempts"] == before
    with pytest.raises(ValueError):
        q.authorize_extra_attempt(tasks[0], reason="duplicate", authorized_at="2026-10-14")
    q = recovery.ParallelRecoveryQueue(q.path, tasks, "frozen")
    task, attempt = q.claim(tasks, "A")
    assert task == tasks[0] and attempt["number"] == 4
    assert q.claim(tasks, "B") is None
    q.accept(task, 4, {"cleanup_verified": True, "valid_score": False,
                      "result": {"error": {"error_type": "ProviderRequestError", "retryable": True}}})
    assert q.state["tasks"][q.key(task)]["status"] == "retry_exhausted"
    assert q.claim(tasks, "A") is None


def test_operator_retry_rejects_unverified_cleanup(tmp_path):
    task = {"split": "seen", "episode_id": "one"}
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", [task], "frozen")
    q.state.update(status="paused")
    q.state["tasks"][q.key(task)].update(status="retry_exhausted", attempts=[{"cleanup_verified": False}])
    with pytest.raises(ValueError):
        q.authorize_extra_attempt(task, reason="continue", authorized_at="2026-10-14")


@pytest.mark.parametrize("mode", ["budget_dataset_sequential", "dataset_budget_sequential"])
def test_all_six_stages_keep_both_lanes_inside_current_dataset(tmp_path, mode):
    tasks = [{"image_budget": budget, "dataset": dataset, "split": "split", "episode_id": str(i)}
             for budget in (10, 20, 30) for dataset in ("aerial_vln", "travel_uav") for i in range(2)]
    manifest = {"experiment": "image_budget", "image_budgets": [10, 20, 30], "tasks": tasks}
    policy = {"dataset_order": ["aerial_vln", "travel_uav"], "mode": mode}
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    groups = [(b, d) for b in (10, 20, 30) for d in policy["dataset_order"]]
    if mode == "dataset_budget_sequential":
        groups.sort(key=lambda g: (policy["dataset_order"].index(g[1]), g[0]))
    for budget, dataset in groups:
            stage, selected = parallel.configuration_stage(manifest, tasks, q, policy)
            assert stage == (budget, dataset)
            trial = [t for t in tasks if t in selected]
            candidates = parallel.stage_candidates(selected, trial, False, q)
            for lane in ("A", "B"):
                task, _ = q.claim(parallel.dispatch_candidates(manifest, candidates, lane, {}, 2, q), lane)
                assert (task["image_budget"], task["dataset"]) == stage
            first, second = [q.state["tasks"][q.key(t)] for t in selected]
            first["status"], second["status"] = "scored", "retry_pending"
            assert parallel.configuration_stage(manifest, tasks, q, policy)[0] == stage
            second["status"] = "retry_exhausted"
            assert parallel.configuration_stage(manifest, tasks, q, policy)[0] == stage
            second["status"] = "scored"
    q.save()
    restored = recovery.ParallelRecoveryQueue(q.path, tasks, "frozen")
    assert parallel.configuration_stage(manifest, tasks, restored, policy)[0] == (30, "travel_uav")
    assert parallel.technical_check_path(tmp_path, (10, "aerial_vln")) != parallel.technical_check_path(tmp_path, (10, "travel_uav"))


def test_reconcile_only_technical_errors_and_preserve_raw_receipts(tmp_path):
    tasks = [{"dataset": "travel_uav", "split": "seen", "episode_id": str(i)} for i in range(2)]
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    for i, task in enumerate(tasks):
        result = {"metrics": {"success": False}, "error": {"error_type": "ProviderRequestError", "retryable": True} if i == 0 else None}
        item = q.state["tasks"][q.key(task)]
        item.update(status="scored", result=result, attempts=[{"number": 1, "status": "scored", "valid_score": True,
                                                            "cleanup_verified": True, "result": result}])
    raw = deepcopy(q.state["tasks"][q.key(tasks[0])]["result"])
    assert len(q.reconcile_scored_errors()) == 1
    bad, normal = [q.state["tasks"][q.key(t)] for t in tasks]
    assert bad["status"] == "retry_pending" and bad["result"] is None
    assert bad["attempts"][0]["result"] == raw
    assert bad["attempts"][0]["status"] == "provider_failed"
    assert normal["status"] == "scored" and normal["result"]["metrics"]["success"] is False
    assert q.reconcile_scored_errors() == []


def test_complete_aerial_metrics_close_before_pooled_costs():
    rows = [{"dataset": dataset, "image_budget": 10, "status": "scored" if dataset == "aerial_vln" else "pending",
             "success": 0, "ndtw": 0, "ne": None, "peak_estimated_input_tokens": None,
             "peak_estimate_coverage_complete": False} for dataset in ("aerial_vln", "travel_uav") for _ in range(100)]
    group = aggregate_group(rows, [], 10)
    assert group["paper"]["aerial_sr"] == group["paper"]["aerial_ndtw"] == 0
    assert group["paper"]["travel_sr"] is None
    assert group["paper"]["input_k"] is None
    assert not group["complete"]


def test_existing_later_results_are_kept_while_current_stage_retries(tmp_path):
    tasks = [{"dataset": dataset, "image_budget": budget, "split": "seen", "episode_id": "one"}
             for budget in (10, 20, 30) for dataset in ("aerial_vln", "travel_uav")]
    manifest = {"experiment": "image_budget", "image_budgets": [10, 20, 30], "tasks": tasks}
    policy = {"dataset_order": ["aerial_vln", "travel_uav"]}
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    q.state["tasks"][q.key(tasks[0])]["status"] = "retry_pending"
    q.state["tasks"][q.key(tasks[1])]["status"] = "scored"
    q.state["tasks"][q.key(tasks[2])]["status"] = "scored"
    q.save()
    resumed = recovery.ParallelRecoveryQueue(q.path, tasks, "frozen")
    assert parallel.configuration_stage(manifest, tasks, resumed, policy)[0] == (10, "aerial_vln")
    resumed.state["tasks"][resumed.key(tasks[0])]["status"] = "scored"
    assert parallel.configuration_stage(manifest, tasks, resumed, policy)[0] == (20, "travel_uav")




def test_storage_gates_current_stage_and_preserves_full_forecast(tmp_path, monkeypatch):
    from types import SimpleNamespace
    tasks = [{"dataset": dataset, "image_budget": b, "split": "seen", "episode_id": str(i)}
             for b in (10, 20, 30) for dataset in ("aerial_vln", "travel_uav") for i in range(100)]
    manifest = {"experiment": "image_budget", "image_budgets": [10, 20, 30], "tasks": tasks}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "parallel").mkdir()
    (tmp_path / "parallel/execution_overlay.json").write_text(json.dumps({"logging_profile": "paper_process_metrics_v1",
        "scheduling_policy": {"dataset_order": ["aerial_vln", "travel_uav"]}}), encoding="utf-8")
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    q.state["tasks"][q.key(tasks[0])]["attempts"] = [{"logging_profile": "paper_process_metrics_v1", "retained_bytes": 53_000_000}]
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=52_000_000_000))
    assert parallel.storage_budget(tmp_path, q, tasks, 2)
    check = json.loads((tmp_path / "storage_check.json").read_text(encoding="utf-8"))
    assert check["remaining_cases"] == 100 and check["total_remaining_cases"] == 600
    assert check["required_bytes"] >= 10 * 1024**3
    assert check["full_campaign_required_bytes"] > check["free_bytes"]
    monkeypatch.setattr(parallel.shutil, "disk_usage", lambda _: SimpleNamespace(free=9 * 1024**3))
    assert not parallel.storage_budget(tmp_path, q, tasks, 2)
