"""Shared routing, frozen CLI and coverage checks without flight.

Runner/resource doubles are explicit: these tests make no model requests and
are not GPU measurements. Cold snapshot checks use actual import/hash behavior.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.core.campaign import compat, parallel, prepare, recovery, watchdog
from evaluation.benchmarks.core.campaign.contracts import PreparedCampaignSplit, TaskIdentity, primary_config
from evaluation.benchmarks.core.campaign.reporting import write_report
from evaluation.benchmarks.core.campaign.support import freeze_source


def task(dataset, split, identity):
    return {"dataset": dataset, "split": split, "episode_id": identity, "scene_id": "scene", "difficulty": "hard"}


def test_legacy_imports_share_one_implementation():
    from evaluation.benchmarks.aerial_vln import parallel_campaign, parallel_recovery, parallel_watchdog
    assert parallel_campaign is parallel
    assert parallel_recovery is recovery
    assert parallel_watchdog is watchdog
    assert watchdog.SeenWatchdog is watchdog.CampaignWatchdog


def test_travel_scopes_all_declared_splits_and_completion_has_no_seen_dependency():
    tasks = [task("travel_uav", split, "same") for split in ("test_seen", "test_unseen_map", "test_unseen_object")]
    manifest = {"version": 2, "dataset": "travel_uav", "tasks": tasks,
                "configs": {"travel_uav/test_seen": {"env_path": "travel.env"}}, "primary_config_key": "travel_uav/test_seen"}
    assert parallel.campaign_tasks(manifest) == tasks
    assert compat.completion_status(manifest) == "campaign_complete"
    assert primary_config(manifest)["env_path"] == "travel.env"
    assert len({TaskIdentity.from_task(t).key for t in tasks}) == 3
    assert watchdog.resume_policy({"status": "campaign_complete", "tasks": {"case": {}}}) == "campaign_complete"
    assert watchdog.resume_policy({"status": "all_done", "tasks": {"case": {}}}, terminal_status="all_done") == "all_done"


def test_generic_configuration_groups_wait_for_valid_coverage(tmp_path):
    tasks = [{**task("travel_uav", "test_seen", "one"), "configuration_id": config} for config in ("original", "compressed")]
    manifest = {"version": 2, "dataset": "travel_uav", "scheduling": {"group_field": "configuration_id", "group_order": ["original", "compressed"]}}
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    assert parallel.configuration_stage(manifest, tasks, q) == ("original", tasks[:1])
    q.state["tasks"][q.key(tasks[0])]["status"] = "retry_exhausted"
    assert parallel.configuration_stage(manifest, tasks, q)[0] == "original"
    q.state["tasks"][q.key(tasks[0])]["status"] = "scored"
    assert parallel.configuration_stage(manifest, tasks, q) == ("compressed", tasks[1:])


def test_mixed_workers_keep_mutable_configs_and_run_directories_private(tmp_path, monkeypatch):
    tasks = [task("aerial_vln", "val_seen", "one"), task("travel_uav", "test_seen", "two")]
    for t in tasks:
        t["config_key"] = t["dataset"]
    configs = {t["dataset"]: {"nested": {"items": []}} for t in tasks}
    manifest = {"version": 2, "configs": configs}
    observed = []
    class Adapter:
        def execute(self, config, current, directory):
            config["nested"]["items"].append(current["episode_id"])
            observed.append((current["dataset"], directory, config))
            return {"episode_id": current["episode_id"], "metrics": {"success": False}}
    monkeypatch.setattr(compat, "adapter_for", lambda *_: Adapter())
    requests = [{"task": t, "attempt": {"number": 1}, "lane": lane, "map_config_path": lane + ".json"}
                for lane, t in zip(("A", "B"), tasks)]
    with ThreadPoolExecutor(2) as pool:
        values = list(pool.map(lambda r: parallel.execute_thread(tmp_path, manifest, r, {}), requests))
    assert {v["episode_id"] for v in values} == {"one", "two"}
    assert len({d for _, d, _ in observed}) == 2
    assert all(c["nested"]["items"] == [] for c in configs.values())
    assert {c["map_config_path"] for _, _, c in observed} == {"A.json", "B.json"}


def test_shared_report_keeps_navigation_failures_and_excludes_infrastructure(tmp_path):
    tasks = [task("travel_uav", "test_seen", str(i)) for i in range(3)]
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", tasks, "frozen")
    for t, success, ne in zip(tasks[:2], (True, False), (5, 35)):
        current, attempt = q.claim([t], "A")
        q.accept(current, attempt["number"], {"cleanup_verified": True, "valid_score": True, "process": {},
                 "result": {"episode_id": t["episode_id"], "metrics": {"success": success, "navigation_error": ne}}})
    report = write_report(tmp_path, {"dataset": "travel_uav", "fingerprint": "frozen"}, q.state)
    group = report["groups"][0]
    assert report["valid_executions"] == 2 and group["planned"] == 3
    assert group["provisional"] == {"success": 50, "navigation_error": 20}
    assert not group["complete"] and all(value is None for value in group["paper"].values())


def test_top_level_provider_exhaustion_never_enters_valid_navigation_scores(tmp_path):
    current = task("travel_uav", "test_seen", "one")
    result = {"episode_id": "one", "metrics": {"success": False, "navigation_error": 50},
              "error": {"error_type": "ProviderRequestError", "retryable": True}, "infrastructure_error": None}
    assert not parallel.valid_result(result)
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", [current], "frozen")
    q.claim([current], "A")
    assert q.accept(current, 1, {"cleanup_verified": True, "valid_score": True, "process": {}, "result": result}) == "deferred"
    assert q.state["tasks"][q.key(current)]["result"] is None


def test_missing_metric_stays_blank_when_other_metrics_have_full_coverage(tmp_path):
    current = task("travel_uav", "test_seen", "one")
    q = recovery.ParallelRecoveryQueue(tmp_path / "progress.json", [current], "frozen")
    q.claim([current], "A")
    q.accept(current, 1, {"cleanup_verified": True, "valid_score": True, "process": {},
                         "result": {"metrics": {"success": False}}})
    report = write_report(tmp_path, {"dataset": "travel_uav", "fingerprint": "frozen"}, q.state)
    assert report["groups"][0]["paper"] == {"success": 0, "navigation_error": None}


def test_standalone_prepare_freezes_context_selection_and_generic_keys(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "configs/context.json").write_text('{}', encoding="utf-8")
    (workspace / "maps.json").write_text(json.dumps({"maps": [], "port_policy": {}}), encoding="utf-8")
    (workspace / ".env").write_text('', encoding="utf-8")
    config = workspace / "travel.json"
    config.write_text(json.dumps({"split": "test_seen", "map_config_path": "maps.json", "context_config_path": "configs/context.json"}), encoding="utf-8")
    class Adapter:
        def prepare(self, data, **_):
            return PreparedCampaignSplit(data, (task("travel_uav", "test_seen", "one"), task("travel_uav", "test_seen", "two")), {})
    monkeypatch.setattr(prepare, "adapter_for", lambda *_: Adapter())
    monkeypatch.setattr(prepare, "environment_diagnostics", lambda: {"cuda_available": True, "mocked": True})
    monkeypatch.setattr(prepare, "load_model_config", lambda *_: SimpleNamespace(redacted_summary=lambda: {"model": "fixed"}))
    output = tmp_path / "campaign"
    result = prepare.prepare(output, [config], "travel_uav", root=workspace)
    manifest = parallel.read_json(output / "manifest.json")
    assert result["tasks"] == 2 and manifest["completion_status"] == "campaign_complete"
    assert manifest["primary_config_key"] == "travel_uav/test_seen"
    assert Path(primary_config(manifest)["context_config_path"]).is_relative_to(output / "source")
    assert set(parallel.read_json(output / "progress.json")["tasks"]) == {"travel_uav/test_seen/one", "travel_uav/test_seen/two"}
    assert "configs/context.json" in manifest["source_hashes"]
    with pytest.raises(ValueError, match="already prepared"):
        prepare.prepare(output, [config], "travel_uav", root=workspace)


def test_frozen_shared_package_starts_in_a_cold_process_and_rejects_tampering(tmp_path, monkeypatch):
    from tests.test_aerial_vln_parallel_campaign import gpu_binding, catalog
    source = freeze_source(tmp_path)
    (source / "evaluation/benchmarks/aerial_vln/campaign.py").write_text("raise RuntimeError('Legacy driver imported')\n", encoding="utf-8")
    parallel.write_json(tmp_path / "maps.json", catalog())
    monkeypatch.setattr(parallel, "discover_gpu_bindings", lambda _: {"A": gpu_binding(0), "B": gpu_binding(2)})
    tasks = [task("travel_uav", "test_unseen_map", "one")]
    manifest = {"version": 2, "dataset": "travel_uav", "fingerprint": "frozen", "tasks": tasks,
                "configs": {"test_unseen_map": {"map_config_path": str(tmp_path / "maps.json")}}}
    parallel.write_json(tmp_path / "manifest.json", manifest)
    overlay = parallel.prepare_extension(tmp_path, manifest)
    assert "campaign/contracts.py" in overlay["source_hashes"] and "adapters/travel_uav.py" in overlay["source_hashes"]
    command = [sys.executable, overlay["controller_script"], "--output", str(tmp_path), "--help"]
    response = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert response.returncode == 0, response.stderr
    assert "prepare" in response.stdout and "validate" in response.stdout
    path = Path(overlay["controller_script"]).parent / "campaign/recovery.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    response = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert response.returncode != 0 and "Frozen campaign helper changed" in response.stderr
