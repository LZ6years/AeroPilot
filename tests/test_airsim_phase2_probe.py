"""The diagnostic batch must retain original identities, failures and cleanup gates."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.manual.airsim_phase2_probe import experiment_cases, export_review, performance_summary
from evaluation.benchmarks.travel_uav.runner import _infrastructure_error


def test_frozen_order_and_limits(tmp_path, monkeypatch):
    import tests.manual.airsim_phase2_probe as probe
    cases = [{} for _ in range(20)]
    cases[12] = {"case_id": "historical13", "scene": "ModernCityMap"}
    cases[15] = {"case_id": "historical16", "scene": "NewYorkCity"}
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    monkeypatch.setattr(probe, "BASELINE", tmp_path)
    cases, sequence = experiment_cases("normal", 10)
    assert len(cases) == 20
    assert [c["reuse"] for c in cases] == [False, True] * 10
    assert len({c["case_id"] for c in cases}) == 1
    cases, sequence = experiment_cases("case16", 3)
    assert len(cases) == 3 and sequence[0]["args"] == {"vx": 0, "vy": 0, "vz": 2, "duration": 3}
    with pytest.raises(ValueError):
        experiment_cases("case16", 4)


def test_native_exit_still_exports_partial_html(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    status = {"exit_code": -1, "cleanup_error": "unverified"}
    result = export_review(output, tmp_path / "review", status)
    assert result["images"] == 0
    assert "无最终评分" in Path(result["path"]).read_text(encoding="utf-8")
    assert json.loads((tmp_path / "review/audit.json").read_text(encoding="utf-8"))["supervision"] == status


def test_scene_recovery_is_infrastructure_even_with_valid_capabilities():
    error = {"error_type": "SceneRecoveryRequired", "error": "freeze failed"}
    execution = SimpleNamespace(outcome=SimpleNamespace(error=error))
    assert _infrastructure_error(execution=execution, setup_error=None,
                                 capabilities={"protocol": "airsim"}, cleanup_errors=[]) == error


def test_missing_timings_are_not_zero():
    result = performance_summary([])
    assert result["False"]["p50_seconds"] is None
    assert not result["recommend_enable"]


def test_supervisor_retains_native_exit_and_cleanup_failure_blocks_next_scene(tmp_path, monkeypatch):
    import sys
    import tests.manual.airsim_phase2_probe as probe
    from tests.manual.process_diagnostics import supervise
    directory = tmp_path / "native"
    status = supervise([sys.executable, "-c", "import os; os._exit(7)"], directory)
    assert status["exit_code"] == 7
    assert (directory / "process.json").exists()
    monkeypatch.setattr(probe, "load_map_config", lambda _: ([SimpleNamespace(scene_id="X")], None))
    monkeypatch.setattr(probe, "cleanup_owned", lambda *a, **k: [{"remote_stop": {"cleanup_verified": False}, "endpoint": {"status": "unreachable"}}])
    with pytest.raises(RuntimeError, match="cleanup not verified"):
        probe.verify_cleanup(directory, {"scene": "X"}, status["descendants"])
    review = export_review(directory, tmp_path / "review", status)
    assert Path(review["path"]).exists()
