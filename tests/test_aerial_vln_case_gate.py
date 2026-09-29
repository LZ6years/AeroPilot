"""Offline explicit-subset checks; no model, SSH or simulator execution."""
import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from tests.support.environment import (
    SelectedCaseProbe,
    _probe_checkpoint,
    _write_probe_images,
    aerial_vln_case_gate,
    audit_case,
    validate_episode_ids,
)
from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig


def test_strict_gate_failure_is_visible_to_existing_batch_abort(tmp_path, monkeypatch):
    from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
    probe = SelectedCaseProbe.__new__(SelectedCaseProbe)
    probe.strict_case_gates = True
    probe.config = SimpleNamespace(scene_map={"12": "map"})
    probe.map_packages_by_id = {"map": object()}
    probe._case_reviews = {"case": object()}
    case = SimpleNamespace(episode_id="case", scene_id="12")
    case_dir = tmp_path / "cases/case"
    case_dir.mkdir(parents=True)
    (case_dir / "case_result.json").write_text('{}')
    monkeypatch.setattr(AerialVlnBenchmarkRunner, "run_case", lambda *a, **k: {"completed": False})
    from evaluation.benchmarks.core.review import BenchmarkReviewRenderer
    def render(*args):
        (case_dir / "review.html").write_text("rendered")
    monkeypatch.setattr(BenchmarkReviewRenderer, "write_case", render)
    def audit(*args):
        assert (case_dir / "review.html").read_text() == "rendered"
        return {"passed": False}
    monkeypatch.setattr(aerial_vln_case_gate, "audit_case", audit)
    result = probe.run_case(case, tmp_path)
    assert result["infrastructure_error"]["stage"] == "manual_audit_gate"
    assert json.loads((case_dir / "case_result.json").read_text())["manual_gate_passed"] is False


def test_manual_gate_missing_artifacts_fails_closed(tmp_path):
    report = audit_case(tmp_path, {}, None)
    assert report["passed"] is False
    assert "FileNotFoundError" in report["error"]


def test_manual_gate_uses_runtime_plugin_tool_surface():
    registry = aerial_vln_case_gate._audit_registry()

    assert "skill.visual_grounding.locate" in registry.names()


@pytest.mark.parametrize("ids", [[], {}, [""], [" x"], [1], ["x", "x"]])
def test_invalid_ids_fail(ids):
    with pytest.raises(ValueError):
        validate_episode_ids(ids)


def test_provider_probe_uses_matching_rgb_dimensions(tmp_path):
    from PIL import Image
    for observation_id, path in _write_probe_images(tmp_path / "rgb").items():
        item = _probe_checkpoint(
            path, observation_id, 1
        ).fragments[0].model_inputs[0]["media_ref"]
        with Image.open(path) as image:
            assert image.size == (item["display_width"], item["display_height"]) == (240, 180)
        assert item["height"] == item["display_height"]
        assert "Original RGB" in item["usage"]


def test_complete_population_order_and_manifest(monkeypatch):
    config_path = Path("evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_probe.local")
    if not config_path.exists():
        pytest.skip("Local dataset configuration absent")
    config = AerialVlnEvalConfig.from_file(config_path)
    if not Path(config.dataset_root).exists():
        pytest.skip("Local dataset absent")
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not execute a case or load a model")
    monkeypatch.setattr(SelectedCaseProbe, "run_case", forbidden)
    monkeypatch.setattr(SelectedCaseProbe, "_model_config", forbidden)
    ids = json.loads(Path("tests/manual/aerial_vln_simple_instruction10.json").read_text())
    prepared = SelectedCaseProbe(config, episode_ids=ids).prepare_evaluation(require_execution_ready=True)
    assert [c.episode_id for c in prepared.cases] == ids
    assert prepared.selection.population_case_count == 333
    assert prepared.selection.selected_counts == {"low": 8, "medium": 2, "high": 0}
    assert prepared.selection.q33_m == 161
    assert len({c.trajectory_id for c in prepared.cases}) == 10
    manifest = prepared.selection.to_dict()
    assert manifest["dataset_sha256"] == prepared.provenance.split_sha256
    assert [c["instruction"] for c in manifest["manual_cases"]] == [c.instruction for c in prepared.cases]
    with pytest.raises(ValueError, match="Unknown episode"):
        SelectedCaseProbe(config, episode_ids=["missing"]).prepare_evaluation(require_execution_ready=False)
