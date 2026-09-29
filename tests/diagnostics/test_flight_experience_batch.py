"""Manual subset orchestration invariants, without simulation or model calls."""
import pytest
from types import SimpleNamespace
from tests.manual import flight_experience_batch_probe as probe


def test_subset_preserves_source_identity():
    source = {"cases": [{"case_id": str(i), "benchmark": "aerial" if i < 10 else "travel"}
                        for i in range(20)]}
    chosen = probe.select_cases(source, list(probe.DEFAULT_SELECTION))
    assert [c["case_id"] for c in chosen] == [str(i - 1) for i in probe.DEFAULT_SELECTION]
    with pytest.raises(ValueError):
        probe.select_cases(source, list(range(1, 11)))
    with pytest.raises(ValueError):
        probe.select_cases(source, [1] * 10)


def test_uniform_frames_are_ordered_and_include_endpoints():
    ids = [f"obs_{i}" for i in range(35)]
    selected = probe.uniform_frames(ids)
    assert len(selected) == len(set(selected)) == 8
    assert selected[0] == ids[0] and selected[-1] == ids[-1]
    assert selected == sorted(selected, key=ids.index)
    assert probe.uniform_frames(ids[:3]) == ids[:3]


def test_resume_never_overwrites_and_requires_cleanup(tmp_path):
    item = {"benchmark": "aerial", "case_id": "example"}
    original = probe.next_attempt(tmp_path, 0, item, False)
    original.mkdir(parents=True)
    with pytest.raises(ValueError):
        probe.next_attempt(tmp_path, 0, item, True)
    probe.batch.write_json(original / "status.json", {"status": "completed", "cleanup_verified": True})
    assert probe.next_attempt(tmp_path, 0, item, True) is None
    probe.batch.write_json(original / "status.json", {"status": "infrastructure_error", "cleanup_verified": False})
    with pytest.raises(ValueError):
        probe.next_attempt(tmp_path, 0, item, True)
    probe.batch.write_json(original / "status.json", {"status": "infrastructure_error", "cleanup_verified": True})
    assert probe.next_attempt(tmp_path, 0, item, True).name.endswith("_recovery_01")


def test_experience_resume_preserves_failures_and_detects_interruption(tmp_path, monkeypatch):
    output = tmp_path / "experiment"
    output.mkdir()
    artifact = output / "experiment.json"
    probe.batch.write_json(artifact, {"status": "failed"})

    def unexpected_call(*args, **kwargs):
        pytest.fail("A saved failure must not silently trigger another model call")

    monkeypatch.setattr(probe.subprocess, "run", unexpected_call)
    probe.invoke_experience(tmp_path / "source", output, "run", [])
    probe.batch.write_json(artifact, {"status": "running"})
    with pytest.raises(ValueError, match="Interrupted"):
        probe.invoke_experience(tmp_path / "source", output, "run", [])
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="before report"):
        probe.invoke_experience(tmp_path / "source", tmp_path / "missing", "run", [])
