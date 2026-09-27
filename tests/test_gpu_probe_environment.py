"""GPU admission and frozen-input checks without loading GPU models."""

import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from tests.manual.gpu_probe_environment import check_interpreter, validate_environment, verify_weights
from tests.manual.read_only_concurrency_probe import import_samples, sha, compare_with_serial, validate_condition, validate_mixed
from tests.manual.gpu_resource_monitor import summarize_resources
from tests.manual.task_profile import synchronize_devices


def env():
    return {"executable": "F:/conda_env/flyagent/python.exe", "dependencies": {"torch": "test+cu"},
            "cuda_build": "12.6", "cuda_available": True, "gpu_names": ["test gpu"], "disk_free_bytes": 10 * 1024 ** 3}


def test_environment_rejects_wrong_interpreter_cpu_and_child_drift():
    record = env()
    validate_environment(record, require_cuda=True, expected=record)
    with pytest.raises(ValueError, match="Wrong interpreter"):
        check_interpreter(actual="F:/Anaconda/python.exe")
    with pytest.raises(ValueError, match="CUDA required"):
        validate_environment({**record, "cuda_available": False}, require_cuda=True)
    with pytest.raises(ValueError, match="environment changed"):
        validate_environment({**record, "dependencies": {"torch": "changed"}}, expected=record)
    with pytest.raises(ValueError, match="Wrong interpreter"):
        validate_environment({**record, "executable": "F:/Anaconda/python.exe"}, expected=record)


def frozen_input(tmp_path):
    from PIL import Image
    original = tmp_path / "original"
    original.mkdir()
    image = original / "rgb.png"
    Image.new("RGB", (4, 4)).save(image)
    samples = [{"number": n, "case_id": f"case{n}", "benchmark": "aerial" if n <= 2 else "travel",
                "calls": [{"tool_name": "skill.visual_grounding.locate", "tool_call_id": f"c{n}_{i}",
                           "args": {"include_distance": False, "observation_id": f"obs{n}", "query": "building"}} for i in (1, 2)],
                "input_hashes": {str(image): sha(image)},
                "observation": {"observation_id": f"obs{n}", "frames": {"rgb": {"path": str(image)}}}}
               for n in range(1, 5)]
    path = original / "manifest.json"
    path.write_text(json.dumps({"samples": samples, "baseline_manifest_sha256": "original"}), encoding="utf-8")
    return path, image, samples


def test_frozen_manifest_preserves_calls_order_and_original_files(tmp_path):
    path, image, expected = frozen_input(tmp_path)
    original_bytes = path.read_bytes(), image.read_bytes()
    copied, baseline = import_samples(path, tmp_path / "new")
    assert baseline == "original"
    assert [s["calls"] for s in copied] == [s["calls"] for s in expected]
    assert [s["case_id"] for s in copied] == [s["case_id"] for s in expected]
    assert (path.read_bytes(), image.read_bytes()) == original_bytes
    assert all(sha(s["observation"]["frames"]["rgb"]["path"]) == sha(image) for s in copied)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="images changed"):
        import_samples(path, tmp_path / "invalid")


def test_serial_reference_detects_even_small_public_difference():
    result = {"ok": True, "message": "ok", "data": {"candidates": [{"confidence": .701}]}}
    serial = [{"results": [result]}]
    different = copy.deepcopy(serial)
    different[0]["results"][0]["data"]["candidates"][0]["confidence"] = .702
    compare_with_serial({2: serial})  # Rotated condition can finish before worker 1.
    with pytest.raises(ValueError, match="serial baseline"):
        compare_with_serial({2: different, 1: serial})


def test_gpu_condition_requires_actual_model_parameter_evidence():
    summary = {"completed_batches": 5, "incomplete_spans": 0, "peak_active_device": {"cuda:0": 1},
               "remaining_active_device": {"cuda:0": 0}, "loaded_models": [{"parameter_devices": ["cuda:0"]}]}
    validate_condition(summary, [], require_cuda=True)
    summary["loaded_models"][0]["parameter_devices"] = ["cpu"]
    with pytest.raises(ValueError, match="parameters"):
        validate_condition(summary, [], require_cuda=True)
    summary["peak_active_device"] = {"cpu": 1}
    with pytest.raises(ValueError, match="CPU fallback"):
        validate_condition(summary, [], require_cuda=True)


def test_cached_weights_changes_and_missing_resource_values(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original")
    records = {"grounding": {"files": {str(weights): sha(weights)}}}
    verify_weights(records)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="assets changed"):
        verify_weights(records)
    assert summarize_resources(tmp_path)["peak_child_rss_bytes"] is None
    rows = [{"available": False, "error": "missing driver"}, {"available": True, "devices": [
        {"uuid": "gpu", "name": "GPU", "used_mib": 123, "gpu_percent": None}]}]
    (tmp_path / "gpu_resources.jsonl").write_text('\n'.join(map(json.dumps, rows)))
    result = summarize_resources(tmp_path)
    assert result["gpu_query_errors"] == 1
    assert result["device_totals"]["gpu"]["peak_gpu_percent"] is None


def test_mixed_batch_requires_fifo_delivery_and_nonstarvation():
    ids = ["vision1", "vision2", "constructed_knowledge_1", "constructed_knowledge_2"]
    row = {"constructed_input": True, "tool_call_ids": ids,
           "results": [{}, {}, {"data": {"name": "guide", "body_included": True}},
                       {"data": {"name": "guide", "body_included": False}}]}
    events = [{"tool_call_id": c, "event": "read_call_started", "monotonic_sec": i} for c, i in zip(ids, [0, 0, 0, 2])]
    events += [{"tool_call_id": c, "event": "read_call_completed", "monotonic_sec": i} for c, i in zip(ids, [10, 20, 1, 3])]
    summary = {"call_events": events}
    validate_mixed(summary, [row], 4)
    events[-2]["monotonic_sec"] = 30
    with pytest.raises(ValueError, match="submission order"):
        validate_mixed(summary, [row], 4)
    events[3]["monotonic_sec"] = 31
    with pytest.raises(ValueError, match="in flight"):
        validate_mixed(summary, [row], 4)


def test_diagnostic_sync_only_targets_actual_cuda_devices(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=calls.append)))
    synchronize_devices(["cpu"])
    assert calls == []
    synchronize_devices(["cuda:0", "cuda:1"])
    assert calls == ["cuda:0", "cuda:1"]
