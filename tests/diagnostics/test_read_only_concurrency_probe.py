"""Offline selection, numerical comparison and partial review contracts."""

import json

from tests.manual.read_only_concurrency_probe import accepted_pairs, public_signature, percentile, render, condition_summary


def test_selection_requires_both_accepted_calls_on_same_observation():
    calls = [{"tool_name": "skill.visual_grounding.locate", "tool_call_id": str(n),
              "args": {"observation_id": "obs1"}} for n in range(2)]
    events = [{"kind": "assistant_turn", "payload": {"tool_calls": calls}}]
    events.extend({"kind": "tool_result", "payload": {"tool_call_id": str(n), "status": "ok"}} for n in range(2))
    assert list(accepted_pairs({"agent_transcript": events})) == [calls]
    events[-1]["payload"]["status"] = "error"
    assert not list(accepted_pairs({"agent_transcript": events}))
    events[-1]["payload"]["status"] = "ok"
    calls[-1]["args"]["observation_id"] = "obs2"
    assert not list(accepted_pairs({"agent_transcript": events}))


def test_signature_keeps_measurement_but_not_artifact_paths():
    result = {"ok": True, "message": "ok", "data": {"candidates": [{"distance_m": None}], "_flyagent_artifact_refs": ["a"]}}
    second = {**result, "data": {**result["data"], "_flyagent_artifact_refs": ["b"]}}
    assert public_signature(result) == public_signature(second)
    second["data"]["candidates"] = [{"distance_m": 0}]
    assert public_signature(result) != public_signature(second)
    assert percentile([], .95) is None
    assert percentile([1, 2, 3], .5) == 2


def test_partial_review_has_real_image_both_pages_and_no_score(tmp_path):
    from PIL import Image
    source = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(source)
    review = tmp_path / "review"
    review.mkdir()
    manifest = {"review_root": str(review), "limitations": "offline", "samples": [{
        "number": 1, "benchmark": "aerial", "case_id": "original", "observation": {
            "observation_id": "obs1", "frames": {"rgb": {"path": str(source)}}}}]}
    render(tmp_path, manifest, [], state="paused", error={"type": "NativeExit"})
    for name in ("review", "audit_review"):
        text = (review / "cases/01" / (name + ".html")).read_text(encoding="utf-8")
        assert 'class="phase-timeline"' in text and 'src="rgb.png"' in text
        assert "无导航评分" in text
    assert json.loads((review / "resource_validation.json").read_text())["image_references"] == 2
    summary = condition_summary(tmp_path)
    assert summary["cold_sec"] is None and summary["completed_batches"] == 0


def test_device_summary_separates_load_forward_and_submission(tmp_path):
    events = [{"event": "read_call_queued"}, {"event": "read_call_queued"}, {"event": "read_call_started"},
              {"event": "model_loaded"}, {"event": "device_execution_started", "device": "cuda:0", "wait_sec": .3},
              {"event": "device_execution_finished", "device": "cuda:0"},
              {"event": "begin", "id": 1, "name": "vision.grounding.forward", "time": 1, "thread": 5, "parent": None},
              {"event": "end", "id": 1, "time": 2}]
    (tmp_path / "timings.jsonl").write_text('\n'.join(map(json.dumps, events)), encoding="utf-8")
    summary = condition_summary(tmp_path)
    assert summary["submitted_calls"] == 2 and summary["started_calls"] == 1
    assert summary["forward_calls"] == {"grounding": 1, "sam": 0}
    assert summary["peak_active_device"] == {"cuda:0": 1}
    assert summary["device_wait_sec"] == .3
