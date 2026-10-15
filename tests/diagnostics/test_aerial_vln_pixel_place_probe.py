"""Evaluation-only cohort, snapshot and new-capability accounting contracts."""
import hashlib
import json
from pathlib import Path

from tests.manual import aerial_vln_plan_context_probe as probe
from tests.manual.aerial_vln_context_module_audit import offline_audit_registry
from tests.manual.aerial_vln_plan_context_acceptance import request_checks, delivered_media
from src.plugins import PluginRegistry
from src.tools.registry import default_registry


def test_pruned_depth_is_unknown_and_unexplained_missing_depth_is_failure(tmp_path):
    from PIL import Image
    from tests.manual.aerial_vln_pixel_place_audit import audit_capability_records
    run = tmp_path / "attempt/runs/one"
    frames = [{"id": "rgb", "observation_id": "obs", "modality": "rgb", "path": "rgb.png"},
              {"id": "depth", "observation_id": "obs", "modality": "depth", "path": "depth.npy"}]
    (run / "memory").mkdir(parents=True)
    (run / "memory/frames.jsonl").write_text('\n'.join(json.dumps(x) for x in frames), encoding="utf-8")
    Image.new('RGB', (4, 4)).save(run / 'rgb.png')
    events = [{"kind": "state_checkpoint", "payload": {"observation_id": "obs"}},
        {"kind": "assistant_turn", "payload": {"tool_calls": [{"tool_call_id": "one", "args": {"observation_id": "obs", "pixel_x": 1, "pixel_y": 1}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "one", "tool_name": "skill.pixel_depth.measure_pixel_depth", "status": "ok",
            "data": {"observation_id": "obs", "pixel_x": 1, "pixel_y": 1, "distance_status": "ok", "distance_m": 4}}}]
    assert audit_capability_records(run, events)["failures"] == ["pixel_depth_missing"]
    probe.write(tmp_path / 'attempt/receipt.json', {"cleanup_verified": True, "temporary_depth_bytes": 64})
    result = audit_capability_records(run, events)
    assert not result['failures'] and result['unavailable_checks']
    assert result['pixel_checks'][0]['raw_depth_verified'] is None


def test_high_sampling_is_order_and_score_independent():
    entries = [{"status": "scored", "task": {"episode_id": f"{scene}_{index}", "split": "val_seen",
        "scene_id": str(scene), "difficulty": "high"}, "result": {"metrics": {"success": False}}}
        for scene in (2, 5, 10, 12, 14) for index in range(8)]
    progress = {"tasks": dict(enumerate(entries))}
    selected = probe.select_hard_cases(progress)
    for entry in entries:
        entry["result"]["metrics"]["success"] = True
    assert selected == probe.select_hard_cases({"tasks": dict(enumerate(reversed(entries)))})
    assert [t["scene_id"] for t in selected] == [str(s) for s in (2, 5, 10, 12, 14) for _ in range(2)]
    expected = sorted([e["task"] for e in entries[:8]], key=lambda t: hashlib.sha256(
        f"20261011:2:high:{t['episode_id']}".encode()).hexdigest())[:2]
    assert selected[:2] == expected
    assert all("result" not in t for t in selected)


def test_plugin_profile_only_replaces_new_snapshot(tmp_path):
    workspace = tmp_path / "workspace"
    default = workspace / "configs/plugins.json"
    profile = workspace / "configs/pixel.json"
    probe.write(default, {"old": True}); probe.write(profile, {"pixel": True})
    original = default.read_bytes()
    root = tmp_path / "experiment"
    hashes = probe.freeze_source(workspace, root / "source")
    probe.freeze_plugins(root, hashes, profile)
    assert default.read_bytes() == original
    assert probe.read(root / "source/configs/plugins.json") == {"pixel": True}
    assert hashes["configs/plugins.json"] == probe.digest(root / "source/configs/plugins.json")


def test_old_tools_are_available_only_in_offline_registry():
    offline = offline_audit_registry("configs/plugins.pixel_map.json", "configs/plugins.json")
    online = default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json"))
    assert "skill.visual_grounding.measure_visible_depth" in offline.names()
    assert "skill.visual_grounding.measure_visible_depth" not in online.names()
    assert "skill.pixel_depth.measure_pixel_depth" in online.names()


def test_plan_windows_are_separate_from_successful_actions_and_pixel_is_counted(tmp_path):
    run = tmp_path / "attempt/runs/native"
    events = [
        {"kind": "assistant_turn", "payload": {"tool_calls": [{"tool_call_id": "p1", "tool_name": "runtime.update_plan", "args": {"operation": "replace"}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "p1", "tool_name": "runtime.update_plan", "status": "ok", "data": {"plan": {"revision": 1, "steps": [{"id": "s1", "description": "old", "status": "in_progress"}]}}}},
        {"kind": "tool_result", "payload": {"tool_name": "uav.move_body", "status": "error"}},
        {"kind": "state_checkpoint", "payload": {"reason": "physical_window_completed", "observation_id": "o2"}},
        {"kind": "assistant_turn", "payload": {"tool_calls": [{"tool_call_id": "p2", "tool_name": "runtime.update_plan", "args": {"operation": "patch"}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "p2", "tool_name": "runtime.update_plan", "status": "ok", "data": {"plan": {"revision": 2, "steps": [{"id": "s1", "description": "corrected", "status": "in_progress"}]}}}},
        {"kind": "tool_result", "payload": {"tool_name": "skill.pixel_depth.measure_pixel_depth", "status": "ok", "data": {"distance_status": "ok", "distance_m": 12.4}}},
        {"kind": "tool_result", "payload": {"tool_name": "skill.pixel_depth.measure_pixel_depth", "status": "ok", "data": {"distance_status": "invalid_depth_at_point", "distance_m": None}}},
        {"kind": "tool_result", "payload": {"tool_name": "skill.place_memory.inspect_place", "status": "ok", "data": {"evidence_origin": "historical_memory"}}},
    ]
    path = run / "memory/agent_transcript.jsonl"; path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    stats = probe.run_stats({"run_dir": str(run)})
    assert stats["physical_windows_between_updates"] == [1]
    assert stats["flight_actions_between_updates"] == [0]
    assert stats["description_correction_count"] == 1
    assert stats["unfinished_final_steps"][0]["description"] == "corrected"
    assert stats["pixel_measurement"]["valid_distance_count"] == 1
    assert stats["measurement_valid_rate"] == .5
    assert stats["place_memory"]["inspect_place_calls"] == 1
    assert stats["context_images"]["delivery_rate"] is None


def test_twenty_case_report_keeps_cohort_versions_separate(tmp_path, monkeypatch):
    from tests.manual import aerial_vln_context_module_audit
    baseline = tmp_path / "baseline"; previous = tmp_path / "previous"; root = tmp_path / "new"
    tasks = [{"split": "val_seen", "episode_id": f"case{i}", "difficulty": "high" if i >= 10 else "low"} for i in range(20)]
    def result(t, success):
        return {"episode_id": t["episode_id"], "status": "ok", "completed": True, "stopped": False, "error": None,
            "run_dir": str(root / t["episode_id"]), "audit_path": str(root / t["episode_id"] / "audit.json"), "metrics": {"success": success, "oracle_success": success, "distance_to_goal": 8 if success else 40}}
    probe.write(baseline / "progress.json", {"tasks": {str(i): {"task": t, "status": "scored", "result": result(t, False)} for i, t in enumerate(tasks)}})
    probe.write(previous / "progress.json", {"cases": [{"task": t, "attempts": [{"result": result(t, True)}]} for t in tasks[:10]]})
    probe.write(root / "progress.json", {"status": "complete", "cases": [{"task": t, "status": "completed", "attempts": [{"result": result(t, True), "number": 1, "status": "completed", "directory": str(root / t["episode_id"])}]} for t in tasks]})
    probe.write(root / "manifest.json", {"baseline": str(baseline), "previous_root": str(previous), "limits": "test",
        "cohorts": [{"name": name, "episode_ids": [t["episode_id"] for t in selected], "primary_baseline": str(source)}
            for name, selected, source in [("original10", tasks[:10], previous), ("hard10", tasks[10:], baseline)]]})
    probe.write(root / "history_inventory.json", {"records": [], "read_errors": []})
    monkeypatch.setattr(probe, "run_stats", lambda _: {})
    monkeypatch.setattr(aerial_vln_context_module_audit, "write_module_report", lambda *args: None)
    data = probe.report(root)
    assert data["selected_cases"] == 20 and data["terminal_cases"] == 20
    assert data["matched_baseline"]["success"]["sum"] == 0
    assert data["success_transitions"] == {"failure_to_success": 20}
    assert data["cohorts"][0]["baseline"]["success"]["sum"] == 10
    assert data["cohorts"][1]["baseline"]["success"]["sum"] == 0
    assert data["pairs"][10]["baseline_source"] == str(baseline)
    assert "20/20" in (root / "comparison.md").read_text(encoding="utf-8")


def test_delivered_map_must_be_unique_latest_and_current_rgb_present():
    registry = default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json"))
    state = {"task_plan": None}
    model_round = {"current_task_state": state, "request": {"messages": [{"role": "system", "content": "CURRENT TASK STATE\n" + json.dumps(state)}],
        "tools": [s.to_provider_tool() for s in registry.model_specs()]},
        "request_media": [{"timeline_role": "CURRENT", "observation_id": "o2"},
            {"label": "CONTEXT IMAGE updated_at_observation_id=o2 frame_id=m2", "frame_id": "m2"}],
        "history_projection": {"selected_visual_evidence": [{"selection_tier": "latest_group", "frame_id": "m2"}]}}
    assert request_checks(model_round, require_map=True)[0] == []
    model_round["request_media"][1]["label"] = "CONTEXT IMAGE updated_at_observation_id=o1 frame_id=m2"
    assert "context_image_observation_stale" in request_checks(model_round, require_map=True)[0]
    model_round["request_media"].append(model_round["request_media"][1])
    assert "latest_context_image_not_unique" in request_checks(model_round, require_map=True)[0]


def test_sandbox_overrides_artifact_path_only_for_explicit_invalid_fixture(tmp_path):
    import numpy as np
    from PIL import Image
    from src.memory.stores import MemoryEventWriter
    from tests.manual.aerial_vln_pixel_place_audit import admit_saved_evidence
    original = tmp_path / "original"; original.mkdir()
    rgb = original / "rgb.png"; Image.new("RGB", (16, 16), "red").save(rgb)
    depth = original / "depth.npy"; np.save(depth, np.full((16, 16), 7.3, dtype=np.float32))
    writer = MemoryEventWriter(original)
    writer.write_observation({"pose": {"position": {"x": 0, "y": 0, "z": 5}, "euler_rad": {"yaw": 0}},
        "frames": {"rgb": {"path": str(rgb)}, "depth": {"path": str(depth),
            "depth_unit": "meter", "capture_source": "depth_perspective"}}, "source": "fixture"})
    (original / "memory/actions.jsonl").touch()
    previous = tmp_path / "previous"
    probe.write(previous / "progress.json", {"cases": [{"attempts": [{"result": {"run_dir": str(original)}}]}]})
    admitted = admit_saved_evidence(tmp_path / "new", previous, probe.write, probe.digest)
    assert admitted["passed"]
    assert admitted["derived_invalid_pixel"]["distance_status"] == "invalid_depth_at_point"
    assert np.isfinite(np.load(depth)).all()


def test_old_map_audit_label_is_read_from_original_request_without_mutation():
    import copy
    label = "CONTEXT IMAGE updated_at_observation_id=o1 frame_id=m1\nimage_size=1600x1100"
    request = {"request": {"messages": [{"content": [{"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": "[provider_media sha256=real]"}}]}]},
        "request_media": [{"sha256": "real", "request_path": ["messages", 0, "content", 1, "image_url", "url"]}]}
    before = copy.deepcopy(request)
    media = delivered_media(request)
    assert media[0]["frame_id"] == "m1" and media[0]["updated_at_observation_id"] == "o1"
    assert media[0]["label_source"] == "captured_adjacent_text"
    assert request == before


def test_immediate_saved_map_identity_requires_captured_latest_image_digest():
    label = "TOOL EVIDENCE observation_id=o1. Task-local visited-place memory, not current camera visibility."
    m = {"request": {"messages": [{"content": [{"type": "text", "text": label}, {"type": "image_url"}]}]},
        "request_media": [{"sha256": "exact", "request_path": ["messages", 0, "content", 1, "image_url", "url"], "label": label}],
        "history_projection": {"selected_visual_evidence": [{"selection_tier": "latest_group", "sha256": "exact", "frame_id": "map", "observation_id": "o1"}]}}
    restored = delivered_media(m)[0]
    assert restored["label"] == label
    assert restored["context_label_variant"] == "tool_result_caption"
    m["request_media"][0]["sha256"] = "different"
    assert "context_image_classification" not in delivered_media(m)[0]


def test_saved_map_new_caption_retains_visibility_boundary(monkeypatch):
    from tests.manual import aerial_vln_plan_context_acceptance as acceptance
    registry = default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json"))
    state = {"task_plan": None}
    model_round = {"current_task_state": state, "request": {"messages": [
        {"role": "system", "content": "CURRENT TASK STATE\n" + json.dumps(state)}],
        "tools": [s.to_provider_tool() for s in registry.model_specs()]},
        "history_projection": {"selected_visual_evidence": [{"selection_tier": "latest_group", "frame_id": "map"}]}}
    media = [{"timeline_role": "CURRENT", "observation_id": "obs"},
        {"context_image_classification": "captured_latest_group", "context_label_variant": "tool_result_caption",
         "selected_observation_id": "obs", "frame_id": "map", "label": "not current visibility or verified free space"}]
    monkeypatch.setattr(acceptance, 'delivered_media', lambda _: media)
    assert acceptance.request_checks(model_round, require_map=True)[0] == []
    media[1]['label'] = 'current visibility'
    assert 'context_image_visibility_boundary_missing' in acceptance.request_checks(model_round, require_map=True)[0]


def test_resume_rechecks_failed_technical_gate_without_repeating_terminal_navigation(tmp_path, monkeypatch):
    row = {"task": {"episode_id": "one"}, "status": "completed", "attempts": [{"result": {"metrics": {"success": False}}}]}
    manifest = {"config": {}, "tasks": [row["task"]], "technical_gate_indices": [0]}
    probe.write(tmp_path / "progress.json", {"status": "paused_audit", "cases": [row]})
    probe.write(tmp_path / "admission.json", {"status": "passed", "plan_probe_status": "passed", "groups_completed": 3,
        "receipts": [{"status": "ok", "attempt": 1}] * 15})
    probe.write(tmp_path / "historical_request_replay.json", {"passed": True})
    monkeypatch.setattr(probe, "guard", lambda _: manifest)
    monkeypatch.setattr(probe, "load_credentials", lambda _: None)
    checks = []
    monkeypatch.setattr(probe, "audit_gate", lambda *args: checks.append(args[-1]) or {"passed": False})
    monkeypatch.setattr(probe, "supervise", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Navigation must not repeat")))
    monkeypatch.setattr(probe, "report", lambda root: probe.read(root / "progress.json"))
    result = probe.run(tmp_path, resume=True)
    assert result["status"] == "paused_audit" and len(checks) == 1
    assert result["cases"][0]["attempts"] == row["attempts"]


def test_failed_window_without_new_image_is_counted_from_explicit_trace(tmp_path):
    run = tmp_path / "run"; (run / "memory").mkdir(parents=True)
    events = [{"kind": "tool_result", "payload": {"tool_call_id": "move", "tool_name": "uav.move_body", "status": "error"}}]
    (run / "memory/agent_transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    audit = tmp_path / "evaluation/audit.json"
    probe.write(audit.parent / "trace.json", {"physical_window_count": 1, "physical_windows": [
        {"window_index": 1, "ordered_steps": [{"tool_call_id": "move"}]}]})
    stats = probe.run_stats({"run_dir": str(run), "audit_path": str(audit)})
    assert stats["physical_windows"] == 1 and stats["observed_physical_windows"] == 0
    assert stats["physical_windows_after_last_update"] == 1
    assert stats["flight_actions_after_last_update"] == 0
    assert stats["physical_windows_source"] == "explicit_trace_windows"
