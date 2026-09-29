"""Regression contracts for RGB input, immutable plans and compact audit review.

These deterministic tests verify plumbing, not a live model's semantic quality.
"""

import json
import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.agent.context.media import MediaHydrator
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.memory.context.registry import ContextModuleRegistry, ContextModuleConfigError
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.runtime.context.media_store import ModelMediaStore
from evaluation.benchmarks.core.review import _initial_plan_review, _phase_decision_review
from evaluation.benchmarks.core.audit import ResolvedFrameAsset


def write_test_png(path, color=(90, 30, 10), size=(8, 8)):
    Image.new("RGB", size, color).save(path, format="PNG")


def decode_data_url_image(value):
    return Image.open(BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")


@pytest.mark.parametrize("overlay", [False, True])
def test_json_image_mode_preserves_depth_source(tmp_path, overlay):
    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.npy"
    write_test_png(rgb, color=(90, 30, 10), size=(8, 8))
    np.save(depth, np.full((8, 8), 2.9))
    original_depth = depth.read_bytes()
    registry = ContextModuleRegistry.from_dict({
        "context_modules": {"image": "default"},
        "context_options": {"image": {"include_current_depth": overlay}},
    })
    registry.modules["image"].bind_media_writer(ModelMediaStore(tmp_path))
    writer = MemoryEventWriter(tmp_path)
    memory = FlightMemory(
        "run", "map", "scene", "uav", memory_reader=MemoryRecordReader(tmp_path)
    )
    observation = {
        "frames": {"rgb": {"path": str(rgb)}, "depth": {"path": str(depth), "depth_unit": "meter"}},
        "modalities": ["rgb", "depth"],
    }
    refs = writer.write_observation(observation)
    memory.append_observation_ref(refs["observation_ref"])
    memory.append_frame_refs(refs["frame_refs"])
    result = registry.modules["image"].build(
        MemorySourceBundle(memory), TaskContextInput("Inspect")
    )
    media_ref = result.fragment.model_inputs[0]["media_ref"]
    rendered = decode_data_url_image(
        MediaHydrator(base_dir=tmp_path).hydrate(media_ref).data_url
    )
    assert depth.read_bytes() == original_depth
    assert result.audit.metadata["visual_evidence"]["has_depth"] is overlay
    assert (rendered.size == (8, 8)) is not overlay
    assert (rendered.getpixel((3, 3)) == (90, 30, 10)) is not overlay
    assert (media_ref.get("color_bar_height", 0) > 0) is overlay


@pytest.mark.parametrize("options", [None, {"image": []}, {"missing": {}},
    {"image": {"include_current_depth": "false"}}, {"image": {"unknown": 1}},
    {"image": {"depth_display_max_m": -1}}, {"image": {"depth_display_max_m": True}},
    {"image": {"depth_display_max_m": float("nan")}}])
def test_module_options_fail_explicitly(options):
    with pytest.raises(ContextModuleConfigError):
        ContextModuleRegistry.from_dict({"context_modules": {"image": "default"}, "context_options": options})


def test_initial_plan_uses_only_authoritative_event():
    audit = SimpleNamespace(events=[])
    assert "unavailable" in _initial_plan_review(audit)
    audit.events = [{"event": "initial_semantic_plan", "payload": {
        "parse_result": {"plan": {"phases": [{"index": 1, "name": "<route>", "goal": "first", "done_when": ["reached"]}]}},
        "protocol_context": "episode rule <test>",
    }}, {"event": "phase_strategy_updated", "payload": {"goal": "future plan"}}]
    rendered = _initial_plan_review(audit)
    assert "&lt;route&gt;" in rendered and "&lt;test&gt;" in rendered
    assert "future plan" not in rendered


def test_initial_plan_survives_observation_time_boundary_but_not_freeze(tmp_path):
    from evaluation.benchmarks.core import EpisodeAuditCursor, BenchmarkAuditReader
    from tests.support.benchmark import append_jsonl
    events = tmp_path / "events.jsonl"
    append_jsonl(events, {"event": "initial_semantic_plan", "timestamp": "2026-01-01T00:00:01+00:00",
                        "payload": {"instruction_id": "inst1", "parse_result": {"plan": {"phases": []}}}})
    append_jsonl(tmp_path / "memory/observations.jsonl", {"id": "obs1", "timestamp": "2026-01-01T00:00:02+00:00"})
    append_jsonl(tmp_path / "memory/agent_transcript.jsonl", {
        "schema_version": "agent_transcript_v2", "instruction_id": "inst1",
        "kind": "user_instruction", "timestamp": "2026-01-01T00:00:01+00:00", "payload": {}})
    append_jsonl(tmp_path / "memory/agent_transcript.jsonl", {
        "schema_version": "agent_transcript_v2", "instruction_id": "inst1",
        "kind": "state_checkpoint", "timestamp": "2026-01-01T00:00:02.100000+00:00",
        "payload": {"schema_version": "state_checkpoint_v2", "checkpoint_id": "checkpoint_0001"}})
    cursor = EpisodeAuditCursor.freeze(tmp_path)
    append_jsonl(events, {"event": "initial_semantic_plan", "timestamp": "2026-01-01T00:00:03+00:00",
                        "payload": {"instruction_id": "inst1", "fake": True}})
    audit = BenchmarkAuditReader().read(cursor)
    plans = [e for e in audit.events if e["event"] == "initial_semantic_plan"]
    assert len(plans) == 1 and "fake" not in plans[0]["payload"]


def test_compact_timeline_keeps_unapplied_strategy_and_explicit_links(tmp_path):
    image = tmp_path / "rgb.png"
    write_test_png(image)
    record = {"path": "decision.json", "decision_id": "d1", "decision_index": 1,
              "phase_key": "1:p", "checkpoint_observation_id": "obs1",
              "model_context": {"current_phase": {"name": "p", "goal": "route"}},
              "model_rounds": [{"canonical_turn": {"public_intent": "<script>bad</script>",
                  "tool_calls": [{"tool_call_id": "c1", "tool_name": "runtime.commit_phase_turn",
                                  "args": {"phase_strategy": {"active_stage": 2}}}]}}]}
    audit = SimpleNamespace(planner_debug=[record], events=[], agent_transcript=[{
        "kind": "tool_result", "payload": {"tool_call_id": "c1", "status": "rejected"}}])
    assets = {"decision.json": [{"path": str(image), "observation_id": "obs1"}]}
    frame = ResolvedFrameAsset("frame1", "obs1", "rgb", None, str(image), None, None, None)
    before = json.dumps(record)
    rendered = _phase_decision_review(audit, (frame,), tmp_path, assets)
    assert "尚未初始化" in rendered and "未应用" in rendered
    assert "&lt;script&gt;" in rendered and '<script>bad' not in rendered
    assert 'id="decision-d1"' in rendered and "Captured CURRENT" in rendered
    assert json.dumps(record) == before
    audit.planner_debug.append({"decision_id": "orphan"})
    assert "ungrouped" in _phase_decision_review(audit, (), tmp_path, {})


def test_single_case_probe_preserves_original_case_identity():
    from tests.support.environment import SelectedCaseProbe
    from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
    # Repository-local dataset check; no network, model or engine is constructed.
    from pathlib import Path
    config_path = Path("evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_probe.local")
    if not config_path.exists():
        pytest.skip("Optional local AerialVLN-S probe config is absent")
    config = AerialVlnEvalConfig.from_file(config_path)
    if not Path(config.dataset_root).exists():
        pytest.skip("Optional local AerialVLN-S dataset is absent")
    episode = "3G0WWMR1U4IB7QGQVGGALABHGGXNQO"
    prepared = SelectedCaseProbe(config, episode).prepare_evaluation(require_execution_ready=False)
    assert [c.episode_id for c in prepared.cases] == [episode]
    assert prepared.selection.selected_counts == {"low": 1, "medium": 0, "high": 0}
    assert "explicit episode ID subset" in prepared.selection.to_dict()["method"]
