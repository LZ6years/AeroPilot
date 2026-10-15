"""Lightweight persistence preserves model evidence and measurable requests."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.experiments.compact_logging import MetricsRunLogger, capture_metrics, reference_recorded_path
from evaluation.experiments.compression_cost import effective_planner_calls, effective_totals
from src.memory.stores import MemoryEventWriter
from src.runtime.task.journal import RuntimeJournal


def test_light_frame_persistence_does_not_mutate_observation_bytes_or_identity(tmp_path):
    image = tmp_path / "rgb.png"
    raw = b"original image bytes"
    image.write_bytes(raw)
    frame = {"artifact_path": str(image), "base64": base64.b64encode(raw).decode(), "width": 256, "height": 256}
    original = copy.deepcopy(frame)
    for name, enabled in (("full", True), ("compact", False)):
        writer = MemoryEventWriter(tmp_path / name, persist_inline_media=enabled)
        writer.write_frame("obs_0001", "rgb", frame, timestamp="fixed")
    full = json.loads((tmp_path / "full/memory/frames.jsonl").read_text(encoding="utf-8"))
    compact = json.loads((tmp_path / "compact/memory/frames.jsonl").read_text(encoding="utf-8"))
    assert frame == original
    assert full["metadata"]["sha256"] == compact["metadata"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert compact["metadata"]["artifact_path"] == full["metadata"]["artifact_path"]
    assert "base64" not in compact["metadata"]
    assert compact["id"] == full["id"]


def test_small_metrics_receipt_reuses_actual_image_and_keeps_cost_and_selection(tmp_path):
    raw = b"original image bytes"
    digest = hashlib.sha256(raw).hexdigest()
    image = tmp_path / "artifacts/rgb.png"
    image.parent.mkdir()
    image.write_bytes(raw)
    logger = MetricsRunLogger(tmp_path)
    journal = RuntimeJournal(logger)
    exchange = {"request": {"messages": "LARGE_ORIGINAL_REQUEST" * 10000},
                "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 10}, "private_reasoning": "PRIVATE"},
                "transport": {"final_status": "success", "attempt_count": 1,
                              "attempts": [{"status": "success", "elapsed_seconds": 5}]},
                "history_projection": {"compression_enabled": False, "compaction_status": "disabled",
                    "estimated_input_tokens": 110, "total_image_count": 1,
                    "selected_visual_evidence": [{"sha256": digest, "ref": "artifacts/rgb.png", "source_kind": "observation"}]},
                "media": [{"data": raw, "sha256": digest, "suffix": ".png", "observation_id": "obs_0001"}]}
    original = copy.deepcopy(exchange)
    planner = SimpleNamespace(last_exchange=exchange)
    session = SimpleNamespace(transcript=SimpleNamespace(instruction_id="instruction"))
    turn = SimpleNamespace(to_dict=lambda: {"public_intent": "Move forward", "tool_calls": []})
    capture_metrics(journal, planner, session, "decision_0001", turn=turn)
    logger.close()
    file = next((tmp_path / "debug/model_call_metrics").rglob("*.json"))
    text = file.read_text(encoding="utf-8")
    assert "LARGE_ORIGINAL_REQUEST" not in text and "PRIVATE" not in text
    assert len(text) < 5000 and exchange == original
    assert not (tmp_path / "debug/provider_media").exists()
    values = effective_totals(effective_planner_calls(tmp_path))
    assert values["calls"] == 1 and values["input_tokens"] == 100 and values["decision_latency_seconds"] == 5
    assert values["peak_estimated_input_tokens"] == 110


def test_review_asset_references_original_without_making_a_copy(tmp_path):
    image = tmp_path / "run/artifacts/rgb.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"same")
    destination = tmp_path / "case/review_assets"
    result = reference_recorded_path("artifacts/rgb.png", tmp_path / "run", destination, "frame")
    assert result == image and not destination.exists()
