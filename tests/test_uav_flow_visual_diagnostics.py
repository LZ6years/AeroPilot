"""Tests for UAV-Flow benchmark visual diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.visual_diagnostics import write_visual_diagnostics
from src.agent.config import ModelConfig
from src.llm import ModelResponse


def make_case() -> UavFlowCase:
    return UavFlowCase(
        case_id="visual",
        file_name="visual.json",
        source_path="visual.json",
        class_name="Move",
        instruction="Move to a position near the person",
        instruction_unified="Move to a position near the person",
        selected_instruction="Move to a position near the person",
        initial_pos=[0, 0, 0, 0, 0, 0],
    )


def write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path)


def test_visual_diagnostics_writes_frame_only_inventory_without_truth_leak(tmp_path):
    run_dir = tmp_path / "run"
    case_dir = tmp_path / "case"
    write_rgb(run_dir / "artifacts" / "rgb" / "frame_000001.png")

    result = write_visual_diagnostics(
        make_case(),
        run_dir=run_dir,
        case_dir=case_dir,
        enabled=False,
        model_config=ModelConfig(model_name="test", api_key="secret"),
    )

    payload = json.loads((case_dir / "visual_diagnostics.json").read_text(encoding="utf-8"))
    assert result["frame_count"] == 1
    assert result["judged_count"] == 0
    assert result["truth_leak_check"]["status"] == "passed"
    assert payload["records"][0]["status"] == "frame_only"
    assert "target_pos" not in json.dumps(payload, ensure_ascii=False)


def test_visual_diagnostics_handles_enabled_mode_without_api_key(tmp_path):
    run_dir = tmp_path / "run"
    case_dir = tmp_path / "case"
    write_rgb(run_dir / "artifacts" / "rgb" / "frame_000001.png")

    result = write_visual_diagnostics(
        make_case(),
        run_dir=run_dir,
        case_dir=case_dir,
        enabled=True,
        model_config=ModelConfig(model_name="test", api_key=""),
    )

    payload = json.loads((case_dir / "visual_diagnostics.json").read_text(encoding="utf-8"))
    assert result["enabled"] is True
    assert result["judged_count"] == 0
    assert payload["records"][0]["status"] == "skipped"
    assert payload["records"][0]["person_visible"] is None


def test_visual_diagnostics_reuses_one_sdk_client_for_sampled_frames(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    case_dir = tmp_path / "case"
    write_rgb(run_dir / "artifacts" / "rgb" / "frame_000001.png")
    write_rgb(run_dir / "artifacts" / "rgb" / "frame_000002.png")

    class FakeClient:
        calls = 0

        def request(self, request, *, error_prefix, transport_trace=None):
            self.calls += 1
            return ModelResponse({"choices": [{"message": {"content": json.dumps({
                "person_visible": True,
                "target_candidate": "person",
                "confidence": "high",
                "evidence_text": "visible candidate",
            })}}]})

    client = FakeClient()
    factory_calls = []
    monkeypatch.setattr(
        "evaluation.benchmarks.uav_flow.visual_diagnostics.create_model_client",
        lambda settings: factory_calls.append(settings) or client,
    )
    result = write_visual_diagnostics(
        make_case(),
        run_dir=run_dir,
        case_dir=case_dir,
        enabled=True,
        model_config=ModelConfig(model_name="test", api_key="secret"),
    )

    assert len(factory_calls) == 1
    assert client.calls == 2
    assert result["judged_count"] == 2
