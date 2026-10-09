"""Contract tests for the frozen five-model, two-stage benchmark matrix."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.config import ModelConfig
from tests.manual import model_matrix_benchmark_probe as matrix


def _manifest() -> dict:
    base = ModelConfig(
        model_name=matrix.MODEL_ORDER[0],
        api_key="unused-in-redacted-summary",
        base_url=matrix.BASE_URL,
        stream=True,
        timeout_sec=120,
    )
    profiles = {
        model: replace(base, model_name=model).redacted_summary()
        for model in matrix.MODEL_ORDER
    }
    cases = [
        {
            "benchmark": "aerial" if index < 5 else "travel",
            "case_id": f"case_{index}",
            "difficulty": "low" if index < 5 else "easy",
            "scene": str(index),
        }
        for index in range(10)
    ]
    return {
        "complete_source_hashes": {},
        "source_inputs": {"admission": {"path": "unused"}},
        "dataset_sha256": {"aerial": "a", "travel": "b"},
        "model_order": list(matrix.MODEL_ORDER),
        "initial_profiles": profiles,
        "pilot": {
            "benchmark": "aerial",
            "case_id": "pilot",
            "difficulty": "low",
            "scene": "17",
        },
        "cases": cases,
    }


def test_model_order_keeps_gpt_56_sol_last_and_has_fifty_formal_cases():
    assert matrix.MODEL_ORDER == (
        "gpt-6.1-sol",
        "gpt-5.6-terra",
        "gpt-6-sol",
        "qwen3.8-flash",
        "gpt-5.6-sol",
    )
    assert len(matrix.AERIAL_CASE_IDS) == 5
    assert len(matrix.TRAVEL_CASE_IDS) == 5
    assert len(matrix.MODEL_ORDER) * (
        len(matrix.AERIAL_CASE_IDS) + len(matrix.TRAVEL_CASE_IDS)
    ) == 50
    assert matrix.PILOT_CASE_ID not in {
        *matrix.AERIAL_CASE_IDS,
        *matrix.TRAVEL_CASE_IDS,
    }


def test_child_environment_is_exact_and_secret_is_not_in_summary(monkeypatch):
    monkeypatch.setenv("FLYAGENT_MODEL_API_KEY", "sk-test-secret-value")
    config = matrix._profile(
        ModelConfig(api_key="sk-test-secret-value"), "qwen3.8-flash"
    )
    env = matrix._profile_environment(config, {"UNCHANGED": "1"})
    assert env["FLYAGENT_MODEL_NAME"] == "qwen3.8-flash"
    assert env["FLYAGENT_MODEL_BASE_URL"] == matrix.BASE_URL
    assert env["FLYAGENT_MODEL_API_KEY"] == "sk-test-secret-value"
    assert env["FLYAGENT_MODEL_API_PROTOCOL"] == "responses"
    assert env["FLYAGENT_MODEL_REASONING_EFFORT"] == "low"
    assert env["FLYAGENT_MODEL_STREAM"] == "false"
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "sk-test-secret-value" not in json.dumps(config.redacted_summary())
    assert matrix._redact("bad sk-test-secret-value", api_key=config.api_key) == (
        "bad ***REDACTED***"
    )
    assert matrix._redact("task-specific route") == "task-specific route"


def test_stage1_summary_requires_all_five_models(tmp_path):
    manifest = _manifest()
    for model in matrix.MODEL_ORDER[:-1]:
        path = matrix._stage1_model_dir(tmp_path, model) / "status.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"model": model, "status": "qualified", "qualified": True}),
            encoding="utf-8",
        )
    summary = matrix.refresh_stage1_summary(tmp_path, manifest)
    assert summary["stage2_ready"] is False
    assert summary["formal_case_count"] == 50
    last = matrix._stage1_model_dir(tmp_path, matrix.MODEL_ORDER[-1]) / "status.json"
    last.parent.mkdir(parents=True)
    last.write_text(
        json.dumps(
            {
                "model": matrix.MODEL_ORDER[-1],
                "status": "qualified",
                "qualified": True,
            }
        ),
        encoding="utf-8",
    )
    assert matrix.refresh_stage1_summary(tmp_path, manifest)["stage2_ready"] is True


def test_stage2_refuses_incomplete_stage1(tmp_path, monkeypatch):
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(matrix, "_guard", lambda *_args: None)
    with pytest.raises(ValueError, match="all five Stage 1 models"):
        matrix.stage2(tmp_path)


def test_stage1_runs_requested_model_only_and_preserves_order(tmp_path, monkeypatch):
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(matrix, "_guard", lambda *_args: None)
    called = []

    def fake(root, frozen, model):
        called.append(model)
        return {"model": model, "status": "qualified", "qualified": True}

    monkeypatch.setattr(matrix, "run_stage1_model", fake)
    monkeypatch.setattr(
        matrix,
        "refresh_stage1_summary",
        lambda *_args: {"stage2_ready": False},
    )
    matrix.stage1(tmp_path, only_model="gpt-6-sol")
    assert called == ["gpt-6-sol"]


def test_recovery_keeps_original_and_allows_only_one_attempt(tmp_path):
    item = {"benchmark": "aerial", "case_id": "case", "scene": "1"}
    original = matrix.batch.case_dir(tmp_path, 0, item)
    original.mkdir(parents=True)
    (original / "status.json").write_text(
        json.dumps(
            {
                "status": "infrastructure_error",
                "cleanup_verified": True,
            }
        ),
        encoding="utf-8",
    )
    recovery = matrix._next_case_directory(tmp_path, 0, item, resume=True)
    assert recovery is not None and recovery.name.endswith("_recovery_01")
    recovery.mkdir()
    (recovery / "status.json").write_text(
        json.dumps(
            {
                "status": "infrastructure_error",
                "cleanup_verified": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Only one"):
        matrix._next_case_directory(tmp_path, 0, item, resume=True)


def test_stream_fallback_requires_transport_evidence():
    assert matrix._suggests_stream_mismatch(
        {"error": "invalid_response while decoding SSE event-stream"}
    )
    assert not matrix._suggests_stream_mismatch(
        {"error": "model has no available distributor"}
    )
    assert not matrix._suggests_stream_mismatch(
        {
            "config": {"stream": True},
            "error": "function tools with reasoning_effort are unsupported",
        }
    )


def test_stage1_stops_after_unknown_pilot_cleanup(tmp_path, monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(matrix, "_guard", lambda *_args: None)
    monkeypatch.setenv("FLYAGENT_MODEL_API_KEY", "sk-test-process-secret")
    config = matrix._profile(
        ModelConfig(api_key="sk-test-process-secret"), matrix.MODEL_ORDER[0]
    )
    monkeypatch.setattr(
        matrix,
        "_admit_transport",
        lambda *_args: (config, []),
    )
    monkeypatch.setattr(
        matrix,
        "_admit_compactor",
        lambda *_args: (config, []),
    )
    monkeypatch.setattr(
        matrix,
        "_run_pilot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("pilot crashed")),
    )
    monkeypatch.setattr(matrix, "refresh_stage1_summary", lambda *_args: {})
    result = matrix.run_stage1_model(tmp_path, manifest, matrix.MODEL_ORDER[0])
    assert result["status"] == "cleanup_failed"
    assert result["blocking_reason"] == "pilot_orchestrator_exception"


def test_profile_rehydration_uses_process_secret(monkeypatch):
    monkeypatch.setenv("FLYAGENT_MODEL_API_KEY", "sk-current-process-secret")
    original = matrix._profile(
        ModelConfig(api_key="sk-current-process-secret"), "gpt-5.6-sol"
    )
    restored = matrix._profile_from_summary(original.redacted_summary())
    assert restored.api_key == "sk-current-process-secret"
    assert restored.redacted_summary() == original.redacted_summary()
