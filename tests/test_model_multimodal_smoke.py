"""Unit checks for the manual Qwen multimodal smoke-test helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from src.agent.config import ModelConfig


def load_smoke_module():
    path = Path("tests") / "manual" / "model_multimodal_smoke.py"
    spec = importlib.util.spec_from_file_location("model_multimodal_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_multimodal_smoke_config_uses_existing_key_with_qwen_override(tmp_path, monkeypatch):
    smoke = load_smoke_module()
    for key in [
        "FLYAGENT_MODEL_API_KEY",
        "FLYAGENT_MODEL_BASE_URL",
        "FLYAGENT_MULTIMODAL_MODEL_NAME",
        "FLYAGENT_MULTIMODAL_MODEL_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "FLYAGENT_MODEL_API_KEY=secret-key",
                "FLYAGENT_MODEL_BASE_URL=https://text-model.example/v1",
                "FLYAGENT_MULTIMODAL_MODEL_NAME=qwen3-vl-plus",
                "FLYAGENT_MULTIMODAL_MODEL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
            ]
        ),
        encoding="utf-8",
    )

    config = smoke.build_multimodal_config(env_path)
    summary = smoke.redacted_config_summary(config)

    assert config.api_key == "secret-key"
    assert config.model_name == "qwen3-vl-plus"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert summary["has_api_key"] is True
    assert "secret-key" not in json.dumps(summary)


def test_multimodal_smoke_payload_uses_encoded_image_without_leaking_secret_or_duplicate_data(tmp_path):
    smoke = load_smoke_module()
    context = smoke.build_probe_context(tmp_path)
    assert "data:image" not in json.dumps(context)
    config = ModelConfig(
        model_name="qwen3-vl-plus",
        api_key="secret-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    payload = smoke.build_payload(config, context)
    redacted = smoke.redacted_payload_summary(payload, config)

    user_content = payload["messages"][1]["content"]
    assert isinstance(user_content, list)
    image_data = user_content[1]["image_url"]["url"]
    assert image_data.startswith("data:image/png;base64,")
    assert user_content[1] == {"type": "image_url", "image_url": {"url": image_data}}
    assert image_data not in user_content[0]["text"]
    assert "secret-key" not in json.dumps(redacted)
    assert "data:image/png;base64," not in json.dumps(redacted)
    assert redacted["payload"]["has_image_part"] is True
