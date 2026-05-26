"""Manual Qwen multimodal smoke test for provider image inputs.

This probe verifies the real end-to-end path:

    generated image -> encoded model_inputs data_url
    -> OpenAI-compatible image_url message -> Qwen multimodal JSON response

It intentionally lives under tests/manual and is not collected by pytest. It
uses the existing FlyAgent model API key, but lets the multimodal model and
base URL be overridden without changing the main planner configuration.

PowerShell examples:

    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/model_multimodal_smoke.py --env-path .env

    $env:FLYAGENT_MULTIMODAL_MODEL_NAME='qwen3-vl-plus'
    $env:FLYAGENT_MULTIMODAL_MODEL_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
    python tests/manual/model_multimodal_smoke.py
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.planning.model_inputs import build_model_user_content
from src.runtime.configs.model_config import ModelConfig, load_model_config


DEFAULT_MULTIMODAL_MODEL = "qwen3-vl-plus"
DEFAULT_MULTIMODAL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real Qwen multimodal smoke test through FlyAgent model_inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--keep-run-dir", action="store_true", help="Keep the temporary run directory for inspection.")
    return parser


def build_multimodal_config(env_path: str | Path = ".env") -> ModelConfig:
    base_config = load_model_config(env_path)
    env_values = _merged_env_values(Path(env_path))
    model_name = env_values.get("FLYAGENT_MULTIMODAL_MODEL_NAME") or DEFAULT_MULTIMODAL_MODEL
    base_url = (
        env_values.get("FLYAGENT_MULTIMODAL_MODEL_BASE_URL")
        or base_config.base_url
        or DEFAULT_MULTIMODAL_BASE_URL
    )
    return replace(
        base_config,
        provider="openai_compatible",
        model_name=model_name,
        base_url=base_url,
    )


def build_probe_context(run_dir: Path) -> dict[str, Any]:
    image_path = run_dir / "artifacts" / "rgb" / "qwen_multimodal_smoke.png"
    write_probe_image(image_path)
    context = {
        "active_instruction": "请判断测试图中是否有红色方块和蓝色圆点。",
        "current_phase": {"name": "manual_multimodal_smoke"},
        "semantic_goal": {"target": "red square and blue dot smoke test"},
        "fragments": [
            {
                "name": "manual_probe_image",
                "content": {"source_image": str(image_path)},
                "model_inputs": [_encoded_image_model_input(image_path)],
            }
        ],
        "metadata": {"run_dir": str(run_dir), "probe": "model_multimodal_smoke"},
    }
    _assert_encoded_image_input(context)
    return context


def write_probe_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 25, 70, 75), fill=(255, 0, 0))
    draw.ellipse((105, 40, 135, 70), fill=(0, 0, 255))
    draw.text((18, 92), "red square + blue dot", fill=(0, 0, 0))
    image.save(path)


def build_payload(config: ModelConfig, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict visual smoke-test judge. Return exactly one JSON object "
                    "with booleans sees_red_square and sees_blue_dot plus a short description."
                ),
            },
            {"role": "user", "content": build_model_user_content(context)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def call_openai_compatible_json(config: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        config.chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen multimodal smoke request failed: HTTP {exc.code} {exc.reason}; body={body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen multimodal smoke request failed: {exc}") from exc
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def redacted_config_summary(config: ModelConfig) -> dict[str, Any]:
    summary = config.redacted_summary()
    summary["chat_completions_url"] = config.chat_completions_url()
    return summary


def redacted_payload_summary(payload: dict[str, Any], config: ModelConfig) -> dict[str, Any]:
    user_content = payload["messages"][1]["content"]
    parts_count = len(user_content) if isinstance(user_content, list) else 1
    has_image_part = isinstance(user_content, list) and any(item.get("type") == "image_url" for item in user_content)
    return {
        "api_key_present": bool(config.api_key),
        "model_config": redacted_config_summary(config),
        "payload": {
            "model": payload.get("model"),
            "messages_count": len(payload.get("messages", [])),
            "user_content_parts_count": parts_count,
            "has_image_part": has_image_part,
            "response_format": payload.get("response_format"),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    config = build_multimodal_config(args.env_path)
    if not config.api_key:
        raise SystemExit("Missing FLYAGENT_MODEL_API_KEY in environment or .env; API key will not be printed.")

    if args.keep_run_dir:
        run_dir = Path.cwd() / "runs" / "manual_multimodal_smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        _run_probe(config, run_dir)
        print(f"run_dir={run_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="flyagent_multimodal_smoke_") as tmp:
        _run_probe(config, Path(tmp))


def _run_probe(config: ModelConfig, run_dir: Path) -> None:
    context = build_probe_context(run_dir)
    payload = build_payload(config, context)
    print("Loaded multimodal model config:")
    print(json.dumps(redacted_config_summary(config), indent=2, ensure_ascii=False))
    print("\nPayload summary:")
    print(json.dumps(redacted_payload_summary(payload, config), indent=2, ensure_ascii=False))
    result = call_openai_compatible_json(config, payload)
    print("\nRaw JSON result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sees_red_square = bool(result.get("sees_red_square"))
    sees_blue_dot = bool(result.get("sees_blue_dot"))
    if not sees_red_square or not sees_blue_dot:
        raise SystemExit("multimodal_smoke: failed; model did not confirm both visual markers.")
    print("\nmultimodal_smoke: ok")
    print(f"model={config.model_name}")
    print("image_input=data_url")
    print(f"sees_red_square={str(sees_red_square).lower()}")
    print(f"sees_blue_dot={str(sees_blue_dot).lower()}")


def _assert_encoded_image_input(context: dict[str, Any]) -> None:
    model_inputs = context["fragments"][0]["model_inputs"]
    assert model_inputs, "custom image module did not produce model_inputs"
    first = model_inputs[0]
    assert first["type"] == "image"
    assert first["encoding"] == "data_url"
    assert first["data"].startswith("data:image/png;base64,")


def _encoded_image_model_input(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "role": "latest_observation",
        "mime_type": mime_type,
        "encoding": "data_url",
        "data": f"data:{mime_type};base64,{encoded}",
        "metadata": {"source_path": str(path), "modality": "rgb", "processing": "manual_smoke_probe"},
    }


def _merged_env_values(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = _strip_env_value(value.strip())
    values.update({key: value for key, value in os.environ.items() if key.startswith("FLYAGENT_")})
    return values


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    main()
