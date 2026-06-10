"""Model configuration and OpenAI-compatible caption calls."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from src.memory.context.schemas import StageContextInput
from src.runtime.configs.model_config import ModelConfig, load_model_config

from .constants import (
    DEFAULT_VISUAL_DESCRIPTION_MODEL,
    MAX_CAPTION_TOKENS,
    SCHEMA_VERSION,
    STRATEGY,
)

CaptionClient = Callable[[ModelConfig, dict[str, Any]], dict[str, Any]]


def caption_model_config(base_config: ModelConfig | None = None, env_path: str | Path = ".env") -> ModelConfig:
    """Load the current model config, with optional visual-caption overrides."""

    config = base_config or load_model_config(env_path)
    env = _merged_flyagent_env(Path(env_path))
    return replace(
        config,
        model_name=env.get("FLYAGENT_VISUAL_DESCRIPTION_MODEL_NAME") or config.model_name or DEFAULT_VISUAL_DESCRIPTION_MODEL,
        base_url=env.get("FLYAGENT_VISUAL_DESCRIPTION_MODEL_BASE_URL") or config.base_url,
    )


def build_visual_caption_payload(
    config: ModelConfig,
    image_path: Path,
    stage_context: StageContextInput,
    frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible multimodal request for phase-aware captioning."""

    current_phase = dict(stage_context.current_phase) if stage_context.current_phase else None
    criteria = current_phase.get("completion_criteria", []) if isinstance(current_phase, dict) else []
    user_context = {
        "active_instruction": stage_context.active_instruction,
        "current_phase": current_phase,
        "phase_completion_criteria": [dict(item) for item in criteria if isinstance(item, dict)],
        "expected_visual_description": {
            "schema_version": SCHEMA_VERSION,
            "strategy": STRATEGY,
        },
        "image_metadata": caption_image_metadata(image_path, frame),
    }
    return {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": _caption_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(user_context, ensure_ascii=False)},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": MAX_CAPTION_TOKENS,
        "response_format": {"type": "json_object"},
    }


def call_visual_caption_model(config: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the configured multimodal model and return parsed JSON plus metrics."""

    request = urllib.request.Request(
        config.chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            raw_response = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"visual caption request failed: HTTP {exc.code} {exc.reason}; body={body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"visual caption request failed: {exc}") from exc
    content = raw_response["choices"][0]["message"]["content"]
    return {
        "raw_response": raw_response,
        "parsed": json.loads(content),
        "duration_sec": round(time.perf_counter() - started, 3),
        "usage": usage_summary(raw_response.get("usage")),
    }


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def caption_image_metadata(path: Path | None, frame: dict[str, Any] | None) -> dict[str, Any]:
    metadata = dict(frame or {})
    return {
        "source_path": str(path) if path else metadata.get("path"),
        "artifact_ref": metadata.get("artifact_ref") or (str(path) if path else None),
        "modality": "rgb",
        "processing": "phase_aware_caption",
        "mock": bool(metadata.get("mock", False)),
    }


def usage_summary(value: Any) -> dict[str, Any]:
    usage = dict(value) if isinstance(value, dict) else {}
    prompt_tokens = _first_int(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _first_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _first_int(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": usage,
    }


def _caption_system_prompt() -> str:
    return (
        "你是无人机 phase 视觉摘要器，只输出 JSON。所有自然语言必须中文。只写视觉事实，不写动作。"
        "visual_description 字段: schema_version=phase_visual_summary_v1, strategy=qwen_vl_phase_caption_v3, "
        "phase, phase_relevance, scene_summary, spatial_overview, relevant_objects, free_space, "
        "landmarks, phase_relevant_cues, uncertainty, confidence。"
        "relevant_objects 最多4个，每个含 name, position, role, phase_relevance, description, confidence；"
        "position 必须选九宫格: foreground-left, foreground-center, foreground-right, "
        "midground-left, midground-center, midground-right, background-left, background-center, "
        "background-right；只有完全看不出位置才用 unknown，不要默认 unknown；"
        "role 只取 target_candidate, landmark, boundary, free_space, vehicle, obstacle, unknown。"
        "找建筑入口时，target_candidate 只给入口、门廊、门洞、入口标识、建筑底层开口；车辆用 vehicle。"
        "free_space/landmarks/phase_relevant_cues/uncertainty 各最多2条短句。"
        "禁写英文描述、天气、建筑用途、无关物体、turn_left、move_forward、recommended_next_observation、"
        "左转、右转、前进、靠近、飞向、悬停、机动、通行、绕行、扫描。"
    )


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _merged_flyagent_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().startswith("FLYAGENT_"):
                values[key.strip()] = _strip_env_value(value.strip())
    values.update({key: value for key, value in os.environ.items() if key.startswith("FLYAGENT_")})
    return values


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
