"""Manual Qwen visual-description probe for real-engine RGB fixtures.

This probe is a bridge toward a future custom image module mode that captions
images before planner use. It sends each fixture through the current
MultimodalImageContextModule model_inputs[data_url] path, asks Qwen-VL for a
structured visual_description_v1 object, and writes reviewable JSON/Markdown
logs under tests/manual_outputs.

PowerShell:

    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/model_visual_description_probe.py --env-path .env
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.multimodal_image.context_modules import (
    BANNED_ACTION_TERMS,
    DEFAULT_VISUAL_DESCRIPTION_MODEL,
    SCHEMA_VERSION,
    STRATEGY,
    MultimodalImageContextModule,
    build_visual_caption_payload,
    call_visual_caption_model,
    caption_model_config,
    normalize_visual_description as normalize_module_visual_description,
)
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.runtime.configs.model_config import ModelConfig


DEFAULT_IMAGE_DIR = REPO_ROOT / "tests" / "fixtures" / "visual_description_images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "manual_outputs" / "visual_description_real_engine"
DEFAULT_INSTRUCTION = "寻找建筑入口。"
DEFAULT_PHASE_NAME = "search_visual_target"
DEFAULT_PHASE_MEANING = "寻找建筑入口、门廊、入口标识或建筑底层开口。"
ALLOWED_POSITIONS = {
    "foreground-left",
    "foreground-center",
    "foreground-right",
    "midground-left",
    "midground-center",
    "midground-right",
    "background-left",
    "background-center",
    "background-right",
    "unknown",
}
ALLOWED_ROLES = {"target_candidate", "landmark", "boundary", "free_space", "vehicle", "obstacle", "unknown"}
BANNED_ACTION_TERMS = [
    "turn_left",
    "turn_right",
    "move_forward",
    "recommended_next_observation",
    "左转",
    "右转",
    "前进",
    "靠近",
    "飞向",
    "悬停",
    "适合",
    "飞行",
    "机动",
    "通行",
    "绕行",
    "扫描",
    "航路规划",
    "路径方向",
]
VEHICLE_TERMS = {"车", "车辆", "校车", "轿车", "货车", "厢式车", "SUV", "van", "bus", "car", "truck"}
ENTRANCE_TARGET_TERMS = {"入口", "出入口", "门", "门廊", "门洞", "开口", "入口标识", "建筑底层", "entry", "entrance", "door"}
SOFT_ACTION_REPLACEMENTS = {
    "可通行": "可见空隙",
    "通行使用": "实际使用",
    "通行路径": "可见空隙",
    "通行条件良好": "可见空隙明显",
    "通行条件": "可见空隙",
    "通行设施": "入口附属结构",
    "通行空间": "可见空隙",
    "通行/绕行": "空间间隔",
    "绕行路径": "空间间隔",
    "路径方向": "空间延伸线索",
    "航路规划": "空间布局判断",
    "适合低空悬停": "画面中无遮挡",
    "适合悬停": "画面中无遮挡",
    "适合飞行机动": "画面中无遮挡",
    "适合飞行器空间变化": "画面中可见开阔区域",
    "适合": "呈现为",
    "飞行或移动观察": "开阔可见",
    "飞行或观察": "开阔可见",
    "无人机开阔可见或空间变化扫描": "开阔可见",
    "空间变化扫描": "空间开阔",
    "移动观察": "视觉观察",
    "低空悬停": "开阔可见",
    "飞行机动": "空间开阔",
    "机动观察": "继续作为视觉参照",
    "机动": "空间变化",
    "进一步扫描": "当前画面未确认",
    "进一步观察": "当前画面未确认",
    "需扫描": "当前画面未确认",
    "需要扫描": "当前画面未确认",
    "应扫描": "当前画面未确认",
    "扫描": "视觉确认",
    "需进一步": "当前画面未确认",
    "靠近": "邻近",
    "飞向": "朝向",
    "通行": "可见空隙",
    "飞行": "空间",
    "悬停": "开阔可见",
    "绕行": "空间间隔",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe real-engine RGB fixtures with Qwen-VL via FlyAgent model_inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--phase-name", default=DEFAULT_PHASE_NAME)
    parser.add_argument("--phase-meaning", default=DEFAULT_PHASE_MEANING)
    parser.add_argument("--write-redacted-payloads", action="store_true")
    return parser


def build_visual_description_config(env_path: str | Path = ".env") -> ModelConfig:
    return caption_model_config(env_path=env_path)


def fixture_images(image_dir: str | Path = DEFAULT_IMAGE_DIR, limit: int = 5) -> list[Path]:
    images = sorted(Path(image_dir).glob("*.png"))
    return images[: max(0, int(limit))]


def build_description_context(
    image_path: Path,
    case_run_dir: Path,
    case_index: int,
    *,
    instruction: str = DEFAULT_INSTRUCTION,
    phase_name: str = DEFAULT_PHASE_NAME,
    phase_meaning: str = DEFAULT_PHASE_MEANING,
    model_config: ModelConfig | None = None,
    caption_client=None,
) -> dict[str, Any]:
    writer = MemoryEventWriter(case_run_dir)
    reader = MemoryRecordReader(case_run_dir)
    memory = FlightMemory("visual_description_probe", "real_engine_fixture", image_path.stem, "uav_1", memory_reader=reader)
    observation_write = writer.write_observation(
        {
            "source": "manual_visual_description_probe",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(image_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])
    stage_context = StageContextInput(
        active_instruction=instruction,
        current_phase={"name": phase_name, "meaning": phase_meaning},
        phase_completion_criteria=[
            {
                "type": "visual_goal",
                "description": phase_meaning,
                "evidence_required": "phase-relevant visual facts only",
            }
        ],
    )
    fragment = MultimodalImageContextModule(
        model_config=model_config,
        caption_client=caption_client,
    ).build(
        MemorySourceBundle(memory),
        stage_context,
    ).to_dict()
    _assert_visual_caption_input(fragment)
    return {
        "active_instruction": instruction,
        "current_phase": {"name": phase_name, "meaning": phase_meaning},
        "phase_completion_criteria": [
            {
                "type": "visual_goal",
                "description": phase_meaning,
                "evidence_required": "phase-relevant visual facts only",
            }
        ],
        "semantic_goal": {"target": "phase_visual_summary_v1 probe"},
        "fragments": [fragment],
        "metadata": {"source_image": str(image_path), "case_index": case_index},
    }


def build_description_payload(config: ModelConfig, context: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return build_visual_caption_payload(
        config,
        image_path,
        StageContextInput(
            active_instruction=str(context.get("active_instruction", "")),
            current_phase=context.get("current_phase") if isinstance(context.get("current_phase"), dict) else None,
            phase_completion_criteria=[
                dict(item) for item in context.get("phase_completion_criteria", []) if isinstance(item, dict)
            ],
        ),
    )


def call_description_model(config: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Call the configured multimodal model and preserve latency and token usage."""

    return call_visual_caption_model(config, payload)


def normalize_visual_description(raw: dict[str, Any], image_path: Path) -> dict[str, Any]:
    return normalize_module_visual_description(raw, image_path)


def assess_descriptions(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [item["visual_description"].get("scene_summary", "") for item in results]
    cue_counts = [len(item["visual_description"].get("phase_relevant_cues", [])) for item in results]
    object_counts = [len(item["visual_description"].get("relevant_objects", [])) for item in results]
    relevant_objects = [
        obj
        for item in results
        for obj in item["visual_description"].get("relevant_objects", [])
        if isinstance(obj, dict)
    ]
    unknown_position_count = sum(1 for obj in relevant_objects if obj.get("position") == "unknown")
    durations = [float(item.get("call_metrics", {}).get("duration_sec", 0)) for item in results]
    token_usage = _aggregate_usage([item.get("call_metrics", {}).get("usage", {}) for item in results])
    prompt_tokens = token_usage.get("prompt_tokens") if isinstance(token_usage.get("prompt_tokens"), int) else None
    completion_tokens = (
        token_usage.get("completion_tokens") if isinstance(token_usage.get("completion_tokens"), int) else None
    )
    target_candidate_misuse_count = sum(
        _target_candidate_misuse_count(item["visual_description"]) for item in results
    )
    soft_action_phrase_count = sum(_soft_action_phrase_count(item["visual_description"]) for item in results)
    english_summary_count = sum(_english_layout_text_count(item["visual_description"]) for item in results)
    return {
        "overall": (
            "The probe summarizes phase-relevant visual facts instead of generic scene descriptions. "
            "Relevant objects include positions and roles so planner can use them as visual context."
        ),
        "strengths": [
            "Relevant objects carry position, role, and phase relevance.",
            "Free-space and landmark fields are separated from general object lists.",
            "Latency and token usage are recorded per image and in aggregate.",
        ],
        "risks": [
            "Single-frame phase relevance is still visual evidence, not an action decision.",
            "Object position is approximate and should be fused with pose/history later.",
            "Obstacle role only means visible occupancy, not collision safety.",
        ],
        "result_count": len(results),
        "average_relevant_object_count": sum(object_counts) / len(object_counts) if object_counts else 0,
        "average_phase_cue_count": sum(cue_counts) / len(cue_counts) if cue_counts else 0,
        "unknown_position_count": unknown_position_count,
        "unknown_position_rate": (
            round(unknown_position_count / len(relevant_objects), 3) if relevant_objects else 0
        ),
        "total_duration_sec": round(sum(durations), 3),
        "average_duration_sec": round(sum(durations) / len(durations), 3) if durations else 0,
        "average_prompt_tokens": round(prompt_tokens / len(results), 3) if prompt_tokens is not None and results else None,
        "average_completion_tokens": (
            round(completion_tokens / len(results), 3) if completion_tokens is not None and results else None
        ),
        "token_usage": token_usage,
        "target_candidate_misuse_count": target_candidate_misuse_count,
        "soft_action_phrase_count": soft_action_phrase_count,
        "english_layout_text_count": english_summary_count,
        "format_consistency": {
            "schema": SCHEMA_VERSION,
            "all_relevant_objects_valid": all(
                all(_is_valid_relevant_object(obj) for obj in item["visual_description"].get("relevant_objects", []))
                for item in results
            ),
            "all_list_fields_are_strings": all(_has_string_lists(item["visual_description"]) for item in results),
            "no_target_candidate_misuse": target_candidate_misuse_count == 0,
            "no_soft_action_phrases": soft_action_phrase_count == 0,
            "all_layout_text_is_chinese": english_summary_count == 0,
            "no_action_suggestion_terms": all(
                not _contains_action_suggestion(item["visual_description"]) for item in results
            ),
        },
        "sample_summaries": summaries[:2],
    }


def write_outputs(
    output_dir: Path,
    config: ModelConfig,
    results: list[dict[str, Any]],
    assessment: dict[str, Any],
    *,
    console_lines: list[str],
    redacted_payloads: list[dict[str, Any]],
    write_redacted_payloads: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = output_dir / "redacted_payloads"
    if write_redacted_payloads:
        payload_dir.mkdir(parents=True, exist_ok=True)
        for item in redacted_payloads:
            name = Path(item["source_image"]).stem + ".json"
            (payload_dir / name).write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
    json_path = output_dir / "qwen_visual_descriptions.json"
    md_path = output_dir / "qwen_visual_descriptions.md"
    console_path = output_dir / "console.txt"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "strategy": STRATEGY,
                "model": config.model_name,
                "base_url": config.base_url,
                "assessment": assessment,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(config, results, assessment), encoding="utf-8")
    console_path.write_text("\n".join(console_lines) + "\n", encoding="utf-8")
    paths = {"json": json_path, "markdown": md_path, "console": console_path}
    if write_redacted_payloads:
        paths["redacted_payloads"] = payload_dir
    return paths


def main() -> None:
    args = build_parser().parse_args()
    config = build_visual_description_config(args.env_path)
    if not config.api_key:
        raise SystemExit("Missing FLYAGENT_MODEL_API_KEY in environment or .env; API key will not be printed.")
    image_paths = fixture_images(args.image_dir, args.limit)
    if not image_paths:
        raise SystemExit(f"No PNG images found in {args.image_dir}")
    output_dir = Path(args.output_dir)
    work_dir = output_dir / "memory_cases"
    console_lines = [
        f"model={config.model_name}",
        f"image_dir={Path(args.image_dir)}",
        f"output_dir={output_dir}",
        f"instruction={args.instruction}",
        f"phase={args.phase_name}: {args.phase_meaning}",
    ]
    results: list[dict[str, Any]] = []
    redacted_payloads: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        context = build_description_context(
            image_path,
            work_dir / image_path.stem,
            index,
            instruction=args.instruction,
            phase_name=args.phase_name,
            phase_meaning=args.phase_meaning,
            model_config=config,
        )
        fragment = context["fragments"][0]
        description = fragment["content"]["visual_description"]
        call_metrics = _caption_call_metrics(fragment)
        results.append(
            {
                "image_name": image_path.name,
                "source_image": str(image_path),
                "model_input": _model_input_summary(context),
                "call_metrics": call_metrics,
                "visual_description": description,
            }
        )
        redacted_payloads.append(_redacted_caption_summary(fragment, config, image_path))
        usage = call_metrics.get("usage", {})
        line = (
            f"[{index}/{len(image_paths)}] {image_path.name}: "
            f"{call_metrics.get('duration_sec')}s, tokens={usage.get('total_tokens')}; "
            f"{description['scene_summary']}"
        )
        console_lines.append(line)
        print(line)
    assessment = assess_descriptions(results)
    paths = write_outputs(
        output_dir,
        config,
        results,
        assessment,
        console_lines=console_lines,
        redacted_payloads=redacted_payloads,
        write_redacted_payloads=args.write_redacted_payloads,
    )
    print("\nAssessment:")
    print(json.dumps(assessment, indent=2, ensure_ascii=False))
    print("\nWrote outputs:")
    for key, path in paths.items():
        print(f"{key}={path}")


def _model_input_summary(context: dict[str, Any]) -> dict[str, Any]:
    model_input = context["fragments"][0]["model_inputs"][0]
    return {
        "type": model_input.get("type"),
        "role": model_input.get("role"),
        "text_chars": len(model_input.get("text", "")),
        "metadata": dict(model_input.get("metadata", {})) if isinstance(model_input.get("metadata"), dict) else {},
    }


def _caption_call_metrics(fragment: dict[str, Any]) -> dict[str, Any]:
    processing = fragment["content"].get("visual_processing", {})
    metrics = processing.get("caption_metrics", {}) if isinstance(processing, dict) else {}
    usage = metrics.get("usage", {}) if isinstance(metrics, dict) else {}
    return {
        "duration_sec": metrics.get("duration_sec") if isinstance(metrics, dict) else None,
        "usage": usage if isinstance(usage, dict) else {},
    }


def _redacted_caption_summary(fragment: dict[str, Any], config: ModelConfig, image_path: Path) -> dict[str, Any]:
    processing = fragment["content"].get("visual_processing", {})
    return {
        "source_image": str(image_path),
        "model": config.model_name,
        "base_url": config.base_url,
        "has_api_key": bool(config.api_key),
        "visual_processing": processing,
        "has_visual_description": "visual_description" in fragment["content"],
        "model_input_count": len(fragment.get("model_inputs", [])),
    }


def _markdown_report(config: ModelConfig, results: list[dict[str, Any]], assessment: dict[str, Any]) -> str:
    lines = [
        "# Qwen Visual Descriptions",
        "",
        f"Model: `{config.model_name}`",
        f"Schema: `{SCHEMA_VERSION}`",
        "",
        "## Assessment",
        "",
        assessment["overall"],
        "",
        "Strengths:",
        *[f"- {item}" for item in assessment["strengths"]],
        "",
        "Risks:",
        *[f"- {item}" for item in assessment["risks"]],
        "",
    ]
    for item in results:
        desc = item["visual_description"]
        image_src = Path(item["source_image"]).resolve().as_posix()
        lines.extend(
            [
                f"## {item['image_name']}",
                "",
                f"![{item['image_name']}]({image_src})",
                "",
                f"- phase_relevance: {desc['phase_relevance']}",
                f"- scene_summary: {desc['scene_summary']}",
                f"- spatial_overview: {desc['spatial_overview']}",
                f"- relevant_objects: {json.dumps(desc['relevant_objects'], ensure_ascii=False)}",
                f"- free_space: {json.dumps(desc['free_space'], ensure_ascii=False)}",
                f"- landmarks: {json.dumps(desc['landmarks'], ensure_ascii=False)}",
                f"- phase_relevant_cues: {json.dumps(desc['phase_relevant_cues'], ensure_ascii=False)}",
                f"- uncertainty: {json.dumps(desc['uncertainty'], ensure_ascii=False)}",
                f"- confidence: {desc['confidence']}",
                f"- call_metrics: {json.dumps(item.get('call_metrics', {}), ensure_ascii=False)}",
                "",
            ]
        )
    return "\n".join(lines)


def _assert_visual_caption_input(fragment: dict[str, Any]) -> None:
    model_inputs = fragment["model_inputs"]
    assert model_inputs, "custom image module did not produce visual caption model_inputs"
    first = model_inputs[0]
    assert first["type"] == "text"
    assert first["role"] == "visual_caption"
    assert "visual_description" in first["text"]
    assert "visual_description" in fragment["content"]


def _list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_stringify_list_item(item) for item in value if _stringify_list_item(item)]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _relevant_objects(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [_relevant_object(item) for item in value if item]


def _relevant_object(item: Any) -> dict[str, str]:
    if isinstance(item, dict):
        name = _text_field(item.get("name") or item.get("class") or item.get("type") or item.get("label") or "unknown")
        position = _position(item.get("position"))
        role = _role(item.get("role"))
        phase_relevance = _text_field(item.get("phase_relevance") or item.get("relevance") or item.get("why_relevant"))
        description = _text_field(item.get("description") or item.get("text") or item.get("value"))
        confidence = _confidence(item.get("confidence"))
    else:
        name = _text_field(item)
        position = "unknown"
        role = "unknown"
        phase_relevance = "模型未提供阶段相关性。"
        description = name
        confidence = "medium"
    return {
        "name": name or "unknown",
        "position": position,
        "role": role,
        "phase_relevance": phase_relevance or "与当前阶段可能相关，但模型未说明原因。",
        "description": description or name or "unknown",
        "confidence": confidence,
    }


def _position(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace("_", "-")
    return text if text in ALLOWED_POSITIONS else "unknown"


def _role(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in ALLOWED_ROLES else "unknown"


def _text_field(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, list):
                item_text = "、".join(str(part) for part in item if str(part).strip())
            else:
                item_text = str(item).strip()
            if item_text:
                parts.append(f"{key}: {item_text}")
        return _sanitize_soft_action_text("；".join(parts))
    if isinstance(value, list):
        return _sanitize_soft_action_text("；".join(str(item).strip() for item in value if str(item).strip()))
    return _sanitize_soft_action_text(str(value or "").strip())


def _stringify_list_item(item: Any) -> str:
    if isinstance(item, dict):
        label = item.get("class") or item.get("cue_type") or item.get("type") or item.get("name")
        description = item.get("description") or item.get("value") or item.get("text")
        if label and description:
            return _sanitize_soft_action_text(f"{label}: {description}")
        if description:
            return _sanitize_soft_action_text(str(description))
        return _sanitize_soft_action_text(json.dumps(item, ensure_ascii=False))
    return _sanitize_soft_action_text(str(item).strip())


def _sanitize_soft_action_text(text: str) -> str:
    sanitized = text
    for old, new in SOFT_ACTION_REPLACEMENTS.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def _sanitize_visual_description(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_visual_description(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_visual_description(item) for item in value]
    if isinstance(value, str):
        return _sanitize_soft_action_text(value)
    return value


def _confidence(value: Any) -> str:
    text = str(value or "medium").lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _usage_summary(value: Any) -> dict[str, Any]:
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


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _aggregate_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": _sum_optional(item.get("prompt_tokens") for item in usages),
        "completion_tokens": _sum_optional(item.get("completion_tokens") for item in usages),
        "total_tokens": _sum_optional(item.get("total_tokens") for item in usages),
    }


def _sum_optional(values) -> int | None:
    total = 0
    found = False
    for value in values:
        if isinstance(value, int):
            total += value
            found = True
    return total if found else None


def _has_string_lists(description: dict[str, Any]) -> bool:
    for key in ["free_space", "landmarks", "phase_relevant_cues", "uncertainty"]:
        value = description.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return False
    return True


def _is_valid_relevant_object(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"name", "position", "role", "phase_relevance", "description", "confidence"}
    return (
        required.issubset(value)
        and value.get("position") in ALLOWED_POSITIONS
        and value.get("role") in ALLOWED_ROLES
        and value.get("confidence") in {"high", "medium", "low"}
    )


def _contains_action_suggestion(description: dict[str, Any]) -> bool:
    return _soft_action_phrase_count(description) > 0


def _soft_action_phrase_count(description: dict[str, Any]) -> int:
    text = json.dumps(description, ensure_ascii=False).lower()
    return sum(1 for term in BANNED_ACTION_TERMS if term.lower() in text)


def _target_candidate_misuse_count(description: dict[str, Any]) -> int:
    phase_text = json.dumps(description.get("phase", {}), ensure_ascii=False).lower()
    phase_targets_entrance = _contains_any(phase_text, ENTRANCE_TARGET_TERMS)
    phase_targets_vehicle = _contains_any(phase_text, VEHICLE_TERMS)
    if not phase_targets_entrance and not phase_targets_vehicle:
        return 0

    count = 0
    for obj in description.get("relevant_objects", []):
        if not isinstance(obj, dict) or obj.get("role") != "target_candidate":
            continue
        obj_text = json.dumps(obj, ensure_ascii=False).lower()
        if phase_targets_entrance and not _contains_any(obj_text, ENTRANCE_TARGET_TERMS):
            count += 1
        elif phase_targets_vehicle and not _contains_any(obj_text, VEHICLE_TERMS):
            count += 1
    return count


def _english_layout_text_count(description: dict[str, Any]) -> int:
    fields = [description.get("scene_summary", ""), description.get("spatial_overview", "")]
    return sum(1 for value in fields if _looks_like_english_sentence(str(value)))


def _looks_like_english_sentence(text: str) -> bool:
    cleaned = text
    for position in ALLOWED_POSITIONS:
        cleaned = cleaned.replace(position, "")
    letters = sum(1 for char in cleaned if char.isascii() and char.isalpha())
    return letters >= 12


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _load_multimodal_smoke_module():
    path = REPO_ROOT / "tests" / "manual" / "model_multimodal_smoke.py"
    spec = importlib.util.spec_from_file_location("model_multimodal_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
