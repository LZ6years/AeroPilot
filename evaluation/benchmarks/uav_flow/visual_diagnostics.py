"""Visual grounding diagnostics for UAV-Flow benchmark case reviews.

This module is benchmark-only. It never exposes evaluation truth such as target
positions to the planner. In frame-only mode it writes a stable RGB inventory;
when enabled, it asks the configured multimodal model for a small JSON judgment
about whether a person/target candidate is visible in each sampled frame.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from src.agent.config import ModelConfig


SCHEMA_VERSION = "uav_flow_visual_diagnostics_v1"
EVALUATION_TRUTH_TERMS = ("target_pos", "reference_path_preprocessed", "end_pos")


def write_visual_diagnostics(
    case: UavFlowCase,
    *,
    run_dir: str | Path,
    case_dir: str | Path,
    enabled: bool,
    model_config: ModelConfig,
    max_frames: int = 12,
) -> dict[str, Any]:
    """Write visual diagnostics JSON and return a compact summary."""

    run_path = Path(run_dir)
    output_path = Path(case_dir) / "visual_diagnostics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _sample_frames(sorted((run_path / "artifacts" / "rgb").glob("*.png")), max_frames)
    records = [
        _diagnose_frame(case, frame, index, enabled=enabled, model_config=model_config)
        for index, frame in enumerate(frames, start=1)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "class_name": case.class_name,
        "instruction": case.selected_instruction,
        "enabled": enabled,
        "model": model_config.model_name if enabled else None,
        "frame_count": len(frames),
        "records": records,
        "truth_leak_check": _truth_leak_check(records),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = _summary(payload)
    return {**summary, "path": str(output_path)}


def _diagnose_frame(
    case: UavFlowCase,
    frame: Path,
    index: int,
    *,
    enabled: bool,
    model_config: ModelConfig,
) -> dict[str, Any]:
    base = {
        "frame_index": index,
        "path": str(frame),
        "person_visible": None,
        "target_candidate": None,
        "confidence": "unknown",
        "evidence_text": "frame listed; model visual diagnostic disabled",
        "status": "frame_only",
    }
    if not enabled:
        return base
    if not model_config.api_key:
        return {**base, "status": "skipped", "evidence_text": "missing model api key"}
    try:
        judged = _call_model(case, frame, model_config)
    except Exception as exc:
        return {
            **base,
            "status": "model_error",
            "error_type": type(exc).__name__,
            "evidence_text": str(exc),
        }
    return {**base, **_normalize_model_result(judged), "status": "judged"}


def _call_model(case: UavFlowCase, frame: Path, config: ModelConfig) -> dict[str, Any]:
    payload = _payload(case, frame, config)
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
        with urllib.request.urlopen(request, timeout=float(config.timeout_sec)) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}; body={body[:500]}") from exc
    content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return _parse_json_object(str(content))


def _payload(case: UavFlowCase, frame: Path, config: ModelConfig) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a UAV-Flow visual grounding diagnostic. "
                    "Only inspect the image and instruction. Return JSON with: "
                    "person_visible boolean|null, target_candidate string|null, "
                    "confidence high|medium|low|unknown, evidence_text concise Chinese. "
                    "Do not suggest actions or mention hidden coordinates."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "instruction": case.selected_instruction,
                                "class_name": case.class_name,
                                "question": "Is the instruction-relevant person or visual target visible in this frame?",
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": _data_url(frame)}},
                ],
            },
        ],
    }


def _normalize_model_result(value: dict[str, Any]) -> dict[str, Any]:
    confidence = str(value.get("confidence", "unknown")).lower().strip()
    if confidence not in {"high", "medium", "low", "unknown"}:
        confidence = "unknown"
    person_visible = value.get("person_visible")
    if not isinstance(person_visible, bool):
        person_visible = None
    target_candidate = value.get("target_candidate")
    if target_candidate is not None:
        target_candidate = str(target_candidate).strip() or None
    evidence = str(value.get("evidence_text", "")).strip()
    return {
        "person_visible": person_visible,
        "target_candidate": target_candidate,
        "confidence": confidence,
        "evidence_text": evidence or "model returned no evidence text",
    }


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    judged = [item for item in records if item.get("status") == "judged"]
    visible = [item for item in judged if item.get("person_visible") is True]
    return {
        "schema_version": payload.get("schema_version"),
        "status": "ok",
        "enabled": payload.get("enabled"),
        "frame_count": payload.get("frame_count", 0),
        "judged_count": len(judged),
        "person_visible_count": len(visible),
        "truth_leak_check": payload.get("truth_leak_check", {}),
    }


def _truth_leak_check(records: list[dict[str, Any]]) -> dict[str, Any]:
    text = json.dumps(records, ensure_ascii=False)
    leaked = [term for term in EVALUATION_TRUTH_TERMS if term in text]
    return {"status": "failed" if leaked else "passed", "leaked_terms": leaked}


def _sample_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[-1]]
    step = (len(frames) - 1) / float(max_frames - 1)
    return [frames[round(index * step)] for index in range(max_frames)]


def _data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("visual diagnostic model did not return a JSON object")
    return value
