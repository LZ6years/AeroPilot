"""Optional LLM-assisted semantic success judge for UAV-Flow evaluation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from src.runtime.configs.model_config import ModelConfig, load_model_config


def judge_success(
    case: UavFlowCase,
    trajectory_summary: dict[str, Any],
    ndtw_result: dict[str, Any],
    *,
    enabled: bool = False,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Return a semantic SR judgment or a skipped record."""

    if not enabled:
        return {"status": "skipped", "reason": "sr_judge_disabled"}
    config = model_config or load_model_config()
    if not config.api_key:
        return {"status": "skipped", "reason": "missing_model_api_key"}
    payload = _payload(case, trajectory_summary, ndtw_result, config)
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
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    content = data["choices"][0]["message"]["content"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"raw_content": content}
    return {"status": "ok", **result}


def _payload(case: UavFlowCase, trajectory_summary: dict[str, Any], ndtw_result: dict[str, Any], config: ModelConfig) -> dict[str, Any]:
    user = {
        "case_id": case.case_id,
        "class": case.class_name,
        "instruction": case.selected_instruction,
        "evaluation_target_pos": case.target_pos,
        "trajectory_summary": trajectory_summary,
        "ndtw": ndtw_result,
    }
    return {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You judge UAV task success from benchmark evidence. "
                    "Return JSON with success, confidence, reason, failure_type, needs_human_review. "
                    "Do not invent observations not present in the evidence."
                ),
            },
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

