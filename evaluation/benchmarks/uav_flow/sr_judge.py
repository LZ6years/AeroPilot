"""Optional LLM-assisted semantic success judge for UAV-Flow evaluation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from src.agent.config import ModelConfig, load_model_config

LOW_NDTW_FAILURE_THRESHOLD = 0.05
ROTATION_CLASSES = {"Turn", "Rotate"}


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
    metric_assessment = metric_success_assessment(case, trajectory_summary, ndtw_result)
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
    return _apply_metric_guard(result, metric_assessment)


def _payload(case: UavFlowCase, trajectory_summary: dict[str, Any], ndtw_result: dict[str, Any], config: ModelConfig) -> dict[str, Any]:
    metric_assessment = metric_success_assessment(case, trajectory_summary, ndtw_result)
    user = {
        "case_id": case.case_id,
        "class": case.class_name,
        "instruction": case.selected_instruction,
        "evaluation_target_pos": case.target_pos,
        "trajectory_summary": trajectory_summary,
        "ndtw": ndtw_result,
        "metric_success_assessment": metric_assessment,
    }
    return {
        "model": config.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You judge UAV task success from benchmark evidence. "
                    "nDTW is a normalized trajectory alignment metric where 1 is best and 0 is worst. "
                    "Never describe a very low nDTW score as high-fidelity alignment. "
                    "For Turn/Rotate, nDTW uses rotation-sequence alignment with xyz position zeroed; "
                    "for Move/Approach, position and rotation both affect nDTW. "
                    "Treat metric_success_assessment as a hard constraint: if it says success=false, "
                    "you must also return success=false and explain the metric/action conflict. "
                    "Return JSON with success, confidence, reason, failure_type, needs_human_review. "
                    "Do not invent observations not present in the evidence."
                ),
            },
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def metric_success_assessment(
    case: UavFlowCase,
    trajectory_summary: dict[str, Any],
    ndtw_result: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic metric gate that the LLM SR explanation cannot override."""

    score = ndtw_result.get("ndtw")
    class_name = str(case.class_name or "")
    metric_mode = (ndtw_result.get("metric_interpretation") or {}).get("mode")
    base = {
        "class_name": class_name or None,
        "metric_mode": metric_mode,
        "ndtw": score,
        "thresholds": {"low_ndtw_failure": LOW_NDTW_FAILURE_THRESHOLD},
    }
    if ndtw_result.get("status") != "ok" or not isinstance(score, (int, float)):
        return {
            **base,
            "success": None,
            "reason": "nDTW is unavailable, so metric success cannot be decided deterministically.",
        }
    if float(score) < LOW_NDTW_FAILURE_THRESHOLD:
        mode_note = (
            "rotation sequence alignment failed"
            if class_name in ROTATION_CLASSES
            else "position/rotation trajectory alignment failed"
        )
        return {
            **base,
            "success": False,
            "failure_type": "low_ndtw",
            "reason": f"nDTW={float(score):.6g} is below {LOW_NDTW_FAILURE_THRESHOLD}; {mode_note}.",
        }
    return {
        **base,
        "success": None,
        "reason": "nDTW is not in the hard-failure band; semantic SR may inspect the rest of the evidence.",
    }


def _apply_metric_guard(model_result: dict[str, Any], metric_assessment: dict[str, Any]) -> dict[str, Any]:
    result = dict(model_result) if isinstance(model_result, dict) else {"raw_content": model_result}
    metric_success = metric_assessment.get("success")
    llm_success = result.get("success") if isinstance(result.get("success"), bool) else None
    if metric_success is False and llm_success is True:
        reason = str(result.get("reason", "")).strip()
        metric_reason = str(metric_assessment.get("reason", "")).strip()
        return {
            **result,
            "status": "conflict",
            "success": False,
            "llm_success": llm_success,
            "confidence": min(_float_confidence(result.get("confidence")), 0.2),
            "failure_type": "metric_gate_conflict",
            "needs_human_review": True,
            "metric_success_assessment": metric_assessment,
            "reason": f"Metric hard gate overrode the LLM SR result. {metric_reason} LLM reason: {reason}",
        }
    if metric_success is False:
        return {
            **result,
            "status": "ok",
            "success": False,
            "metric_success_assessment": metric_assessment,
            "needs_human_review": bool(result.get("needs_human_review", False)),
        }
    return {**result, "status": "ok", "metric_success_assessment": metric_assessment}


def _float_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
