"""Model-backed semantic plan patch generation.

The updater produces a limited patch for the current semantic phase. It does
not execute UAV tools, read evaluator truth, or rewrite the full semantic plan.
Runtime code applies the patch only after version and phase checks pass.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.semantic.task_parser import _loads_json_object
from src.agent.semantic.validator import SemanticPlanValidationError, validate_semantic_plan_patch
from src.runtime.configs.model_config import ModelConfig, load_model_config


class SemanticPlanUpdaterConfigError(RuntimeError):
    """Raised when semantic plan updating is requested without usable config."""


class SemanticPlanUpdater:
    """Generate a constrained semantic patch with the configured model."""

    def __init__(self, model_config: ModelConfig | None = None, prompt_path: str | None = None) -> None:
        self.model_config = model_config or load_model_config()
        self.system_prompt = self._load_system_prompt(prompt_path)

    def update(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.model_config.api_key:
            raise SemanticPlanUpdaterConfigError("Semantic plan updater requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise SemanticPlanUpdaterConfigError(f"Unsupported model provider: {self.model_config.provider}")
        response = self._call_openai_compatible(context)
        patch = validate_semantic_plan_patch(response)
        return validate_semantic_plan_patch_numeric_provenance(
            patch,
            instruction=str(context.get("user_instruction", "")),
            current_phase=context.get("current_phase"),
        )

    def _call_openai_compatible(self, context: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Semantic plan updater request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Semantic plan updater request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _load_system_prompt(self, prompt_path: str | None) -> str:
        path = Path(prompt_path) if prompt_path else Path(__file__).resolve().parent / "prompts" / "semantic_plan_update_system.txt"
        return path.read_text(encoding="utf-8").strip()


_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_PHASE_NUMERIC_SOURCE_FIELDS = ("goal", "done_when", "continue_when", "metrics")


def validate_semantic_plan_patch_numeric_provenance(
    patch: dict[str, Any],
    *,
    instruction: str,
    current_phase: Any,
) -> dict[str, Any]:
    """Reject semantic-update numbers absent from user or current contract.

    This check belongs to the updater boundary: it does not infer what a number
    means, but it prevents an asynchronous repair from inventing new thresholds.
    """

    allowed_numbers = _numbers_in_value(instruction)
    if isinstance(current_phase, dict):
        for field in _PHASE_NUMERIC_SOURCE_FIELDS:
            allowed_numbers.extend(_numbers_in_value(current_phase.get(field)))
    patch_numbers = _numbers_in_value(patch)
    unsupported = [number for number in patch_numbers if not _contains_abs(allowed_numbers, number)]
    if unsupported:
        unique = list(dict.fromkeys(unsupported))
        raise SemanticPlanValidationError(
            f"Semantic plan update introduced unsupported numeric value(s): {unique}"
        )
    return patch


def _numbers_in_value(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers_in_value(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers_in_value(item)]
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(match.group(0)) for match in _NUMBER_PATTERN.finditer(str(value))]


def _contains_abs(values: list[float], target: float) -> bool:
    return any(abs(abs(value) - abs(target)) <= 1e-6 for value in values)
