"""Model-backed semantic plan patch generation.

The updater produces a limited patch for the current semantic phase. It does
not execute UAV tools, read evaluator truth, or rewrite the full semantic plan.
Runtime code applies the patch only after version and phase checks pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agent.provider_http import request_openai_compatible_json
from src.agent.semantic.task_parser import (
    DEFAULT_AGENT_RULES_PATH,
    DEFAULT_DONE_WHEN_RULES_PATH,
    DEFAULT_USER_PREFERENCES_PATH,
    _configured_evidence_message,
    _loads_json_object,
    _normalize_configured_evidence,
)
from src.agent.semantic.validator import SemanticPlanValidationError, validate_semantic_plan_patch
from src.runtime.configs.model_config import ModelConfig


class SemanticPlanUpdaterConfigError(RuntimeError):
    """Raised when semantic plan updating is requested without usable config."""


class SemanticPlanUpdater:
    """Generate a constrained semantic patch with the configured model."""

    def __init__(
        self,
        model_config: ModelConfig,
        prompt_path: str | None = None,
        agent_rules_path: str | None = None,
    ) -> None:
        # Model configuration belongs to the application composition root.
        # Reading .env here would make background updates depend on cwd, which
        # differs for Git worktrees and remote runners.
        self.model_config = model_config
        self.agent_rules = self._load_agent_rules(agent_rules_path)
        self.user_preferences = self._load_required_text(
            DEFAULT_USER_PREFERENCES_PATH,
            label="FlyAgent user language preferences",
        )
        self.done_when_rules = self._load_required_text(
            DEFAULT_DONE_WHEN_RULES_PATH,
            label="FlyAgent done_when generation rules",
        )
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
        payload = {
            "model": self.model_config.model_name,
            "messages": self._build_messages(context),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data = request_openai_compatible_json(
            self.model_config,
            payload,
            error_prefix="Semantic plan updater request failed",
        )
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _build_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        update_context = dict(context)
        selected_guidance = update_context.pop("selected_skill_guidance", [])
        configured_evidence = _normalize_configured_evidence(
            update_context.pop("configured_evidence", [])
        )
        messages = [
            {"role": "system", "content": self.agent_rules},
            {"role": "system", "content": self.user_preferences},
        ]
        if isinstance(selected_guidance, list) and selected_guidance:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Selected knowledge from the triggering planning window follows. "
                        "Treat it as domain guidance, not observation, tools, or output schema.\n\n"
                        + json.dumps(selected_guidance, ensure_ascii=False)
                    ),
                }
            )
        if configured_evidence:
            messages.append(
                {
                    "role": "system",
                    "content": _configured_evidence_message(configured_evidence),
                }
            )
        messages.append(
            {"role": "system", "content": self.done_when_rules}
        )
        messages.extend(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(update_context, ensure_ascii=False)},
            ]
        )
        return messages

    def _load_system_prompt(self, prompt_path: str | None) -> str:
        path = Path(prompt_path) if prompt_path else Path(__file__).resolve().parent / "prompts" / "semantic_plan_update_system.txt"
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_agent_rules(agent_rules_path: str | None) -> str:
        path = Path(agent_rules_path) if agent_rules_path else DEFAULT_AGENT_RULES_PATH
        return SemanticPlanUpdater._load_required_text(path, label="FlyAgent rules")

    @staticmethod
    def _load_required_text(path: Path, *, label: str) -> str:
        if not path.exists():
            raise SemanticPlanUpdaterConfigError(f"{label} file not found: {path}")
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise SemanticPlanUpdaterConfigError(f"{label} file is empty: {path}")
        return text


_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
_PHASE_NUMERIC_SOURCE_FIELDS = ("goal", "done_when", "continue_when")


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
