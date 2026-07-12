"""Model-backed semantic plan patch generation.

The updater produces a limited patch for the current semantic phase. It does
not execute UAV tools, read evaluator truth, or rewrite the full semantic plan.
Runtime code applies the patch only after version and phase checks pass.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.semantic.task_parser import _loads_json_object
from src.agent.semantic.validator import validate_semantic_plan_patch
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
        return validate_semantic_plan_patch(response)

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
