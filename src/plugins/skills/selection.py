"""Model-selected skill validation and optional model selector implementation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from src.agent.semantic.task_parser import _loads_json_object
from src.plugins.skills.catalog import SkillCatalog
from src.runtime.configs.model_config import ModelConfig, load_model_config


SKILL_SELECTION_SYSTEM_PROMPT = """You select UAV-Agent knowledge skills.

You receive a JSON context with the current instruction, semantic phase, and an
available_skills list. Select zero or more skill names that should be loaded
before action planning. Choose only the exact "name" values from
available_skills, not plugin ids or qualified names.

Select the smallest useful set. Most instructions need 0-2 skills; use 3 only
when the instruction clearly combines three different concerns. Never select all
skills as a safe default. Returning [] is valid for simple primitive motion or
hover instructions.

Select only skills that directly help the current instruction and phase. Do not
select safety guidance unless the instruction or scene context mentions people,
obstacles, narrow space, collision risk, or unsafe uncertainty. Do not select
completion-evidence guidance unless the task is explicitly about deciding whether
to finish, confirming success, or avoiding premature completion.

If the instruction is only a primitive command such as "hover", "rotate N
degrees", or "move forward N meters" and it does not mention a target, safety
condition, search uncertainty, or completion decision, return [].

If the instruction says find, search, locate, look for, or "somewhere around" a
target whose current visibility is unknown, prefer search_under_uncertainty over
approach or completion skills. Return exactly:
{"selected_skills": ["skill_name"]}.
"""


@dataclass(frozen=True)
class SkillSelectionResult:
    requested_skill_names: list[str] = field(default_factory=list)
    accepted_skill_names: list[str] = field(default_factory=list)
    rejected_skill_names: list[str] = field(default_factory=list)
    status: str = "not_requested"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "model_selected",
            "status": self.status,
            "requested_skill_names": list(self.requested_skill_names),
            "accepted_skill_names": list(self.accepted_skill_names),
            "rejected_skill_names": list(self.rejected_skill_names),
            "warnings": list(self.warnings),
        }


class ModelSkillSelector:
    """Ask the configured model which enabled knowledge skills to load."""

    def __init__(self, model_config: ModelConfig | None = None, *, enabled: bool = True) -> None:
        self.model_config = model_config or load_model_config()
        self.enabled = bool(enabled)

    def select(self, selection_context: dict[str, Any], catalog: SkillCatalog) -> SkillSelectionResult:
        if not self.enabled:
            return SkillSelectionResult(status="disabled")
        if not catalog.skills:
            return SkillSelectionResult(status="empty_catalog")
        if not self.model_config.api_key:
            return SkillSelectionResult(status="skipped_no_api_key", warnings=["skill_selection_skipped:no_api_key"])
        try:
            response = self._call_model(selection_context)
        except Exception as exc:  # pragma: no cover - network failure path is integration-level.
            return SkillSelectionResult(status="model_error", warnings=[f"skill_selection_failed:{str(exc)[:240]}"])
        response_warnings = [str(item) for item in response.get("_warnings", []) if str(item).strip()]
        raw_names = response.get("selected_skills", [])
        if not isinstance(raw_names, list):
            raw_names = []
        requested = [str(item).strip() for item in raw_names if str(item).strip()]
        accepted, rejected = catalog.validate_selection(requested)
        return SkillSelectionResult(
            requested_skill_names=requested,
            accepted_skill_names=[skill.name for skill in accepted],
            rejected_skill_names=rejected,
            status="selected" if accepted else "none_selected",
            warnings=response_warnings + [f"skill_selection_rejected:{name}" for name in rejected],
        )

    def _call_model(self, selection_context: dict[str, Any]) -> dict[str, Any]:
        first_content = self._request_content(selection_context)
        try:
            return _loads_json_object(first_content)
        except Exception as first_exc:
            repair_context = {
                **dict(selection_context),
                "format_repair": (
                    "The previous response was not valid JSON. Return one compact JSON object only, "
                    "with no explanation and no Markdown."
                ),
                "previous_error": str(first_exc)[:240],
            }
            second_content = self._request_content(repair_context)
            try:
                return _loads_json_object(second_content)
            except Exception as second_exc:
                fallback_names = _skill_names_from_response_text(
                    f"{first_content}\n{second_content}",
                    selection_context,
                )
                return {
                    "selected_skills": fallback_names,
                    "_warnings": [
                        (
                            "skill_selection_invalid_json_recovered:"
                            if fallback_names
                            else "skill_selection_invalid_json_fallback:"
                        )
                        + f"{str(first_exc)[:100]} | {str(second_exc)[:100]}"
                    ],
                }

    def _request_content(self, selection_context: dict[str, Any]) -> str:
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": SKILL_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(selection_context, ensure_ascii=False)},
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.model_config.chat_completions_url(),
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
            raise RuntimeError(f"Skill selection request failed: HTTP {exc.code} {exc.reason}; body={body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Skill selection request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]


def _skill_names_from_response_text(text: str, selection_context: dict[str, Any]) -> list[str]:
    available = selection_context.get("available_skills", [])
    names = [
        str(item.get("name", "")).strip()
        for item in available
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]
    selected = []
    seen = set()
    for name in names:
        if name in text and name not in seen:
            selected.append(name)
            seen.add(name)
    if len(names) > 3 and len(selected) == len(names):
        return []
    return selected
