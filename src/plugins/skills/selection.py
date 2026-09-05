"""Model-selected skill validation and optional model selector implementation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from src.agent.semantic.task_parser import _loads_json_object
from src.plugins.skills.catalog import SkillCatalog
from src.runtime.configs.model_config import ModelConfig


SKILL_SELECTION_SYSTEM_PROMPT = """You select UAV-Agent knowledge skills.

You receive a JSON context with the current instruction, semantic phase, and an
available_skills list. Select zero or more skill names that should be loaded
before action planning. Choose only the exact "name" values from
available_skills.

Read the active_instruction, semantic_goal, current_phase, phase_progress_brief,
and each available skill's name, description, summary, and reference manifest.
Select the minimal useful set whose descriptions and summaries directly match
the active task concerns. Never select all skills as a safe default.

Selection process:
1. Identify the primary concern of the current semantic phase.
2. Select at most one primary skill whose description directly covers that
   concern.
3. Add an auxiliary skill only when it covers a separate concern that is active
   now, not merely generally useful.
4. Request reference files only when the selected skill summary is not enough
   for the current concern. Reference paths must come from the selected skill's
   references list. Do not request references as insurance.

Default to 0-1 skills. Select 2 only when two independent concerns are both
active. Select 3 only for clearly complex phases with three independent current
concerns. If several skills seem relevant, prefer the smallest set that prevents
missing important guidance.

Select skills for the current semantic phase only. Use current_phase.goal,
current_phase.done_when, and current_phase.continue_when
as the primary basis. Use active_instruction and semantic_goal only to understand
the target, user intent, and final relation; do not preload skills for future
phases whose concern is not active in the current phase.

Treat side/front/back relations as modifiers of the primary task unless the
current phase itself makes the side relation part of route traversal. A task
"on/from the left/right/front side" still primarily needs the skill that covers
the current phase's main concern; do not switch to a route or passage skill
unless traversal is the active current concern.

Do not add search guidance just because a landmark exists; add it only when the
target, route cue, or candidate is missing or ambiguous. Do not add visual
attention guidance just because an object is visible; add it when facing,
centering, observing, or visual confirmation is the main concern. Do not add
obstacle or safety guidance just because a route exists; add it when the current
route is narrow, cluttered, occluded, close to people, close to obstacles, or
otherwise risk-sensitive.

Reference request boundary:
- Selecting a skill does not mean its references are needed.
- Request a search or uncertainty reference only when the current concern is
  absent target evidence, weak candidate evidence, hidden target evidence, or
  an explicit search/find/locate/identify-candidate task whose target visibility
  has not yet been confirmed.
- Do not request a search reference for ordinary face, center, keep-in-view,
  observe, confirm-visible, or decide-if-current-observation-is-sufficient tasks
  when the selected skill's summary and main guidance are enough.

If the instruction is only a primitive command such as "hover", "rotate N
degrees", or "translate N meters" and no available skill description adds
task-relevant knowledge beyond the primitive, return []. If an available skill
description clearly covers an important current-phase nuance beyond the
primitive itself, select that skill.

Return exactly:
{"selected_skills": ["skill_name"], "requested_references": {"skill_name": ["references/file.md"]}}.
Omit requested_references or return {} when no reference is needed.
"""


@dataclass(frozen=True)
class SkillSelectionResult:
    requested_skill_names: list[str] = field(default_factory=list)
    accepted_skill_names: list[str] = field(default_factory=list)
    rejected_skill_names: list[str] = field(default_factory=list)
    requested_references: dict[str, list[str]] = field(default_factory=dict)
    accepted_references: dict[str, list[str]] = field(default_factory=dict)
    rejected_references: dict[str, list[str]] = field(default_factory=dict)
    status: str = "not_requested"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "model_selected",
            "status": self.status,
            "requested_skill_names": list(self.requested_skill_names),
            "accepted_skill_names": list(self.accepted_skill_names),
            "rejected_skill_names": list(self.rejected_skill_names),
            "requested_references": _copy_reference_map(self.requested_references),
            "accepted_references": _copy_reference_map(self.accepted_references),
            "rejected_references": _copy_reference_map(self.rejected_references),
            "warnings": list(self.warnings),
        }


class ModelSkillSelector:
    """Ask the configured model which enabled knowledge skills to load."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        enabled: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        if self.enabled and model_config is None:
            raise ValueError(
                "Enabled ModelSkillSelector requires an explicit ModelConfig; "
                "load it once at the application entry point"
            )
        # A disabled selector is a deliberate no-I/O dependency used by
        # rule-based runtimes and tests.  It must not search the current working
        # directory for a .env file as a hidden side effect.
        self.model_config = model_config or ModelConfig()
        self.system_prompt = system_prompt or SKILL_SELECTION_SYSTEM_PROMPT

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
        requested_references = _requested_references_from_response(response)
        accepted_references, rejected_references = catalog.validate_reference_selection(accepted, requested_references)
        reference_warnings = [
            f"skill_reference_rejected:{skill_name}:{path}"
            for skill_name, paths in rejected_references.items()
            for path in paths
        ]
        return SkillSelectionResult(
            requested_skill_names=requested,
            accepted_skill_names=[skill.name for skill in accepted],
            rejected_skill_names=rejected,
            requested_references=requested_references,
            accepted_references=accepted_references,
            rejected_references=rejected_references,
            status="selected" if accepted else "none_selected",
            warnings=response_warnings
            + [f"skill_selection_rejected:{name}" for name in rejected]
            + reference_warnings,
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
                {"role": "system", "content": self.system_prompt},
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


def _requested_references_from_response(response: dict[str, Any]) -> dict[str, list[str]]:
    raw = response.get("requested_references", {})
    if not isinstance(raw, dict):
        return {}
    requested: dict[str, list[str]] = {}
    for raw_skill_name, raw_paths in raw.items():
        skill_name = str(raw_skill_name).strip()
        if not skill_name:
            continue
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, list):
            paths = raw_paths
        else:
            paths = []
        normalized = []
        seen = set()
        for raw_path in paths:
            path = str(raw_path).strip().replace("\\", "/")
            if path and path not in seen:
                normalized.append(path)
                seen.add(path)
        if normalized:
            requested[skill_name] = normalized
    return requested


def _copy_reference_map(data: dict[str, list[str]]) -> dict[str, list[str]]:
    return {str(key): list(value) for key, value in data.items()}
