"""Parse user instructions into lightweight semantic task plans."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agent.semantic.parser_contracts import (
    SemanticPlanStructure,
    merge_semantic_plan,
    validate_phase_criteria,
)
from src.agent.semantic.schemas import SemanticPlan, TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict, validate_semantic_plan
from src.runtime.configs.model_config import ModelConfig, load_model_config


DEFAULT_AGENT_RULES_PATH = Path(__file__).resolve().parents[3] / "memory" / "FLYAGENT.md"
DEFAULT_USER_PREFERENCES_PATH = Path(__file__).resolve().parents[3] / "memory" / "USER.md"
DEFAULT_DONE_WHEN_RULES_PATH = Path(__file__).resolve().parent / "prompts" / "done_when_generation_system.txt"
MAX_CONFIGURED_EVIDENCE_ITEMS = 8
MAX_CONFIGURED_EVIDENCE_CHARS = 160


class TaskParserConfigError(RuntimeError):
    """Raised when model parsing is requested without usable model config."""


class TaskParser:
    def __init__(
        self,
        mode: str | None = None,
        model_config: ModelConfig | None = None,
        *,
        phase_skill_resolver: Any | None = None,
        protocol_context: str | None = None,
    ) -> None:
        self.model_config = model_config or load_model_config()
        self.mode = mode or self.model_config.task_parser_mode
        self.config_warnings: list[str] = []
        self.last_parse_trace: dict[str, Any] = {}
        self.configured_evidence: list[str] = []
        # Benchmark protocols may clarify how user-facing terminal verbs map
        # onto an episode without changing the original instruction or the
        # shared semantic-plan schema.
        self.protocol_context = (protocol_context or "").strip()
        if self.mode == "model":
            self.agent_rules = self._load_agent_rules()
            self.user_preferences = self._load_required_text(
                DEFAULT_USER_PREFERENCES_PATH,
                label="FlyAgent user language preferences",
            )
            self.done_when_rules = self._load_required_text(
                DEFAULT_DONE_WHEN_RULES_PATH,
                label="FlyAgent done_when generation rules",
            )
            self.structure_prompt = self._load_structure_prompt()
            self.criteria_prompt = self._load_criteria_prompt()
            self.phase_skill_resolver = phase_skill_resolver or self._default_phase_skill_resolver()
        else:
            self.agent_rules = ""
            self.user_preferences = ""
            self.done_when_rules = ""
            self.structure_prompt = ""
            self.criteria_prompt = ""
            self.phase_skill_resolver = None

    def configure_evidence_capabilities(self, capabilities: list[str] | None) -> None:
        """Configure potential runtime evidence used only by criteria generation."""

        self.configured_evidence = _normalize_configured_evidence(capabilities)

    def parse(self, instruction: str) -> TaskParseResult:
        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise ValueError("instruction must not be empty")
        if self.mode == "rule_based":
            self.last_parse_trace = {"strategy": "rule_based", "phases": [], "warnings": []}
            plan = self._parse_rule_based(clean_instruction)
        elif self.mode == "model":
            plan = self._parse_with_model(clean_instruction)
        else:
            raise ValueError(f"Unsupported task parser mode: {self.mode}")
        return TaskParseResult(clean_instruction, self.mode, validate_semantic_plan(plan))

    def _parse_rule_based(self, instruction: str) -> SemanticPlan:
        if self._is_observe_and_clarify(instruction):
            return semantic_plan_from_dict(
                {
                    "intent": "observe_and_clarify",
                    "task_goal": {
                        "target": "right-side area target",
                        "target_attributes": {},
                        "final_state": "uav_hovering_or_waiting_for_clarification",
                    },
                    "phases": [
                        _phase(
                            "orient_right",
                            "orient the view toward the right-side area",
                            ["right-side area is in the forward view"],
                            ["right-side area is not yet in view"],
                        ),
                        _phase(
                            "observe_area",
                            "observe the right-side area for possible targets",
                            ["the area has been observed with current visual evidence"],
                            ["the observation is missing or unusable"],
                        ),
                        _phase(
                            "clarify_if_uncertain",
                            "hold safely if the target remains ambiguous",
                            ["uncertainty is acknowledged and the UAV is stable"],
                            ["the UAV is not stable enough to wait"],
                        ),
                    ],
                }
            )
        if self._contains_any(instruction, ("入口", "门口")):
            return semantic_plan_from_dict(
                {
                    "intent": "navigate_by_visual_landmark",
                    "task_goal": {
                        "target": "building entrance",
                        "target_attributes": {"category": "building_entrance"},
                        "final_state": "uav_hovering_in_front_of_target",
                    },
                    "phases": [
                        _phase(
                            "orient_left",
                            "orient toward the left-side search area",
                            ["left-side area is visible in the latest observation"],
                            ["left-side area is not yet in view"],
                        ),
                        _phase(
                            "search_landmark",
                            "identify the building entrance candidate",
                            ["building entrance or a strong entrance candidate is visible"],
                            ["entrance identity is unclear"],
                        ),
                        _phase(
                            "approach_landmark",
                            "move closer to the entrance front area",
                            ["entrance remains identifiable", "UAV has made meaningful approach progress"],
                            ["entrance identity is unclear", "movement has not made approach progress"],
                        ),
                        _phase(
                            "stop",
                            "hold in front of the entrance",
                            ["UAV is stable near the entrance front area"],
                            ["UAV is not stable or not plausibly in front of the entrance"],
                        ),
                    ],
                }
            )
        if self._contains_any(instruction, ("车", "车辆", "汽车")):
            return semantic_plan_from_dict(
                {
                    "intent": "navigate_by_visual_landmark",
                    "task_goal": {
                        "target": "red vehicle",
                        "target_attributes": {"color": "red", "category": "vehicle"},
                        "final_state": "uav_near_target_with_safe_distance",
                    },
                    "phases": [
                        _phase(
                            "search_landmark",
                            "identify the red vehicle candidate",
                            ["red vehicle or strong vehicle candidate is visible"],
                            ["vehicle identity is unclear"],
                        ),
                        _phase(
                            "approach_landmark",
                            "move closer to the vehicle while preserving safe standoff",
                            ["vehicle remains identifiable", "UAV is closer while keeping a safe distance"],
                            ["vehicle identity is unclear", "safe approach relation is not yet plausible"],
                        ),
                        _phase(
                            "stop",
                            "hold near the vehicle with safe distance",
                            ["UAV is stable near the vehicle with safe distance"],
                            ["UAV is not stable or safe distance is unclear"],
                        ),
                    ],
                }
            )
        if self._contains_any(instruction, ("路灯", "灯杆")):
            return semantic_plan_from_dict(
                {
                    "intent": "navigate_by_visual_landmark",
                    "task_goal": {
                        "target": "black street lamp",
                        "target_attributes": {"color": "black", "category": "street_lamp"},
                        "final_state": "uav_facing_target_and_hovering",
                    },
                    "phases": [
                        _phase(
                            "orient_right",
                            "orient toward the right-side search area",
                            ["right-side area is visible in the latest observation"],
                            ["right-side area is not yet in view"],
                        ),
                        _phase(
                            "search_landmark",
                            "identify the black street lamp candidate",
                            ["black street lamp or strong street-lamp candidate is visible"],
                            ["street lamp identity is unclear"],
                        ),
                        _phase(
                            "align_to_landmark",
                            "face the street lamp well enough to finish",
                            ["street lamp remains visible", "view is roughly aligned with the target"],
                            ["target identity is unclear", "view is not plausibly aligned"],
                        ),
                        _phase(
                            "stop",
                            "hover after facing the street lamp",
                            ["UAV is stable and the target-facing task is satisfied"],
                            ["UAV is not stable"],
                        ),
                    ],
                }
            )
        raise ValueError("rule_based parser does not support this instruction yet")

    def _parse_with_model(self, instruction: str) -> SemanticPlan:
        if not self.model_config.api_key:
            raise TaskParserConfigError("Model task parser requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise TaskParserConfigError(f"Unsupported model provider: {self.model_config.provider}")
        structure = self._call_json_contract(
            self._build_structure_messages(instruction),
            SemanticPlanStructure.from_dict,
            contract_name="semantic structure",
        )
        criteria_by_phase: list[dict[str, Any]] = []
        phase_traces: list[dict[str, Any]] = []
        for phase in structure.phases:
            guidance, selection_trace = self._resolve_phase_guidance(instruction, structure, phase)

            def validate_criteria(
                data: dict[str, Any],
                immutable_phase: dict[str, Any] = phase,
            ) -> dict[str, Any]:
                return validate_phase_criteria(
                    data,
                    immutable_phase,
                    instruction=instruction,
                )

            criteria = self._call_json_contract(
                self._build_phase_criteria_messages(instruction, structure, phase, guidance),
                validate_criteria,
                contract_name=f"phase criteria {phase['index']}:{phase['name']}",
            )
            criteria_by_phase.append(criteria)
            phase_trace = {
                "index": phase["index"],
                "name": phase["name"],
                **selection_trace,
                "criteria_status": "ok",
            }
            phase_traces.append(phase_trace)
        self.last_parse_trace = {
            "strategy": "structure_then_phase_skill_criteria",
            "structure_strategy": "single_pass_navigable_checkpoint",
            "structure_phase_count": len(structure.phases),
            "configured_evidence": list(self.configured_evidence),
            "phases": phase_traces,
            "warnings": list(self.config_warnings),
        }
        return merge_semantic_plan(structure, criteria_by_phase)

    def _call_json_contract(
        self,
        messages: list[dict[str, str]],
        validator: Callable[[dict[str, Any]], Any],
        *,
        contract_name: str,
    ) -> Any:
        first_content = self._request_model_content(messages)
        try:
            return validator(_loads_json_object(first_content))
        except Exception as first_exc:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": first_content},
                {
                    "role": "user",
                    "content": (
                        f"The previous response violated the {contract_name} contract: "
                        f"{str(first_exc)[:500]}. Return one corrected JSON object only."
                    ),
                },
            ]
            second_content = self._request_model_content(repair_messages)
            return validator(_loads_json_object(second_content))

    def _request_model_content(self, messages: list[dict[str, str]]) -> str:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": messages,
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
                f"Model task parser request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model task parser request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return str(content)

    def _build_structure_messages(self, instruction: str) -> list[dict[str, str]]:
        """Build the skill-free task-structure request."""

        messages = [
            {"role": "system", "content": self.agent_rules},
            {"role": "system", "content": self.user_preferences},
        ]
        if self.protocol_context:
            messages.append(
                {"role": "system", "content": self.protocol_context}
            )
        messages.extend(
            [
                {"role": "system", "content": self.structure_prompt},
                {"role": "user", "content": instruction},
            ]
        )
        return messages

    def _build_phase_criteria_messages(
        self,
        instruction: str,
        structure: SemanticPlanStructure,
        phase: dict[str, Any],
        guidance: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self.agent_rules},
            {"role": "system", "content": self.user_preferences},
        ]
        if guidance:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Selected knowledge for this immutable semantic phase follows. "
                        "Treat it as domain guidance, not observation, tools, or output schema.\n\n"
                        + json.dumps(guidance, ensure_ascii=False)
                    ),
                }
            )
        if self.configured_evidence:
            messages.append(
                {
                    "role": "system",
                    "content": _configured_evidence_message(self.configured_evidence),
                }
            )
        # Keep an optional external protocol adjacent to the criteria contract.
        # Domain skills and evidence describe how to judge a state, but do not
        # override an explicitly supplied protocol definition.
        if self.protocol_context:
            messages.append(
                {"role": "system", "content": self.protocol_context}
            )
        # Completion-condition quality is a shared semantic contract rather
        # than a task tutorial. Keep it in its own message so Criteria remains
        # a small schema assembler and domain Skills remain reusable.
        messages.append(
            {"role": "system", "content": self.done_when_rules}
        )
        messages.extend(
            [
                {"role": "system", "content": self.criteria_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": instruction,
                            "task_goal": structure.task_goal,
                            "phase": dict(phase),
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        return messages

    def _resolve_phase_guidance(
        self,
        instruction: str,
        structure: SemanticPlanStructure,
        phase: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            result = self.phase_skill_resolver.resolve(instruction, structure.task_goal, phase)
            trace = result.trace()
            return list(result.selected_skills), trace
        except Exception as exc:
            return [], {
                "selection_status": "resolver_error",
                "selected_skills": [],
                "warnings": [f"semantic_criteria_skill_resolution_failed:{str(exc)[:240]}"],
            }

    def _load_structure_prompt(self) -> str:
        configured = self.model_config.task_parser_structure_prompt_path
        path = Path(configured) if configured else self._default_prompt_path("task_parser_structure_system.txt")
        return path.read_text(encoding="utf-8").strip()

    def _load_criteria_prompt(self) -> str:
        configured = self.model_config.task_parser_criteria_prompt_path
        legacy = self.model_config.task_parser_prompt_path
        if configured:
            path = Path(configured)
            if legacy:
                self.config_warnings.append("legacy_task_parser_prompt_path_ignored")
        elif legacy:
            path = Path(legacy)
            self.config_warnings.append("legacy_task_parser_prompt_path_used_for_criteria")
        else:
            path = self._default_prompt_path("task_parser_phase_criteria_system.txt")
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _default_prompt_path(name: str) -> Path:
        return Path(__file__).resolve().parent / "prompts" / name

    def _default_phase_skill_resolver(self):
        # Lazy import keeps rule-based parsing independent from plugin discovery
        # and avoids making semantic schemas depend on plugin implementation.
        from src.plugins.skills.semantic_criteria import SemanticCriteriaSkillResolver

        return SemanticCriteriaSkillResolver(self.model_config)

    @staticmethod
    def _load_agent_rules() -> str:
        return TaskParser._load_required_text(
            DEFAULT_AGENT_RULES_PATH,
            label="FlyAgent rules",
        )

    @staticmethod
    def _load_required_text(path: Path, *, label: str) -> str:
        if not path.exists():
            raise TaskParserConfigError(f"{label} file not found: {path}")
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise TaskParserConfigError(f"{label} file is empty: {path}")
        return text

    @staticmethod
    def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
        return any(candidate in text for candidate in candidates)

    def _is_observe_and_clarify(self, instruction: str) -> bool:
        return (
            self._contains_any(instruction, ("观察", "看看", "看一看"))
            and self._contains_any(instruction, ("右侧", "右边"))
            and self._contains_any(instruction, ("不确定", "不明确", "询问", "问我"))
        )


def _phase(
    name: str,
    goal: str,
    done_when: list[str],
    continue_when: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "goal": goal,
        "done_when": done_when,
        "continue_when": continue_when,
    }


def _normalize_configured_evidence(capabilities: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in capabilities or []:
        if not isinstance(item, str):
            continue
        value = " ".join(item.split())
        if not value:
            continue
        if len(value) > MAX_CONFIGURED_EVIDENCE_CHARS:
            value = value[: MAX_CONFIGURED_EVIDENCE_CHARS - 3].rstrip() + "..."
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= MAX_CONFIGURED_EVIDENCE_ITEMS:
            break
    return normalized


def _configured_evidence_message(capabilities: list[str]) -> str:
    bullets = "\n".join(f"- {item}" for item in capabilities)
    return (
        "Potential semantic evidence configured for this run is listed below. "
        "It describes evidence that may become available during execution, not "
        "facts already observed in the current scene.\n"
        + bullets
    )


def _loads_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model task parser response must be a JSON object")
    return value
