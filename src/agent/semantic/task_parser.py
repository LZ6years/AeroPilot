"""Parse user instructions into lightweight semantic task plans."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.semantic.schemas import SemanticPlan, TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict, validate_semantic_plan
from src.runtime.configs.model_config import ModelConfig, load_model_config


class TaskParserConfigError(RuntimeError):
    """Raised when model parsing is requested without usable model config."""


class TaskParser:
    def __init__(self, mode: str | None = None, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or load_model_config()
        self.mode = mode or self.model_config.task_parser_mode
        self.system_prompt = self._load_system_prompt()

    def parse(self, instruction: str) -> TaskParseResult:
        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise ValueError("instruction must not be empty")
        if self.mode == "rule_based":
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
                    "uncertainties": [
                        "target_description_is_underspecified",
                        "multiple_candidates_may_exist",
                        "user_clarification_may_be_required",
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
                            observe=["latest_observation", "phase_start_reference", "phase_execution_summary"],
                        ),
                        _phase(
                            "stop",
                            "hold in front of the entrance",
                            ["UAV is stable near the entrance front area"],
                            ["UAV is not stable or not plausibly in front of the entrance"],
                        ),
                    ],
                    "uncertainties": [
                        "left_turn_angle_may_not_be_exact",
                        "entrance_may_not_be_visible_after_first_turn",
                        "front_of_target_requires_observation_and_depth",
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
                            observe=["latest_observation", "phase_start_reference", "phase_execution_summary"],
                        ),
                        _phase(
                            "stop",
                            "hold near the vehicle with safe distance",
                            ["UAV is stable near the vehicle with safe distance"],
                            ["UAV is not stable or safe distance is unclear"],
                        ),
                    ],
                    "uncertainties": [
                        "target_identity_requires_observation",
                        "safe_distance_requires_depth_or_user_constraint",
                        "target_may_move_or_be_occluded",
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
                            observe=["phase_execution_summary"],
                        ),
                    ],
                    "uncertainties": [
                        "right_turn_angle_may_not_be_exact",
                        "target_may_not_be_visible_after_first_turn",
                        "target_identity_requires_observation",
                    ],
                }
            )
        raise ValueError("rule_based parser does not support this instruction yet")

    def _parse_with_model(self, instruction: str) -> SemanticPlan:
        if not self.model_config.api_key:
            raise TaskParserConfigError("Model task parser requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise TaskParserConfigError(f"Unsupported model provider: {self.model_config.provider}")
        response = self._call_openai_compatible(instruction)
        return semantic_plan_from_dict(response)

    def _call_openai_compatible(self, instruction: str) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
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
                f"Model task parser request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model task parser request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _load_system_prompt(self) -> str:
        prompt_path = self.model_config.task_parser_prompt_path
        if prompt_path:
            path = Path(prompt_path)
        else:
            path = Path(__file__).resolve().parent / "prompts" / "task_parser_system.txt"
        return path.read_text(encoding="utf-8").strip()

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
    *,
    observe: list[str] | None = None,
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "goal": goal,
        "done_when": done_when,
        "observe": observe or ["latest_observation", "phase_execution_summary"],
        "continue_when": continue_when,
        "metrics": metrics or [],
    }


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
