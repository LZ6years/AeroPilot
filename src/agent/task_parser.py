"""Parse user instructions into semantic task plans."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.schemas import SemanticPlan, TaskParseResult
from src.agent.semantic_plan import semantic_plan_from_dict, validate_semantic_plan
from src.runtime.configs.model_config import ModelConfig, load_model_config


class TaskParserConfigError(RuntimeError):
    """Raised when model parsing is requested without usable model config."""


class TaskParser:
    def __init__(self, mode: str | None = None, model_config: ModelConfig | None = None) -> None:
        # Use model mode by default; tests and offline smoke can opt into rule_based.
        self.model_config = model_config or load_model_config()
        self.mode = mode or self.model_config.task_parser_mode
        self.system_prompt = self._load_system_prompt()

    def parse(self, instruction: str) -> TaskParseResult:
        # Return a validated semantic plan without executing any primitive action.
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
        # First M3 slice: deterministic Chinese examples for schema and runtime integration.
        if self._is_observe_and_clarify(instruction):
            return semantic_plan_from_dict(
                {
                    "intent": "observe_and_clarify",
                    "goal": {
                        "target": {"description": "右侧区域中的目标", "attributes": {}},
                        "final_state": "uav_hovering_or_waiting_for_clarification",
                    },
                    "phases": [
                        {"name": "orient_right", "meaning": "向右侧调整视角"},
                        {"name": "observe_area", "meaning": "观察右侧区域"},
                        {"name": "clarify_if_uncertain", "meaning": "目标不明确时悬停并请求用户确认"},
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
                    "goal": {
                        "target": {"description": "建筑入口", "attributes": {"category": "building_entrance"}},
                        "final_state": "uav_hovering_in_front_of_target",
                    },
                    "phases": [
                        {"name": "orient_left", "meaning": "根据用户线索先向左侧观察"},
                        {"name": "search_landmark", "meaning": "寻找建筑入口"},
                        {"name": "approach_landmark", "meaning": "逐步接近入口前方区域"},
                        {"name": "stop", "meaning": "在入口前悬停"},
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
                    "goal": {
                        "target": {
                            "description": "红色车辆",
                            "attributes": {"color": "red", "category": "vehicle"},
                        },
                        "final_state": "uav_near_target_with_safe_distance",
                    },
                    "phases": [
                        {"name": "search_landmark", "meaning": "在前方视野中寻找红色车辆"},
                        {"name": "approach_landmark", "meaning": "在安全距离约束下靠近目标"},
                        {"name": "stop", "meaning": "悬停并等待下一步指令"},
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
                    "goal": {
                        "target": {
                            "description": "黑色路灯",
                            "attributes": {"color": "black", "category": "street_lamp"},
                        },
                        "final_state": "uav_facing_target_and_hovering",
                    },
                    "phases": [
                        {"name": "orient_right", "meaning": "根据用户线索先向右侧观察"},
                        {"name": "search_landmark", "meaning": "寻找黑色路灯"},
                        {"name": "align_to_landmark", "meaning": "让 UAV 面向路灯"},
                        {"name": "stop", "meaning": "悬停并结束任务"},
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
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
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
            self._contains_any(instruction, ("观察", "看看", "看一下"))
            and self._contains_any(instruction, ("右侧", "右边"))
            and self._contains_any(instruction, ("不确定", "不明确", "询问", "问我"))
        )


def _loads_json_object(content: str) -> dict[str, Any]:
    # Some providers wrap JSON in Markdown despite the prompt, so accept that but still require one object.
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
