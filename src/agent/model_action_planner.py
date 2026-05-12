"""Plan short-horizon actions with a configured model provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.schemas import PlannedStep, ShortHorizonPlan
from src.agent.task_parser import _loads_json_object
from src.runtime.configs.model_config import ModelConfig, load_model_config


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when model action planning is requested without usable config."""


class ModelActionPlanner:
    planner_mode = "model"
    ALLOWED_TOOL_STEPS = {"uav.hover", "uav.turn_relative", "uav.move_body"}

    def __init__(self, model_config: ModelConfig | None = None, prompt_path: str | None = None) -> None:
        self.model_config = model_config or load_model_config()
        self.prompt_path = prompt_path or self.model_config.action_planner_prompt_path
        self.system_prompt = self._load_system_prompt()

    def plan_next(self, state: MissionState, context: dict | None = None) -> ShortHorizonPlan:
        if context is None:
            raise ValueError("ModelActionPlanner requires M5 context")
        if not self.model_config.api_key:
            raise ModelActionPlannerConfigError("Model action planner requires FLYAGENT_MODEL_API_KEY")
        if self.model_config.provider != "openai_compatible":
            raise ModelActionPlannerConfigError(f"Unsupported model provider: {self.model_config.provider}")
        response = self._call_openai_compatible(context)
        plan = self._normalize_plan(ShortHorizonPlan.from_dict(response))
        self._validate_plan(plan, state.current_phase_name)
        return plan

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
                f"Model action planner request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model action planner request failed: {exc}") from exc
        content = data["choices"][0]["message"]["content"]
        return _loads_json_object(content)

    def _validate_plan(self, plan: ShortHorizonPlan, expected_phase: str) -> None:
        if plan.phase != expected_phase:
            raise ValueError(f"Model action planner phase mismatch: {plan.phase} != {expected_phase}")
        if not 1 <= len(plan.steps) <= 3:
            raise ValueError("Model action planner must return 1 to 3 steps")
        for step in plan.steps:
            if step.kind == "observe":
                if step.tool is not None:
                    raise ValueError("observe step must not include a tool")
                continue
            if step.kind != "tool":
                raise ValueError(f"Unsupported model planned step kind: {step.kind}")
            if step.tool not in self.ALLOWED_TOOL_STEPS:
                raise ValueError(f"Tool is not allowed in model action plan: {step.tool}")

    def _normalize_plan(self, plan: ShortHorizonPlan) -> ShortHorizonPlan:
        steps = []
        for step in plan.steps:
            if step.kind != "tool" or step.tool != "uav.move_body":
                steps.append(step)
                continue
            steps.append(
                PlannedStep.tool_step(
                    step.tool,
                    _normalize_move_body_args(step.args),
                    step.reason,
                )
            )
        return ShortHorizonPlan(plan.phase, steps)

    def _load_system_prompt(self) -> str:
        path = Path(self.prompt_path) if self.prompt_path else Path(__file__).resolve().parent / "prompts" / "action_planner_system.txt"
        return path.read_text(encoding="utf-8").strip()


def _normalize_move_body_args(args: dict[str, Any]) -> dict[str, Any]:
    data = dict(args)
    direction = str(data.get("direction", "")).lower().strip()
    direction_aliases = {
        "front": "forward",
        "ahead": "forward",
        "back": "backward",
        "backs": "backward",
        "backwards": "backward",
        "rear": "backward",
        "leftward": "left",
        "rightward": "right",
        "upward": "up",
        "downward": "down",
    }
    if direction:
        normalized = direction_aliases.get(direction, direction)
        if normalized not in {"forward", "backward", "left", "right", "up", "down"}:
            raise ValueError(f"Unsupported uav.move_body direction: {direction}")
        return {"direction": normalized, "duration": _duration(data)}

    if any(key in data for key in ("vx", "vy", "vz")):
        return {
            "vx": float(data.get("vx", 0.0)),
            "vy": float(data.get("vy", 0.0)),
            "vz": float(data.get("vz", 0.0)),
            "duration": _duration(data),
        }

    distance_x = _first_number(data, ("x", "x_m", "forward_m"))
    distance_y = _first_number(data, ("y", "y_m", "right_m"))
    distance_z = _first_number(data, ("z", "z_m", "down_m"))
    if distance_x is not None or distance_y is not None or distance_z is not None:
        duration = _duration(data)
        if duration <= 0.0:
            raise ValueError("uav.move_body duration must be greater than 0")
        return {
            "vx": float(distance_x or 0.0) / duration,
            "vy": float(distance_y or 0.0) / duration,
            "vz": float(distance_z or 0.0) / duration,
            "duration": duration,
        }

    allowed = {"direction", "duration", "vx", "vy", "vz"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unsupported uav.move_body args: {', '.join(unknown)}")
    return {"direction": "forward", "duration": _duration(data)}


def _duration(data: dict[str, Any]) -> float:
    return float(data.get("duration", data.get("duration_sec", 1.0)))


def _first_number(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data:
            return float(data[key])
    return None
