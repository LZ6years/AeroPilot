"""Plan short-horizon actions with a configured model provider."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.planning.schemas import PlannedStep, ShortHorizonPlan
from src.agent.semantic.task_parser import _loads_json_object
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
        plan = ShortHorizonPlan.from_dict(response)
        self._validate_plan(plan, state.current_phase_name, context)
        return self._normalize_plan(plan)

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

    def _validate_plan(self, plan: ShortHorizonPlan, expected_phase: str, context: dict[str, Any]) -> None:
        if plan.phase != expected_phase:
            raise ValueError(f"Model action planner phase mismatch: {plan.phase} != {expected_phase}")
        if not 1 <= len(plan.steps) <= 3:
            raise ValueError("Model action planner must return 1 to 3 steps")
        constraints = _action_constraints(context)
        allowed_tools = _allowed_tools(constraints)
        forbidden_tools = set(constraints.get("forbidden_tools", []))
        for step in plan.steps:
            if step.kind == "observe":
                raise ValueError("observe steps are managed by runtime perception policy")
            if step.kind != "tool":
                raise ValueError(f"Unsupported model planned step kind: {step.kind}")
            if step.tool in forbidden_tools:
                raise ValueError(f"Tool is forbidden in model action plan: {step.tool}")
            if step.tool not in allowed_tools:
                raise ValueError(f"Tool is not allowed in model action plan: {step.tool}")
            self._validate_tool_args(step, constraints)

    def _validate_tool_args(self, step: PlannedStep, constraints: dict[str, Any]) -> None:
        if step.tool == "uav.hover":
            if step.args:
                raise ValueError("uav.hover args must be {}")
            return
        if step.tool == "uav.turn_relative":
            _validate_turn_relative_args(step.args, constraints)
            return
        if step.tool == "uav.move_body":
            _validate_move_body_args(step.args, constraints)

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


def _action_constraints(context: dict[str, Any]) -> dict[str, Any]:
    constraints = context.get("allowed_action_constraints", {}) if isinstance(context, dict) else {}
    return constraints if isinstance(constraints, dict) else {}


def _allowed_tools(constraints: dict[str, Any]) -> set[str]:
    tools = constraints.get("planner_tools")
    if isinstance(tools, list) and tools:
        return {str(item) for item in tools}
    return set(ModelActionPlanner.ALLOWED_TOOL_STEPS)


def _validate_turn_relative_args(args: dict[str, Any], constraints: dict[str, Any]) -> None:
    if set(args) != {"yaw_deg"}:
        raise ValueError("uav.turn_relative args must be exactly {'yaw_deg'}")
    yaw_deg = _number_arg(args, "yaw_deg", "uav.turn_relative")
    limits = (
        constraints.get("action_schemas", {})
        .get("uav.turn_relative", {})
        .get("limits", {})
    )
    min_yaw = limits.get("yaw_deg_min")
    max_yaw = limits.get("yaw_deg_max")
    if min_yaw is not None and yaw_deg < float(min_yaw):
        raise ValueError(f"uav.turn_relative yaw_deg below limit: {yaw_deg} < {float(min_yaw)}")
    if max_yaw is not None and yaw_deg > float(max_yaw):
        raise ValueError(f"uav.turn_relative yaw_deg above limit: {yaw_deg} > {float(max_yaw)}")


def _validate_move_body_args(args: dict[str, Any], constraints: dict[str, Any]) -> None:
    keys = set(args)
    if keys == {"direction", "duration"}:
        direction = str(args.get("direction", "")).lower().strip()
        if direction not in {"forward", "backward", "left", "right", "up", "down"}:
            raise ValueError(f"Unsupported uav.move_body direction: {args.get('direction')}")
        _validate_duration(args, constraints)
        return
    if keys == {"vx", "vy", "vz", "duration"}:
        limits = (
            constraints.get("action_schemas", {})
            .get("uav.move_body", {})
            .get("limits", {})
        )
        vx = _number_arg(args, "vx", "uav.move_body")
        vy = _number_arg(args, "vy", "uav.move_body")
        vz = _number_arg(args, "vz", "uav.move_body")
        max_xy = limits.get("abs_vx_vy_max")
        max_z = limits.get("abs_vz_max")
        if max_xy is not None and (abs(vx) > float(max_xy) or abs(vy) > float(max_xy)):
            raise ValueError("uav.move_body planar velocity exceeds context limits")
        if max_z is not None and abs(vz) > float(max_z):
            raise ValueError("uav.move_body vertical velocity exceeds context limits")
        _validate_duration(args, constraints)
        return
    raise ValueError(
        "uav.move_body args must be exactly {'direction', 'duration'} or {'vx', 'vy', 'vz', 'duration'}"
    )


def _validate_duration(args: dict[str, Any], constraints: dict[str, Any]) -> None:
    duration = _number_arg(args, "duration", "uav.move_body")
    if duration <= 0.0:
        raise ValueError("uav.move_body duration must be greater than 0")
    limit = (
        constraints.get("action_schemas", {})
        .get("uav.move_body", {})
        .get("limits", {})
        .get("duration_max")
    )
    if limit is not None and duration > float(limit):
        raise ValueError(f"uav.move_body duration exceeds context limit: {duration} > {float(limit)}")


def _number_arg(args: dict[str, Any], key: str, tool: str) -> float:
    if key not in args:
        raise ValueError(f"{tool} requires numeric arg: {key}")
    try:
        return float(args[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{tool} arg must be numeric: {key}") from exc
