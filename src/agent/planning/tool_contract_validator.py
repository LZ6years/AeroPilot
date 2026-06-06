"""Validate model-produced action plans against the planner tool contract.

This module is intentionally narrower than the model planner.  The planner
talks to the model and parses JSON; this validator checks whether the parsed
plan obeys the action contract carried in PlannerContext.  Keeping the contract
logic here makes it reusable by future repair-planning code without giving that
future layer direct access to model-call details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from src.agent.planning.schemas import ACTION_WINDOW_SCOPE_CURRENT_PHASE, PlannedStep, ShortHorizonPlan


DEFAULT_ALLOWED_TOOL_STEPS = {"uav.hover", "uav.turn_relative", "uav.move_body"}


@dataclass(frozen=True)
class ToolContractViolation:
    """Structured description of one planner tool-contract failure."""

    code: str
    message: str
    step_index: int | None = None
    tool: str | None = None
    field: str | None = None
    expected: Any | None = None
    actual: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for logs and repair prompts."""

        data = {
            "code": self.code,
            "message": self.message,
        }
        optional = {
            "step_index": self.step_index,
            "tool": self.tool,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }
        data.update({key: value for key, value in optional.items() if value is not None})
        return data


class ToolContractValidationError(ValueError):
    """Raised when a model action plan violates PlannerContext's action contract."""

    def __init__(self, violation: ToolContractViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)

    def to_dict(self) -> dict[str, Any]:
        return self.violation.to_dict()


class ToolContractValidator:
    """Validate ShortHorizonPlan tool steps against PlannerContext constraints."""

    def __init__(self, fallback_allowed_tools: set[str] | None = None) -> None:
        self.fallback_allowed_tools = set(fallback_allowed_tools or DEFAULT_ALLOWED_TOOL_STEPS)

    def validate(self, plan: ShortHorizonPlan, expected_phase: str, context: dict[str, Any]) -> ShortHorizonPlan:
        """Validate a parsed plan and return it unchanged when it satisfies the contract."""

        if plan.phase != expected_phase:
            self._fail(
                "phase_mismatch",
                f"Model action planner phase mismatch: {plan.phase} != {expected_phase}",
                field="phase",
                expected=expected_phase,
                actual=plan.phase,
            )
        if not 1 <= len(plan.steps) <= 3:
            self._fail(
                "step_count_out_of_range",
                "Model action planner must return 1 to 3 steps",
                field="steps",
                expected="1..3",
                actual=len(plan.steps),
            )
        self._validate_action_window(plan)

        constraints = self._action_constraints(context)
        allowed_tools = self._allowed_tools(constraints)
        forbidden_tools = set(constraints.get("forbidden_tools", []))
        for index, step in enumerate(plan.steps):
            self._validate_step(step, index, constraints, allowed_tools, forbidden_tools)
        return plan

    def _validate_action_window(self, plan: ShortHorizonPlan) -> None:
        window = plan.action_window
        if not window.intent.strip():
            self._fail(
                "action_window_intent_missing",
                "action_window.intent must be a non-empty description",
                field="action_window.intent",
                expected="non-empty string",
                actual=window.intent,
            )
        if window.scope != ACTION_WINDOW_SCOPE_CURRENT_PHASE:
            self._fail(
                "action_window_scope_invalid",
                "action_window.scope must be current_phase_only",
                field="action_window.scope",
                expected=ACTION_WINDOW_SCOPE_CURRENT_PHASE,
                actual=window.scope,
            )
        if window.requires_intermediate_observation and len(plan.steps) > 1:
            self._fail(
                "action_window_intermediate_observation_requires_single_step",
                "action_window requires an intermediate observation, so the plan must contain exactly one step",
                field="action_window.requires_intermediate_observation",
                expected="true only with exactly one step",
                actual={"requires_intermediate_observation": True, "step_count": len(plan.steps)},
            )
        if not window.expected_effect.strip():
            self._fail(
                "action_window_expected_effect_missing",
                "action_window.expected_effect must be a non-empty description",
                field="action_window.expected_effect",
                expected="non-empty string",
                actual=window.expected_effect,
            )

    def _validate_step(
        self,
        step: PlannedStep,
        index: int,
        constraints: dict[str, Any],
        allowed_tools: set[str],
        forbidden_tools: set[str],
    ) -> None:
        if step.kind == "observe":
            self._fail(
                "observe_managed_by_runtime",
                "observe steps are managed by runtime perception policy",
                step_index=index,
                tool=step.tool,
                field="kind",
                expected="tool",
                actual="observe",
            )
        if step.kind != "tool":
            self._fail(
                "unsupported_step_kind",
                f"Unsupported model planned step kind: {step.kind}",
                step_index=index,
                tool=step.tool,
                field="kind",
                expected="tool",
                actual=step.kind,
            )
        if step.tool in forbidden_tools:
            self._fail(
                "forbidden_tool",
                f"Tool is forbidden in model action plan: {step.tool}",
                step_index=index,
                tool=step.tool,
                field="tool",
                expected="not in forbidden_tools",
                actual=step.tool,
            )
        if step.tool not in allowed_tools:
            self._fail(
                "tool_not_allowed",
                f"Tool is not allowed in model action plan: {step.tool}",
                step_index=index,
                tool=step.tool,
                field="tool",
                expected=sorted(allowed_tools),
                actual=step.tool,
            )
        self._validate_tool_args(step, index, constraints)

    def _validate_tool_args(self, step: PlannedStep, index: int, constraints: dict[str, Any]) -> None:
        if step.tool == "uav.hover":
            if step.args:
                self._fail(
                    "hover_args_not_empty",
                    "uav.hover args must be {}",
                    step_index=index,
                    tool=step.tool,
                    field="args",
                    expected={},
                    actual=step.args,
                )
            return
        if step.tool == "uav.turn_relative":
            self._validate_turn_relative_args(step.args, index, constraints)
            return
        if step.tool == "uav.move_body":
            self._validate_move_body_args(step.args, index, constraints)
            return
        if str(step.tool).startswith("skill."):
            self._validate_skill_tool_args(step.tool or "", step.args, index, constraints)
            return
        self._fail(
            "tool_schema_not_supported",
            f"Tool has no validator in model action plan: {step.tool}",
            step_index=index,
            tool=step.tool,
            field="tool",
            expected=sorted(DEFAULT_ALLOWED_TOOL_STEPS),
            actual=step.tool,
        )

    def _validate_skill_tool_args(self, tool: str, args: dict[str, Any], index: int, constraints: dict[str, Any]) -> None:
        schema = constraints.get("action_schemas", {}).get(tool, {})
        input_schema = schema.get("input_schema") if isinstance(schema, dict) else None
        if not isinstance(input_schema, dict):
            self._fail(
                "skill_tool_schema_missing",
                f"Skill tool has no input_schema in planner context: {tool}",
                step_index=index,
                tool=tool,
                field="input_schema",
                expected="JSON schema object",
                actual=schema,
            )
        _validate_json_schema_subset(self, tool, args, input_schema, index)

    def _validate_turn_relative_args(self, args: dict[str, Any], index: int, constraints: dict[str, Any]) -> None:
        if set(args) != {"yaw_deg"}:
            self._fail(
                "turn_args_schema_error",
                "uav.turn_relative args must be exactly {'yaw_deg'}",
                step_index=index,
                tool="uav.turn_relative",
                field="args",
                expected={"yaw_deg": "number"},
                actual=args,
            )
        yaw_deg = self._number_arg(args, "yaw_deg", "uav.turn_relative", index)
        limits = (
            constraints.get("action_schemas", {})
            .get("uav.turn_relative", {})
            .get("limits", {})
        )
        min_yaw = limits.get("yaw_deg_min")
        max_yaw = limits.get("yaw_deg_max")
        if min_yaw is not None and yaw_deg < float(min_yaw):
            self._fail(
                "turn_yaw_below_limit",
                f"uav.turn_relative yaw_deg below limit: {yaw_deg} < {float(min_yaw)}",
                step_index=index,
                tool="uav.turn_relative",
                field="yaw_deg",
                expected=f">= {float(min_yaw)}",
                actual=yaw_deg,
            )
        if max_yaw is not None and yaw_deg > float(max_yaw):
            self._fail(
                "turn_yaw_above_limit",
                f"uav.turn_relative yaw_deg above limit: {yaw_deg} > {float(max_yaw)}",
                step_index=index,
                tool="uav.turn_relative",
                field="yaw_deg",
                expected=f"<= {float(max_yaw)}",
                actual=yaw_deg,
            )

    def _validate_move_body_args(self, args: dict[str, Any], index: int, constraints: dict[str, Any]) -> None:
        keys = set(args)
        if keys == {"direction", "duration"}:
            direction = str(args.get("direction", "")).lower().strip()
            if direction not in {"forward", "backward", "left", "right", "up", "down"}:
                self._fail(
                    "move_body_direction_invalid",
                    f"Unsupported uav.move_body direction: {args.get('direction')}",
                    step_index=index,
                    tool="uav.move_body",
                    field="direction",
                    expected="forward|backward|left|right|up|down",
                    actual=args.get("direction"),
                )
            self._validate_duration(args, index, constraints)
            return
        if keys == {"vx", "vy", "vz", "duration"}:
            limits = (
                constraints.get("action_schemas", {})
                .get("uav.move_body", {})
                .get("limits", {})
            )
            vx = self._number_arg(args, "vx", "uav.move_body", index)
            vy = self._number_arg(args, "vy", "uav.move_body", index)
            vz = self._number_arg(args, "vz", "uav.move_body", index)
            max_xy = limits.get("abs_vx_vy_max")
            max_z = limits.get("abs_vz_max")
            if max_xy is not None and (abs(vx) > float(max_xy) or abs(vy) > float(max_xy)):
                self._fail(
                    "move_body_planar_velocity_limit_exceeded",
                    "uav.move_body planar velocity exceeds context limits",
                    step_index=index,
                    tool="uav.move_body",
                    field="vx/vy",
                    expected=f"abs(vx), abs(vy) <= {float(max_xy)}",
                    actual={"vx": vx, "vy": vy},
                )
            if max_z is not None and abs(vz) > float(max_z):
                self._fail(
                    "move_body_vertical_velocity_limit_exceeded",
                    "uav.move_body vertical velocity exceeds context limits",
                    step_index=index,
                    tool="uav.move_body",
                    field="vz",
                    expected=f"abs(vz) <= {float(max_z)}",
                    actual=vz,
                )
            self._validate_duration(args, index, constraints)
            return
        self._fail(
            "move_body_args_schema_error",
            "uav.move_body args must be exactly {'direction', 'duration'} or {'vx', 'vy', 'vz', 'duration'}",
            step_index=index,
            tool="uav.move_body",
            field="args",
            expected=[
                {"direction": "forward|backward|left|right|up|down", "duration": "number"},
                {"vx": "number", "vy": "number", "vz": "number", "duration": "number"},
            ],
            actual=args,
        )

    def _validate_duration(self, args: dict[str, Any], index: int, constraints: dict[str, Any]) -> None:
        duration = self._number_arg(args, "duration", "uav.move_body", index)
        if duration <= 0.0:
            self._fail(
                "move_body_duration_non_positive",
                "uav.move_body duration must be greater than 0",
                step_index=index,
                tool="uav.move_body",
                field="duration",
                expected="> 0",
                actual=duration,
            )
        limit = (
            constraints.get("action_schemas", {})
            .get("uav.move_body", {})
            .get("limits", {})
            .get("duration_max")
        )
        if limit is not None and duration > float(limit):
            self._fail(
                "move_body_duration_limit_exceeded",
                f"uav.move_body duration exceeds context limit: {duration} > {float(limit)}",
                step_index=index,
                tool="uav.move_body",
                field="duration",
                expected=f"<= {float(limit)}",
                actual=duration,
            )

    def _number_arg(self, args: dict[str, Any], key: str, tool: str, index: int) -> float:
        if key not in args:
            self._fail(
                "numeric_arg_missing",
                f"{tool} requires numeric arg: {key}",
                step_index=index,
                tool=tool,
                field=key,
                expected="number",
            )
        try:
            return float(args[key])
        except (TypeError, ValueError):
            self._fail(
                "numeric_arg_invalid",
                f"{tool} arg must be numeric: {key}",
                step_index=index,
                tool=tool,
                field=key,
                expected="number",
                actual=args.get(key),
            )

    def _action_constraints(self, context: dict[str, Any]) -> dict[str, Any]:
        constraints = context.get("allowed_action_constraints", {}) if isinstance(context, dict) else {}
        return constraints if isinstance(constraints, dict) else {}

    def _allowed_tools(self, constraints: dict[str, Any]) -> set[str]:
        tools = constraints.get("planner_tools")
        if isinstance(tools, list) and tools:
            return {str(item) for item in tools}
        return set(self.fallback_allowed_tools)

    def _fail(
        self,
        code: str,
        message: str,
        *,
        step_index: int | None = None,
        tool: str | None = None,
        field: str | None = None,
        expected: Any | None = None,
        actual: Any | None = None,
    ) -> NoReturn:
        raise ToolContractValidationError(
            ToolContractViolation(
                code=code,
                message=message,
                step_index=step_index,
                tool=tool,
                field=field,
                expected=expected,
                actual=actual,
            )
        )


def _validate_json_schema_subset(
    validator: ToolContractValidator,
    tool: str,
    args: dict[str, Any],
    schema: dict[str, Any],
    index: int,
) -> None:
    if schema.get("type", "object") != "object":
        validator._fail(
            "skill_tool_schema_type_unsupported",
            f"Skill tool schema must be an object schema: {tool}",
            step_index=index,
            tool=tool,
            field="input_schema.type",
            expected="object",
            actual=schema.get("type"),
        )
    if not isinstance(args, dict):
        validator._fail(
            "skill_tool_args_invalid",
            f"Skill tool args must be an object: {tool}",
            step_index=index,
            tool=tool,
            field="args",
            expected="object",
            actual=args,
        )
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    for name in required:
        if name not in args:
            validator._fail(
                "skill_tool_required_arg_missing",
                f"Skill tool requires arg: {name}",
                step_index=index,
                tool=tool,
                field=str(name),
                expected="present",
            )
    additional_allowed = bool(schema.get("additionalProperties", True))
    if not additional_allowed:
        extra = sorted(set(args) - set(properties))
        if extra:
            validator._fail(
                "skill_tool_additional_arg_forbidden",
                f"Skill tool received unsupported args: {extra}",
                step_index=index,
                tool=tool,
                field="args",
                expected=sorted(properties),
                actual=extra,
            )
    for name, value in args.items():
        prop_schema = properties.get(name)
        if isinstance(prop_schema, dict):
            _validate_json_value_subset(validator, tool, name, value, prop_schema, index)


def _validate_json_value_subset(
    validator: ToolContractValidator,
    tool: str,
    name: str,
    value: Any,
    schema: dict[str, Any],
    index: int,
) -> None:
    expected_type = schema.get("type")
    if expected_type == "string" and not isinstance(value, str):
        _type_fail(validator, tool, name, value, "string", index)
    if expected_type == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        _type_fail(validator, tool, name, value, "number", index)
    if expected_type == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        _type_fail(validator, tool, name, value, "integer", index)
    if expected_type == "boolean" and not isinstance(value, bool):
        _type_fail(validator, tool, name, value, "boolean", index)
    if expected_type == "array":
        if not isinstance(value, list):
            _type_fail(validator, tool, name, value, "array", index)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item_index, item in enumerate(value):
                _validate_json_value_subset(validator, tool, f"{name}[{item_index}]", item, item_schema, index)
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        validator._fail(
            "skill_tool_enum_invalid",
            f"Skill tool arg {name} must be one of {enum}",
            step_index=index,
            tool=tool,
            field=name,
            expected=enum,
            actual=value,
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and float(value) < float(minimum):
            validator._fail(
                "skill_tool_number_below_minimum",
                f"Skill tool arg {name} below minimum",
                step_index=index,
                tool=tool,
                field=name,
                expected=f">= {minimum}",
                actual=value,
            )
        if maximum is not None and float(value) > float(maximum):
            validator._fail(
                "skill_tool_number_above_maximum",
                f"Skill tool arg {name} above maximum",
                step_index=index,
                tool=tool,
                field=name,
                expected=f"<= {maximum}",
                actual=value,
            )


def _type_fail(
    validator: ToolContractValidator,
    tool: str,
    name: str,
    value: Any,
    expected: str,
    index: int,
) -> None:
    validator._fail(
        "skill_tool_arg_type_invalid",
        f"Skill tool arg {name} must be {expected}",
        step_index=index,
        tool=tool,
        field=name,
        expected=expected,
        actual=value,
    )
