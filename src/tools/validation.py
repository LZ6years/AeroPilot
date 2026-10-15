"""Small JSON-Schema subset validator for native tool arguments.

FlyAgent intentionally supports only the schema features emitted by its tool
catalog. Keeping this validator local avoids a new runtime dependency while
still enforcing the exact contract before any side effect occurs.
"""

from __future__ import annotations

import math
from typing import Any

from src.depth_display import display_normalized_roi_to_scene_pixels


class ToolArgumentError(ValueError):
    """Raised when native tool arguments violate their ToolSpec schema."""

    def __init__(self, message: str, *, code: str = "invalid_arguments", path: str | None = None):
        super().__init__(message)
        self.code = code
        self.path = path


def validate_tool_args(args: dict[str, Any], schema: dict[str, Any], path: str = "args") -> None:
    _validate(args, schema, path)


def validate_runtime_bound_tool_args(
    args: dict[str, Any],
    schema: dict[str, Any],
    latest_observation: dict[str, Any],
    path: str = "args",
) -> None:
    """Validate schema-declared arguments bound to the latest observation."""

    contract = schema.get("x-flyagent-observation-roi")
    if not isinstance(contract, dict):
        return
    observation_field = str(contract.get("observation_field") or "observation_id")
    requested = str(args.get(observation_field) or "")
    latest = str(latest_observation.get("observation_id") or "")
    if not latest or requested != latest:
        raise ToolArgumentError(
            f"{path}.{observation_field} must reference the latest observation_id {latest!r}"
        )
    frames = latest_observation.get("frames")
    modality = str(contract.get("modality") or "depth")
    frame = frames.get(modality) if isinstance(frames, dict) else None
    if not isinstance(frame, dict):
        return
    try:
        width = int(frame.get("width"))
        height = int(frame.get("height"))
        min_pixels = int(contract.get("min_scene_pixels", 3))
        coordinates = {
            "x_min": float(args[str(contract["x_min_field"])]),
            "y_min": float(args[str(contract["y_min_field"])]),
            "x_max": float(args[str(contract["x_max_field"])]),
            "y_max": float(args[str(contract["y_max_field"])]),
        }
        if contract.get("coordinate_space") == "depth_overlay_display_normalized":
            x0, y0, x1, y1 = display_normalized_roi_to_scene_pixels(
                **coordinates,
                scene_width=width,
                scene_height=height,
            )
        else:
            x0 = math.floor(coordinates["x_min"] * width)
            y0 = math.floor(coordinates["y_min"] * height)
            x1 = math.ceil(coordinates["x_max"] * width)
            y1 = math.ceil(coordinates["y_max"] * height)
    except ValueError as exc:
        if "color bar" in str(exc):
            raise ToolArgumentError(f"{path} {exc}") from exc
        return
    except (KeyError, TypeError):
        return
    if x1 - x0 < min_pixels or y1 - y0 < min_pixels:
        raise ToolArgumentError(
            f"{path} ROI must cover at least {min_pixels}x{min_pixels} scene pixels"
        )


def _validate(value: Any, schema: dict[str, Any], path: str) -> None:
    """Attach the deepest validator path without parsing human error text."""
    try:
        _validate_at_path(value, schema, path)
    except ToolArgumentError as exc:
        if exc.path is None:
            exc.path = path
        raise


def _validate_at_path(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, list):
        # Select by JSON type, then apply the same constraints recursively.
        # A union must never bypass object, array or nullable validation.
        matches = {
            "null": value is None,
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        }
        selected = next((kind for kind in expected if matches.get(kind, False)), None)
        if selected is None:
            raise ToolArgumentError(f"{path} must have a type in {expected}")
        _validate_at_path(value, {**schema, "type": selected}, path)
        return
    if expected == "object":
        if not isinstance(value, dict):
            raise ToolArgumentError(f"{path} must be an object")
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required", [])
        required = required if isinstance(required, list) else []
        missing = [str(key) for key in required if key not in value]
        if missing:
            raise ToolArgumentError(f"{path} missing required fields: {', '.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ToolArgumentError(f"{path} has unsupported fields: {', '.join(extra)}")
        for key, item in value.items():
            child = properties.get(key)
            if isinstance(child, dict):
                _validate(item, child, f"{path}.{key}")
        _validate_coordinate_spans(value, schema, path)
    elif expected == "array":
        if not isinstance(value, list):
            raise ToolArgumentError(f"{path} must be an array")
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < int(minimum):
            raise ToolArgumentError(f"{path} must contain at least {minimum} items")
        if maximum is not None and len(value) > int(maximum):
            raise ToolArgumentError(f"{path} must contain at most {maximum} items")
        child = schema.get("items")
        if isinstance(child, dict):
            for index, item in enumerate(value):
                _validate(item, child, f"{path}[{index}]")
    elif expected == "string":
        if not isinstance(value, str):
            raise ToolArgumentError(f"{path} must be a string")
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ToolArgumentError(
                f"{path} must contain at least {schema['minLength']} characters"
            )
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ToolArgumentError(
                f"{path} must contain at most {schema['maxLength']} characters"
            )
    elif expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ToolArgumentError(f"{path} must be a number")
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ToolArgumentError(f"{path} must be an integer")
    elif expected == "boolean" and not isinstance(value, bool):
        raise ToolArgumentError(f"{path} must be a boolean")
    elif expected == "null" and value is not None:
        raise ToolArgumentError(f"{path} must be null")

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ToolArgumentError(f"{path} must be one of {enum}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and float(value) < float(schema["minimum"]):
            raise ToolArgumentError(f"{path} must be >= {schema['minimum']}")
        if "exclusiveMinimum" in schema and float(value) <= float(schema["exclusiveMinimum"]):
            raise ToolArgumentError(f"{path} must be > {schema['exclusiveMinimum']}")
        if "maximum" in schema and float(value) > float(schema["maximum"]):
            raise ToolArgumentError(f"{path} must be <= {schema['maximum']}")


def _validate_coordinate_spans(
    value: dict[str, Any], schema: dict[str, Any], path: str
) -> None:
    """Enforce small normalized regions using a FlyAgent schema extension.

    Provider APIs safely ignore the ``x-`` keyword, while Runtime applies the
    relational constraint before a read-only capability is invoked.
    """

    rules = schema.get("x-flyagent-coordinate-spans")
    if not isinstance(rules, list):
        return
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        minimum_field = rule.get("minimum_field")
        maximum_field = rule.get("maximum_field")
        maximum_span = rule.get("maximum_span")
        if not all(
            isinstance(item, str) and item
            for item in (minimum_field, maximum_field)
        ) or not isinstance(maximum_span, (int, float)):
            continue
        minimum_value = value.get(minimum_field)
        maximum_value = value.get(maximum_field)
        if not isinstance(minimum_value, (int, float)) or isinstance(
            minimum_value, bool
        ):
            continue
        if not isinstance(maximum_value, (int, float)) or isinstance(
            maximum_value, bool
        ):
            continue
        span = float(maximum_value) - float(minimum_value)
        if span <= 0.0:
            raise ToolArgumentError(
                f"{path}.{maximum_field} must be greater than {path}.{minimum_field}"
            )
        if span > float(maximum_span) + 1e-12:
            raise ToolArgumentError(
                f"{path} region span {maximum_field}-{minimum_field} must be <= {maximum_span}"
            )
