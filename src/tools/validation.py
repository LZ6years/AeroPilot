"""Small JSON-Schema subset validator for native tool arguments.

FlyAgent intentionally supports only the schema features emitted by its tool
catalog. Keeping this validator local avoids a new runtime dependency while
still enforcing the exact contract before any side effect occurs.
"""

from __future__ import annotations

from typing import Any


class ToolArgumentError(ValueError):
    """Raised when native tool arguments violate their ToolSpec schema."""


def validate_tool_args(args: dict[str, Any], schema: dict[str, Any], path: str = "args") -> None:
    _validate(args, schema, path)


def _validate(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
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
    elif expected == "string" and not isinstance(value, str):
        raise ToolArgumentError(f"{path} must be a string")
    elif expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ToolArgumentError(f"{path} must be a number")
    elif expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ToolArgumentError(f"{path} must be an integer")
    elif expected == "boolean" and not isinstance(value, bool):
        raise ToolArgumentError(f"{path} must be a boolean")

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
