"""Optional plan storage contract; planning preference belongs to this tool."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def planning_tool() -> ToolSpec:
    return ToolSpec(
        name="runtime.update_plan", wire_name="runtime_update_plan", kind="runtime_control",
        description=("Create or revise an optional execution plan when a task has a long or difficult route, "
                     "or multiple ordered stages whose progress must be retained across decisions. Use it to "
                     "record completed progress, the current and remaining stages, and the next intended "
                     "objective. Revise it when the recorded progress or route approach changes. Submit the "
                     "complete plan snapshot using stable step IDs. Returns the saved plan and revision. The "
                     "plan does not execute actions or establish environmental facts or task completion."),
        runtime=ToolRuntimePolicy("runtime_control", "immediate", "sidecar", True),
        parameters=(
            ToolParameter("explanation", "string", "Brief reason for creating the plan or changing its recorded progress or approach.",
                          True, {"minLength": 1}),
            ToolParameter("steps", "array", "Complete ordered task plan snapshot with persistent step identities.", True, {
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["id", "description", "status", "evidence_refs"],
                          "properties": {
                              "id": {"type": "string", "minLength": 1, "description": "Stable step identifier."},
                              "description": {"type": "string", "minLength": 1, "description": "User-required outcome with its order and conditions."},
                              "status": {"type": "string", "description": "Current progress state for this outcome.",
                                         "enum": ["pending", "in_progress", "completed", "cancelled"]},
                              "evidence_refs": {"type": "array", "description": "Prior observation IDs or successful ToolResult/action references supporting this step. Required and non-empty when status is completed; use an empty array when no evidence is claimed.",
                                                "items": {"type": "string", "minLength": 1},
                                                "uniqueItems": True}}}}),
        ),
    )
