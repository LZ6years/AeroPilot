"""Optional plan storage contract; planning preference belongs to this tool."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def planning_tool() -> ToolSpec:
    return ToolSpec(
        name="runtime.update_plan", wire_name="runtime_update_plan", kind="runtime_control",
        description=("Create or revise a task plan using stable step IDs. Use before the first physical action "
                     "for multiple ordered subgoals, landmark-dependent actions or progress across observations; "
                     "a simple single-step task can act directly. Update when evidence establishes progress or "
                     "changes the approach, without rewriting unchanged plans. Returns the saved plan "
                     "and its revision. A plan records intended work and progress; it does not execute flight "
                     "or provide environmental completion evidence."),
        runtime=ToolRuntimePolicy("runtime_control", "immediate", "single", True),
        parameters=(
            ToolParameter("explanation", "string", "Why this plan or evidence-based revision is needed.",
                          True, {"minLength": 1}),
            ToolParameter("steps", "array", "Ordered task steps with persistent identities.", True, {
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["id", "description", "status"],
                          "properties": {
                              "id": {"type": "string", "minLength": 1, "description": "Stable step identifier."},
                              "description": {"type": "string", "minLength": 1, "description": "Required outcome."},
                              "status": {"type": "string", "description": "Progress supported by observed results.",
                                         "enum": ["pending", "in_progress", "completed", "cancelled"]}}}}),
        ),
    )
