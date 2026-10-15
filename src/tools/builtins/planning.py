"""Optional plan storage contract; planning preference belongs to this tool."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def planning_tool() -> ToolSpec:
    # The shared validator checks field shape; TaskPlan owns operation semantics
    # and merged-state invariants, before Runtime allows any side effects.
    step_properties = {
        "id": {"type": "string", "minLength": 1, "description": "Stable step identifier."},
        "description": {"type": ["string", "null"], "minLength": 1,
                        "description": "User-required outcome with its order and conditions."},
        "status": {"type": ["string", "null"], "description": "Model-recorded progress; null retains the previous value in patch mode.",
                   "enum": ["pending", "in_progress", "completed", "cancelled", None]},
    }
    return ToolSpec(
        name="runtime.update_plan", wire_name="runtime_update_plan", kind="runtime_control",
        description=("Create or revise an optional execution plan when a task has a long or difficult route, "
                     "or multiple ordered stages whose progress must be retained across decisions. Use it to "
                     "record completed progress, the current and remaining stages, and the next intended "
                     "objective. Update after a milestone, a corrected progress judgment, or a changed route "
                     "approach; unchanged progress needs no update. Use operation=replace for a complete "
                     "ordered snapshot (non-null description and status), or operation=patch to change existing "
                     "IDs (null retains a field; each item changes at least one field). All fields are required. "
                     "Keep stable step IDs; mark superseded steps cancelled instead of dropping them. At most one "
                     "step may be in_progress after applying all changes. Returns the saved plan and revision. The "
                     "plan does not execute actions or establish environmental facts or task completion."),
        runtime=ToolRuntimePolicy("runtime_control", "immediate", "sidecar", True),
        strict=True,
        parameters=(
            ToolParameter("operation", "string", "replace creates or reorders a complete plan; patch changes existing steps without reordering.",
                          True, {"enum": ["replace", "patch"]}),
            ToolParameter("explanation", ["string", "null"], "Brief reason; null preserves the previous explanation.",
                          True, {"minLength": 1}),
            ToolParameter("steps", "array", "Complete ordered snapshot for replace, or non-empty existing-ID changes for patch.", True, {
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["id", "description", "status"],
                          "properties": step_properties}}),
        ),
    )
