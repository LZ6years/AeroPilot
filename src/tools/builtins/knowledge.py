"""Knowledge-reading contract; catalog and content remain plugin-owned."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def knowledge_tool() -> ToolSpec:
    return ToolSpec(
        name="knowledge.load", wire_name="knowledge_load", kind="knowledge",
        description=("Read an enabled knowledge entry and its declared references. Applicable when a task "
                     "needs the methods or explanations described by that catalog entry. When the next action "
                     "method is unclear and a relevant entry exists, load it before proceeding; reuse guidance "
                     "already loaded. Missing scene facts require observation or perception. Returns newly "
                     "requested body text and references incrementally; previously delivered text is not resent. "
                     "The catalog is a directory, not loaded guidance. Knowledge provides methods, not "
                     "current-scene observations or measurements, and does not execute actions."),
        runtime=ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True),
        parameters=(
            ToolParameter("name", "string", "Exact enabled entry name from the knowledge catalog.", True,
                          {"minLength": 1}),
            ToolParameter("reference_paths", "array", "Optional paths from the entry's returned reference manifest.",
                          constraints={"uniqueItems": True, "items": {"type": "string", "minLength": 1}}),
        ),
    )
