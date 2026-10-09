"""Knowledge-reading contract; catalog and content remain plugin-owned."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def knowledge_tool() -> ToolSpec:
    return ToolSpec(
        name="knowledge.load", wire_name="knowledge_load", kind="knowledge",
        description=("Read an enabled knowledge entry and its declared references when its methods or "
                     "explanations help the current decision. Reuse guidance already loaded. Returns newly "
                     "requested body text and references incrementally; previously delivered text is not resent. "
                     "The catalog is a directory, not loaded guidance. Knowledge provides flight methods; "
                     "environmental observations and measurements come from the corresponding tools."),
        runtime=ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True),
        parameters=(
            ToolParameter("name", "string", "Exact enabled entry name from the knowledge catalog.", True,
                          {"minLength": 1}),
            ToolParameter("reference_paths", "array", "Optional paths from the entry's returned reference manifest.",
                          constraints={"uniqueItems": True, "items": {"type": "string", "minLength": 1}}),
        ),
    )
