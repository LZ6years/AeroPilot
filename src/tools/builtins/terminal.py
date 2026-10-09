"""Explicit mission termination contracts, separate from advisory plan status."""

from src.tools.schemas import ToolParameter, ToolRuntimePolicy, ToolSpec


def terminal_tools() -> list[ToolSpec]:
    tools = []
    for name in ("report_completed", "report_blocked"):
        parameters = (
            ToolParameter("reason", "string", "Concise explanation of the task outcome.", True, {"minLength": 1}),
            ToolParameter("evidence_refs", "array", "Use current-instruction observation IDs, or tool_call_id/result_ref "
                          "from successful environmental tool results. Every reference must be valid. "
                          "Image frame IDs and image paths by themselves are not accepted references; "
                          "plan and knowledge references are not environmental evidence.", True,
                          {"minItems": 1, "uniqueItems": True, "items": {"type": "string", "minLength": 1}}),
        )
        if name == "report_blocked":
            parameters += (
                ToolParameter("suggested_user_action", "string", "Optional action the user can take to resolve the blocker."),
            )
        tools.append(ToolSpec(
            name=f"runtime.{name}", wire_name=f"runtime_{name}", kind="runtime_control",
            description=(
                "Report the original instruction as completed. Applicable when environmental evidence supports "
                "all requested outcomes. Returns the accepted terminal state or rejection feedback; Runtime "
                "handles final stabilization and cleanup. Plan status alone is not completion evidence. "
                if name == "report_completed" else
                "Report the instruction as incomplete only after distinct useful actions or viewpoint changes "
                "still cannot provide a path forward. Temporary non-visibility or one unsuccessful observation "
                "is not sufficient. Returns the accepted terminal state or rejection feedback; Runtime handles "
                "final stabilization and cleanup. This ends the instruction without claiming that its requested "
                "outcomes were achieved."
            ),
            runtime=ToolRuntimePolicy("runtime_control", "immediate", "single", True),
            parameters=parameters,
        ))
    return tools
