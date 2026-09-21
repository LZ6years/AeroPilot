"""JSON-configured entrypoint for the visual-grounding capability."""

from __future__ import annotations

from src.plugins.contracts.skill import SkillCapabilityResult

from plugins.capabilities.visual_grounding.locate.scripts.service import service_from_settings


def handle(payload: dict) -> SkillCapabilityResult:
    args = dict(payload.get("args", {}))
    runtime_context = dict(payload.get("runtime_context", {}))
    settings = dict(payload.get("settings", {}))
    observation = dict(runtime_context.get("latest_observation", {}))
    result = service_from_settings(settings).locate(
        observation=observation,
        observation_id=str(args.get("observation_id") or ""),
        query=str(args.get("query") or "").strip(),
        include_distance=bool(args.get("include_distance")),
        run_dir=str(runtime_context.get("run_dir") or "") or None,
        tool_call_id=str(runtime_context.get("tool_call_id") or "") or None,
    )
    candidates = result["data"]["candidates"]
    with_distance = bool(args.get("include_distance"))
    message = (
        f"Located {len(candidates)} candidate(s)"
        + (" with optional robust metric depth." if with_distance else ".")
    )
    if not candidates:
        message = ("No sufficiently confident detection candidates were returned. "
                   "Whether the target is present remains uncertain.")
    return SkillCapabilityResult(
        message=message,
        data=result["data"],
        artifact_refs=tuple(result["artifact_refs"]),
        model_inputs=tuple(result["model_inputs"]),
    )
