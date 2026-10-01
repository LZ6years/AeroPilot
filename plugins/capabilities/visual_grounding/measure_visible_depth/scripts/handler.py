"""JSON-configured entrypoint for the visual-grounding capability."""

from __future__ import annotations

from src.plugins.contracts.skill import SkillCapabilityResult

from plugins.capabilities.visual_grounding.measure_visible_depth.scripts.service import service_from_settings


def handle(payload: dict) -> SkillCapabilityResult:
    args = dict(payload.get("args", {}))
    runtime_context = dict(payload.get("runtime_context", {}))
    settings = dict(payload.get("settings", {}))
    observation = dict(runtime_context.get("latest_observation", {}))
    result = service_from_settings(settings).measure_visible_depth(
        observation=observation,
        observation_id=str(args.get("observation_id") or ""),
        query=str(args.get("query") or "").strip(),
        run_dir=str(runtime_context.get("run_dir") or "") or None,
        tool_call_id=str(runtime_context.get("tool_call_id") or "") or None,
    )
    candidates = result["data"]["candidates"]
    message = (
        f"Measured visible depth for {len(candidates)} candidate(s)"
        + f"; {sum(c.get('distance_status') == 'ok' and c.get('distance_m') is not None for c in candidates)} valid distance estimate(s). Check distance_status; execution success is not measurement validity."
    )
    if not candidates:
        message = ("Visible-depth measurement found no matching measurable candidate. "
                   "This is not a search result and gives no evidence that the object is absent.")
    return SkillCapabilityResult(
        message=message,
        data=result["data"],
        artifact_refs=tuple(result["artifact_refs"]),
        model_inputs=tuple(result["model_inputs"]),
    )
