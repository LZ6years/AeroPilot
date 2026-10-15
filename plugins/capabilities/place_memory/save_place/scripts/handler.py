"""Persist task-local place annotations through the generic capability API."""
from plugins.capabilities.place_memory.store import PlaceStore, place_summary, InvalidPlacePhotos
from plugins.capabilities.place_memory.render import render_map
from src.plugins.contracts.skill import SkillCapabilityResult, SkillCapabilityFailure


def handle(payload):
    context = payload["runtime_context"]
    store = PlaceStore(context["run_dir"], context["instruction_id"])
    try:
        graph = store.annotate(payload["args"], context)
    except InvalidPlacePhotos as exc:
        raise SkillCapabilityFailure(str(exc), data=exc.data) from exc
    media, page = render_map(store, graph)
    node = next(n for n in graph["nodes"] if n["id"] == payload["args"]["place_id"])
    current = next(n for n in graph["nodes"] if n["id"] == graph["current_place_id"])
    return SkillCapabilityResult("Place annotation saved; latest map attached.",
        {"place_id": node["id"], **place_summary(graph, node),
         "current_place": place_summary(graph, current), "evidence_origin": "historical_memory"},
        (str(page), str(store.path)), (media,))
