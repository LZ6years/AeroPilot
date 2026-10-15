"""Retrieve historical place evidence without creating a new observation."""
from plugins.capabilities.place_memory.store import PlaceStore, annotation_status, available_photo_ids, historical_photo
from src.memory.context.media_refs import image_media_ref
from src.plugins.contracts.skill import SkillCapabilityResult


def handle(payload):
    context = payload["runtime_context"]
    store = PlaceStore(context["run_dir"], context["instruction_id"])
    graph = store.load()
    node = next((n for n in graph["nodes"] if n["id"] == payload["args"]["place_id"]), None)
    if node is None:
        raise ValueError("unknown place id in this task")
    inputs = tuple(image_media_ref(store.photo_path(p["ref"]), run_dir=store.run_dir,
        observation_id=p["observation_id"], frame_id=p["frame_id"], role="historical_place_photo",
        usage=f"HISTORY at {node['id']}, observed {p['timestamp']}; not a new CURRENT view.").to_model_input()
        for p in node["photos"])
    data = {"place_id": node["id"], "summary": node["summary"], "landmarks": node["landmarks"],
            "visits": node["visits"], "actions": node["actions"][-8:],
            "photos": [historical_photo(graph, p) for p in node["photos"]],
            "evidence_refs": node.get("evidence_refs", []),
            "available_photo_observation_ids": available_photo_ids(graph, node),
            "evidence_origin": "historical_memory", **annotation_status(node)}
    return SkillCapabilityResult("Historical place memory retrieved.", data, (), inputs)
