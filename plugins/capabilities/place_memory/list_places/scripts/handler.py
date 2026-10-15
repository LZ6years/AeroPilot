"""Read the saved-only directory through the generic capability API."""
from plugins.capabilities.place_memory.store import PlaceStore
from src.plugins.contracts.skill import SkillCapabilityResult


def handle(payload):
    context = payload["runtime_context"]
    store = PlaceStore(context["run_dir"], context["instruction_id"])
    data = store.list_saved_places(**payload["args"])
    return SkillCapabilityResult("Saved place directory retrieved.", data)
