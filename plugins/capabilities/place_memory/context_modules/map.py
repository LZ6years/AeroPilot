"""Map-slot contribution composing measured task evidence with plugin notes."""
from src.memory.context.modules.base import ContextModule
from src.memory.context.contracts import ContextFragment, ContextModuleResult, FragmentAudit
from plugins.capabilities.place_memory.store import PlaceStore, DEFAULTS
from plugins.capabilities.place_memory.render import render_map


class PluginContextModule(ContextModule):
    name = "map"
    priority = 82

    def __init__(self):
        self.options = dict(DEFAULTS)

    def configure(self, options):
        if set(options) - set(DEFAULTS):
            raise ValueError("unknown place map options")
        self.options.update(options)

    def build(self, sources, stage_context):
        memory = sources.flight_memory
        store = PlaceStore(memory.memory_reader.run_dir, memory.active_instruction_id, self.options)
        start = memory.observation_anchors.get("task_initial", {}).get("id")
        latest = memory.recent_observation_refs[-1]["id"]
        graph = store.update(start, latest)
        media, page = render_map(store, graph)
        return ContextModuleResult(ContextFragment(self.name, {}, (media,)), FragmentAudit(self.name, self.__class__.__name__,
            artifact_refs=(str(page), str(store.path))))

    def describe_output(self):
        return {"summary": "Task-local map of measured visited path and UAV heading; P-IDs mark UAV observation sites, not building coordinates. When short labels and available history cannot resolve a revisit or earlier judgment, inspect_place retrieves complete notes and historical views. Use list_places when a saved ID is unknown.",
                "semantic_evidence": ["Visited observation sites, measured route and model-authored place annotations; no hidden targets or inferred free space."]}
