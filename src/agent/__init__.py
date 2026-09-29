"""Agent semantic planning package."""

from importlib import import_module

# Importing a generic helper must not initialize Context and PluginDiscovery.
# Exports resolve lazily so pure helpers do not create cross-layer cycles.
_EXPORTS = {
    "MissionState": "mission_state",
    "ModelActionPlanner": "planning.model_action_planner",
    "TaskPlan": "planning.task_plan",
    "TaskPlanStep": "planning.task_plan",
    "PlannedStep": "planning.schemas",
    "SafetySupervisor": "safety",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Resolve one public export without eager cross-layer imports."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
