"""Short-horizon action planning."""

from importlib import import_module

_EXPORTS = {
    "ModelActionPlanner": "model_action_planner",
    "ModelActionPlannerConfigError": "model_action_planner",
    "PlannedStep": "schemas",
    "TaskPlan": "task_plan",
    "TaskPlanStep": "task_plan",
}

__all__ = ["ModelActionPlanner", "ModelActionPlannerConfigError", "PlannedStep", "TaskPlan", "TaskPlanStep"]


def __getattr__(name):
    """Loading pure input/plan helpers must not eagerly load the Planner."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
