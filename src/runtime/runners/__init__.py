"""Runtime execution flows that coordinate EnvClient calls."""

from importlib import import_module

__all__ = ["SmokeRuntime", "InteractiveTaskSession", "TaskRuntime"]

_EXPORTS = {
    "SmokeRuntime": "src.runtime.runners.smoke_runtime",
    "InteractiveTaskSession": "src.runtime.task.interactive_session",
    "TaskRuntime": "src.runtime.runners.task_runtime",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
