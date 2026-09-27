"""Process-wide device gates shared by perception backends.

This module never imports Torch, loads a model, reads mission state or chooses
a device. Backends supply their resolved device identity. Diagnostic callbacks
are optional and context-local; limits apply even when no observer is installed.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import threading
import time

_registry_lock = threading.Lock()
_gates = {}
_held = threading.local()
_observer = ContextVar("inference_observer", default=None)


@contextmanager
def inference_observer(callback):
    """Observe waits/execution in this context (copied into read workers)."""
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def inference_slot(device, operation):
    """Serialize actual device work across backend instances and model kinds.

Nested backend load checks use the outer slot and do not count as additional
active inference. Device strings must already resolve aliases such as cuda.
    """
    device = str(device)
    if device == "cuda":
        raise ValueError("Device gate requires an explicit CUDA device index")
    with _registry_lock:
        gate = _gates.setdefault(device, threading.Lock())
    held = getattr(_held, "devices", set())
    if device in held:
        yield
        return
    started = time.perf_counter()
    observer = _observer.get()
    if observer:
        observer("device_wait_started", {"device": device, "operation": operation})
    gate.acquire()
    acquired = time.perf_counter()
    _held.devices = held | {device}
    error = None
    try:
        if observer:
            observer("device_execution_started", {"device": device, "operation": operation,
                                                 "wait_sec": acquired - started})
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - acquired
        _held.devices = held
        # Emit completion while owning the slot, preserving diagnostic order.
        try:
            if observer:
                observer("device_execution_finished", {"device": device, "operation": operation,
                                                       "elapsed_sec": elapsed, "error_type": error})
        finally:
            gate.release()


def serialized_inference(method):
    """Guard an entire backend call, including CPU materialization of outputs."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with inference_slot(self.device, type(self).__name__ + "." + method.__name__):
            return method(self, *args, **kwargs)
    return guarded
