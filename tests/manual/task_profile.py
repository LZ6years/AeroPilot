"""Opt-in diagnostic timing for an entire task, without changing execution policy.

Hooks exist only inside a supervised experiment process. Spans use a monotonic
clock and thread-local nesting; background disk work is never added to main
thread wall time. Begin records survive native process exits. Arguments, model
content and credentials are deliberately absent from this performance sidecar.
"""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import json
from pathlib import Path
import threading
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def synchronize_devices(devices):
    """Diagnostic boundary: GPU spans must measure completion, not enqueue time."""
    cuda_devices = [device for device in devices if device.startswith("cuda:")]
    if cuda_devices:
        torch = sys.modules["torch"]
        for device in cuda_devices:
            torch.cuda.synchronize(device)


class TaskProfile:
    """Persist paired spans; each thread owns its own current parent/decision."""

    def __init__(self, path, *, clock=time.perf_counter):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.parent = ContextVar("profile_parent", default=None)
        self.decision = ContextVar("profile_decision", default=None)
        self.call_id = ContextVar("profile_call", default=None)
        self.lock = threading.Lock()
        self.sequence = 0

    def emit(self, record):
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    @contextmanager
    def span(self, name):
        with self.lock:
            self.sequence += 1
            identifier = self.sequence
        record = {"id": identifier, "parent": self.parent.get(), "name": name,
                  "thread": threading.get_ident(), "decision": self.decision.get(), "tool_call_id": self.call_id.get()}
        self.emit({**record, "event": "begin", "time": self.clock()})
        token = self.parent.set(identifier)
        error = {}
        try:
            yield
        except BaseException as exc:
            error = {"error_type": type(exc).__name__}
            raise
        finally:
            ended = self.clock()
            self.parent.reset(token)
            self.emit({**record, "event": "end", "time": ended, **error})

    def wrap(self, stack, owner, attribute, name, *, before=None, after=None):
        """Preserve descriptors, return values and exceptions, including failures."""
        original = getattr(owner, attribute)

        @wraps(original)
        def measured(*args, **kwargs):
            if before:
                before(args, kwargs)
            label = name(args, kwargs) if callable(name) else name
            with self.span(label):
                result = original(*args, **kwargs)
                if after:
                    after(args, result)
                return result

        replacement = staticmethod(measured) if isinstance(
            inspect.getattr_static(owner, attribute), staticmethod) else measured
        stack.enter_context(patch.object(owner, attribute, replacement))

    def inference_event(self, event, payload):
        """Per-stage allocator readings are process values, not exclusive ownership."""
        row = {"event": event, "time": self.clock(), "thread": threading.get_ident(),
               "decision": self.decision.get(), "tool_call_id": self.call_id.get(), **payload}
        torch = sys.modules.get("torch")
        if torch is not None and str(payload.get("device", "")).startswith("cuda:") and event != "device_wait_started":
            try:
                device = payload["device"]
                if event == "device_execution_started":
                    torch.cuda.reset_peak_memory_stats(device)
                row["torch_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(device),
                                       "reserved_bytes": torch.cuda.memory_reserved(device),
                                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            except Exception as exc:
                row["memory_unavailable"] = type(exc).__name__
        self.emit(row)


@contextmanager
def transport_hooks(profile):
    """Time actual HTTP attempts and retry sleeps, scoped to the LLM transport.

Use local module facades so unrelated urllib callers and simulator sleeps are
untouched. The request/response bytes, timeout and original retry policy pass
through verbatim; the response read stays inside the HTTP span.
    """
    from src.llm import transport
    original_open = transport.urllib.request.urlopen

    @contextmanager
    def measured_open(*args, **kwargs):
        with profile.span("model.http_attempt"):
            with original_open(*args, **kwargs) as response:
                yield response

    with ExitStack() as stack:
        request = SimpleNamespace(Request=transport.urllib.request.Request, urlopen=measured_open)
        stack.enter_context(patch.object(transport, "urllib", SimpleNamespace(
            request=request, error=transport.urllib.error)))
        clock = SimpleNamespace(sleep=transport.time.sleep)
        profile.wrap(stack, clock, "sleep", "model.retry_wait")
        stack.enter_context(patch.object(transport, "time", clock))
        yield


@contextmanager
def install_profile(profile):
    """Install observational hooks after imports, only for this experiment."""
    from src.agent.planning.model_action_planner import ModelActionPlanner
    from src.llm.client import DefaultModelClient
    from src.memory.stores.event_store import MemoryEventWriter
    from src.runtime.control.stabilization import UavStabilizer
    from src.runtime.logging.async_run_logger import AsyncRunLogger
    from src.runtime.runners import task_runtime
    from src.tools.dispatcher import ToolDispatcher
    from src.runtime.control.read_only_executor import ReadOnlyBatchExecutor
    from src.perception.inference_resources import inference_observer
    from src.memory.context.builder import ModelContextBuilder
    from src.runtime.task.loop import InstructionLoop
    from src.runtime.task.physical_window import PhysicalWindowExecutor
    from src.runtime.task.scene_session import SceneSessionService
    from src.runtime.task.turn_executor import TurnExecutor
    from src.runtime.task import turn_executor
    from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController
    from plugins.capabilities.visual_grounding.locate.scripts import service
    from plugins.capabilities.visual_grounding.locate.scripts.backends.grounding_dino import GroundingDinoBackend
    from plugins.capabilities.visual_grounding.locate.scripts.backends.sam import SamBackend

    with ExitStack() as stack:
        stack.enter_context(transport_hooks(profile))
        stack.enter_context(inference_observer(profile.inference_event))
        original_invoke = ReadOnlyBatchExecutor._invoke
        def measured_invoke(executor, job):
            token = profile.call_id.set(job.call_id)
            try:
                with profile.span("runtime.read_job"):
                    return original_invoke(executor, job)
            finally:
                profile.call_id.reset(token)
        stack.enter_context(patch.object(ReadOnlyBatchExecutor, "_invoke", measured_invoke))
        profile.wrap(stack, ModelActionPlanner, "plan_turn", "model.plan_turn",
                     before=lambda a, k: profile.decision.set(k.get("decision_index")))
        profile.wrap(stack, DefaultModelClient, "request", "model.request")
        profile.wrap(stack, ModelContextBuilder, "build_checkpoint", "runtime.context")
        profile.wrap(stack, SceneSessionService, "acquire", "scene.acquire")
        profile.wrap(stack, TravelUavScenarioController, "prepare_episode", "scene.target_setup")
        for method, label in (
            ("run_task", "runtime.task"),
            ("_prepare_scene_for_task", "scene.prepare"),
            ("_apply_initial_pose", "scene.initial_pose"),
            ("_record_observation", "runtime.observation"),
            ("_capture_instruction_exchange", "audit.exchange"),
            ("_write_state", "audit.state"), ("_run_shutdown", "scene.shutdown"),
        ):
            profile.wrap(stack, task_runtime.TaskRuntime, method, label)
        profile.wrap(stack, InstructionLoop, "run", "runtime.decisions")
        profile.wrap(
            stack,
            PhysicalWindowExecutor,
            "execute",
            "runtime.physical_window",
        )
        profile.wrap(
            stack,
            TurnExecutor,
            "_execute_native_read_calls",
            "runtime.read_calls",
        )
        profile.wrap(stack, turn_executor, "execute_builtin_control", "runtime.control")
        profile.wrap(stack, UavStabilizer, "wait_after_step", "runtime.stabilize")
        profile.wrap(stack, ToolDispatcher, "call",
                     lambda a, k: "tool." + str(k.get("tool_name", a[3] if len(a) > 3 else "unknown")))
        profile.wrap(stack, ToolDispatcher, "observe", "sensor.observe")
        for method in ("_append_with_id",):
            profile.wrap(stack, MemoryEventWriter, method, "disk.memory")
        for method in ("_append_jsonl", "_append_text", "_write_text", "_write_bytes"):
            profile.wrap(stack, AsyncRunLogger, method, "disk.logger")
        profile.wrap(stack, AsyncRunLogger, "close", "audit.flush_wait")
        for method, label in (("load_rgb_frame", "vision.image_read"),
                              ("load_metric_depth", "vision.depth_read"),
                              ("estimate_candidate_depth", "vision.depth_estimate"),
                              ("_write_candidate_evidence", "vision.artifact_write")):
            profile.wrap(stack, service, method, label)
        wrapped_models = set()
        model_hook_lock = threading.Lock()
        def forward_hook(label):
            def after_load(args, result):
                model = result[1]
                with model_hook_lock:
                    if id(model) not in wrapped_models:
                        wrapped_models.add(id(model))
                        devices = sorted({str(p.device) for p in model.parameters()})
                        profile.emit({"event": "model_loaded", "backend": label, "time": profile.clock(),
                                      "thread": threading.get_ident(), "tool_call_id": profile.call_id.get(),
                                      "device": args[0].device,
                                      "parameter_devices": devices,
                                      "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                                      "model_commit": getattr(getattr(model, "config", None), "_commit_hash", None)})
                        profile.wrap(stack, model, "forward", "vision." + label + ".forward",
                                     before=lambda a, k: synchronize_devices(devices),
                                     after=lambda a, result: synchronize_devices(devices))
            return after_load
        for cls, label, operation in ((GroundingDinoBackend, "grounding", "locate"),
                                       (SamBackend, "sam", "segment")):
            for method, suffix in (("_torch", "torch_import"), ("_load", "load"), (operation, "pipeline")):
                profile.wrap(stack, cls, method, "vision." + label + "." + suffix,
                             after=forward_hook(label) if method == "_load" else None)
        yield


def interval_union(intervals):
    """Union lengths, not sum, so overlapping intervals never inflate coverage."""
    total, boundary = 0.0, float("-inf")
    for start, end in sorted(intervals):
        total += max(0., end - max(start, boundary))
        boundary = max(boundary, end)
    return total


def summarize(path):
    """Keep incomplete spans and missing timing explicit; aggregate per thread."""
    path = Path(path)
    if not path.exists():
        return {"available": False, "threads": {}, "incomplete": [], "spans": []}
    starts, ends, malformed, device_events = {}, {}, 0, []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1  # A native exit may leave the last write incomplete.
            continue
        if row["event"] not in {"begin", "end"}:
            device_events.append(row)
            continue
        (starts if row["event"] == "begin" else ends)[row["id"]] = row
    complete = [{**row, "ended": ends[key]["time"], "error_type": ends[key].get("error_type")}
                for key, row in starts.items() if key in ends]
    children = defaultdict(list)
    for row in complete:
        children[(row["thread"], row["parent"])].append(row)
    threads, durations = {}, defaultdict(list)
    unfinished_parents = {(row["thread"], row["parent"]) for key, row in starts.items() if key not in ends}
    for row in complete:
        duration = row["ended"] - row["time"]
        intervals = [(max(row["time"], child["time"]), min(row["ended"], child["ended"]))
                     for child in children[(row["thread"], row["id"])]]
        row["inclusive_sec"] = duration
        row["exclusive_sec"] = max(0., duration - interval_union(intervals))
        row["exclusive_complete"] = (row["thread"], row["id"]) not in unfinished_parents
        stages = threads.setdefault(str(row["thread"]), {})
        stage = stages.setdefault(row["name"], {"count": 0, "inclusive_sec": 0., "exclusive_sec": 0., "errors": 0})
        stage["count"] += 1
        stage["inclusive_sec"] += duration
        stage["exclusive_sec"] += row["exclusive_sec"]
        stage["errors"] += bool(row["error_type"])
        durations[(str(row["thread"]), row["name"])].append(duration)
    for (thread, name), values in durations.items():
        # Linear percentiles agree with the previous NumPy report, without a
        # new runtime dependency. First-call costs remain visible separately.
        ordered = sorted(values)
        def percentile(q):
            index = (len(ordered) - 1) * q
            lower = int(index)
            return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)
        threads[thread][name].update(p50_sec=percentile(.5), p95_sec=percentile(.95), first_sec=values[0])
    return {"available": True, "threads": threads, "spans": complete, "device_events": device_events,
            "malformed_lines": malformed, "incomplete": [row for key, row in starts.items() if key not in ends],
            "accounting": "Inclusive parents contain children. Sum exclusive stages within one thread only. "
                          "Worker disk time overlaps the main thread; missing/unclosed spans are not zero. "
                          "Decision tags start at planner entry; context preparation before it has the previous tag."}
