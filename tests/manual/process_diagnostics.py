"""Diagnostic-only stage tracing and subprocess supervision; never repairs model output."""

from functools import wraps
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def supervise(command, directory, timeout=None, *, monitor_gpu=False, gpu_sampler=None, windows_gpu_memory=False):
    """Keep logs and exit evidence in a parent that survives native child exits."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    children = {}
    timed_out = False
    with (directory/'stdout.log').open('w', encoding='utf-8') as stdout, \
            (directory/'stderr.log').open('w', encoding='utf-8') as stderr, \
            (directory/'resources.jsonl').open('w', encoding='utf-8') as resources, ExitStack() as diagnostics:
        if monitor_gpu:
            from tests.manual.gpu_resource_monitor import GpuResourceMonitor
            options = {} if gpu_sampler is None else {"sampler": gpu_sampler}
            diagnostics.enter_context(GpuResourceMonitor(directory / 'gpu_resources.jsonl', **options))
        if windows_gpu_memory:
            from tests.manual.windows_gpu_memory import WindowsGpuMemoryMonitor
            diagnostics.enter_context(WindowsGpuMemoryMonitor(directory))
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONFAULTHANDLER': '1'})
        try:
            root = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            root = None  # A native child can exit before the first sample.
        while process.poll() is None:
            try:
                if root is None:
                    break
                for child in root.children(recursive=True):
                    children[child.pid] = {'pid': child.pid, 'created': child.create_time(), 'command': child.cmdline()}
                resources.write(json.dumps({'elapsed': time.monotonic()-started, 'rss': root.memory_info().rss,
                                            'system_available': psutil.virtual_memory().available})+'\n')
                resources.flush()
            except psutil.Error:
                pass
            if timeout is not None and time.monotonic()-started > timeout:
                timed_out = True
                process.kill()
                break
            time.sleep(1)
        code = process.wait()
    result = {'exit_code': code, 'timed_out': timed_out, 'elapsed_seconds': time.monotonic()-started,
              'pid': process.pid, 'descendants': list(children.values())}
    write_json(directory/'process.json', result)
    return result


def install_grounding_trace(directory):
    """Instrument existing adapters in this child only; keep detection semantics intact."""
    from plugins.capabilities.visual_grounding.locate.scripts import service
    from plugins.capabilities.visual_grounding.locate.scripts.backends.grounding_dino import GroundingDinoBackend
    from plugins.capabilities.visual_grounding.locate.scripts.backends.sam import SamBackend

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ('torch', 'transformers', 'numpy', 'pillow', 'psutil'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(directory/'dependencies.json', versions)

    def event(stage, status, **extra):
        with (directory/'stages.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': time.time(), 'stage': stage, 'status': status,
                                    'rss': psutil.Process().memory_info().rss, **extra})+'\n')
            stream.flush()

    def wrap(owner, name, stage, static=False, after=None):
        original = getattr(owner, name)
        @wraps(original)
        def traced(*args, **kwargs):
            event(stage, 'begin')
            try:
                result = original(*args, **kwargs)
                if after:
                    after(args, result)
                event(stage, 'end')
                return result
            except BaseException as exc:
                event(stage, 'error', error_type=type(exc).__name__, message=str(exc))
                raise
        setattr(owner, name, staticmethod(traced) if static else traced)

    def instrument_model(stage):
        def after(args, result):
            instance = args[0]
            event(stage, 'device', device=instance.device)
            model = result[1]
            if not getattr(model, '_diagnostic_wrapped', False):
                wrap(model, 'forward', stage+'.inference')
                model._diagnostic_wrapped = True
        return after

    wrap(service, 'load_rgb_frame', 'image.read')
    wrap(service, 'load_metric_depth', 'depth.read')
    wrap(service, 'estimate_candidate_depth', 'depth.estimate')
    wrap(service, '_write_candidate_evidence', 'artifact.write')
    for cls, label, operation in [(GroundingDinoBackend, 'grounding', 'locate'), (SamBackend, 'sam', 'segment')]:
        wrap(cls, '_torch', label+'.torch_import', static=True)
        wrap(cls, '_load', label+'.load', after=instrument_model(label))
        wrap(cls, operation, label+'.pipeline')
