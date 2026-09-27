"""Opt-in offline CUDA timeline; CPU-only traces are explicitly rejected.

The profiler changes measurement overhead. Traced batches must be excluded
from ordinary warm statistics. No precision, allocator or device policy changes.
"""
from collections import defaultdict
from contextlib import contextmanager
import json
from unittest.mock import patch

from tests.manual.task_profile import TaskProfile, interval_union


def require_cuda_profiler(directory):
    """Fail before loading visual models if this build cannot trace CUDA."""
    import torch
    activities = torch.profiler.supported_activities()
    record = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
              'kineto_available': torch.profiler.kineto_available(),
              'supported_activities': sorted(str(a) for a in activities)}
    (directory / 'profiler_environment.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not record['cuda_available'] or torch.profiler.ProfilerActivity.CUDA not in activities:
        raise ValueError('Current PyTorch profiler cannot capture CUDA activity; no CPU-only substitute')
    return record


def summarize_trace(path):
    events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    kernels = [e for e in complete if e.get('cat') == 'kernel']
    if not kernels:
        raise ValueError('No CUDA kernel events captured; CPU trace is insufficient')
    result = {}
    for name, rows in (
        ('kernels', kernels),
        ('copies', [e for e in complete if e.get('cat') == 'gpu_memcpy']),
        ('memset', [e for e in complete if e.get('cat') == 'gpu_memset']),
        ('host_sync', [e for e in complete if e.get('cat') in ('cuda_runtime', 'cuda_driver') and 'Synchronize' in e.get('name', '')]),
    ):
        totals = defaultdict(float)
        for e in rows:
            totals[e['name']] += e['dur'] / 1e6
        result[name] = {'count': len(rows), 'sum_sec': sum(e['dur'] for e in rows) / 1e6,
                        'top': sorted(totals.items(), key=lambda pair: pair[1], reverse=True)[:15]}
    device = [e for e in complete if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    result['device_active_union_sec'] = interval_union([(e['ts'], e['ts'] + e['dur']) for e in device]) / 1e6
    result['device_span_sec'] = (max(e['ts'] + e['dur'] for e in device) - min(e['ts'] for e in device)) / 1e6
    result['device_gap_sec'] = result['device_span_sec'] - result['device_active_union_sec']
    result['annotations'] = [{'name': e['name'], 'wall_sec': e['dur'] / 1e6} for e in complete
                             if e.get('cat') == 'user_annotation' and e['name'].startswith('vision.')]
    result['limitations'] = 'Single-device timeline; host waits overlap device activity. Do not add categories. Gaps do not prove paging or external contention.'
    return result


@contextmanager
def capture_batch(directory):
    import torch
    require_cuda_profiler(directory)
    original = TaskProfile.span

    @contextmanager
    def annotated(self, name):
        with torch.profiler.record_function(name), original(self, name):
            yield

    trace = directory / 'cuda_trace.json'
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            with patch.object(TaskProfile, 'span', annotated):
                yield
        profiler.export_chrome_trace(str(trace))
        summary = summarize_trace(trace)
        (directory / 'cuda_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    except BaseException as exc:
        (directory / 'cuda_error.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc)}), encoding='utf-8')
        raise
