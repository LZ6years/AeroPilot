"""Single-thread, opt-in CUDA Event stage measurements without CUPTI.

Event intervals are stream elapsed time, including GPU scheduling and host
submission gaps, NOT sums of kernel compute. Extra per-stage synchronization
perturbs execution. Never mix these batches into uninstrumented benchmarks.
"""
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import json
import time
from unittest.mock import patch

from tests.manual.task_profile import TaskProfile


def summarize_rows(rows):
    groups = {}
    for row in rows:
        key = row['stage']
        group = groups.setdefault(key, {'count': 0, 'host_sec': 0., 'stream_sec': 0., 'bytes': 0})
        group['count'] += 1
        for field in ('host_sec', 'stream_sec', 'bytes'):
            group[field] += row.get(field, 0)
    return groups


@contextmanager
def capture_events(directory, *, sam_modules=False):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Event probe requires exactly one CUDA device')
    current = ContextVar('event_probe_stage', default='outside')
    rows = []
    original_span = TaskProfile.span
    original_sync = torch.cuda.synchronize

    @contextmanager
    def timed(stage, **metadata):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
        start.record(stream)
        began = time.perf_counter()
        try:
            yield
        finally:
            end.record(stream)
            end.synchronize()
            rows.append({'stage': stage, 'host_sec': time.perf_counter() - began,
                         'stream_sec': start.elapsed_time(end) / 1000., **metadata})

    @contextmanager
    def span(self, name):
        measured = name.startswith('vision.') and name.endswith('.forward')
        token = current.set(name) if name.startswith('vision.') else None
        try:
            if measured:
                with timed(name), original_span(self, name):
                    yield
            else:
                with original_span(self, name):
                    yield
        finally:
            if token is not None:
                current.reset(token)

    def sync(*args, **kwargs):
        began = time.perf_counter()
        try:
            return original_sync(*args, **kwargs)
        finally:
            rows.append({'stage': current.get() + '.explicit_sync', 'host_sec': time.perf_counter() - began})

    def transfer(original, operation):
        def measured(tensor, *args, **kwargs):
            stage = current.get()
            # Do not add synchronization to every internal model operation.
            if not stage.endswith('.pipeline'):
                return original(tensor, *args, **kwargs)
            source = str(tensor.device)
            if operation == 'cpu':
                target = 'cpu'
            else:
                requested = kwargs.get('device', args[0] if args else None)
                target = str(getattr(requested, 'device', requested))
            if 'cuda' not in source and 'cuda' not in target:
                return original(tensor, *args, **kwargs)
            with timed(stage + '.transfer.' + source + '_to_' + target,
                       bytes=tensor.numel() * tensor.element_size(), operation=operation):
                return original(tensor, *args, **kwargs)
        return measured

    try:
        with ExitStack() as stack:
            if sam_modules:
                from transformers.models.sam import modeling_sam
                for class_name, label in (('SamVisionEncoder', 'image_encoder'),
                                          ('SamPromptEncoder', 'prompt_encoder'),
                                          ('SamMaskDecoder', 'mask_decoder')):
                    cls = getattr(modeling_sam, class_name)
                    original = cls.forward
                    def measured(module, *args, _original=original, _label=label, **kwargs):
                        with timed('sam.module.' + _label):
                            return _original(module, *args, **kwargs)
                    stack.enter_context(patch.object(cls, 'forward', measured))
            stack.enter_context(patch.object(TaskProfile, 'span', span))
            stack.enter_context(patch.object(torch.cuda, 'synchronize', sync))
            stack.enter_context(patch.object(torch.Tensor, 'to', transfer(torch.Tensor.to, 'to')))
            stack.enter_context(patch.object(torch.Tensor, 'cpu', transfer(torch.Tensor.cpu, 'cpu')))
            yield
    finally:
        record = {'rows': rows, 'groups': summarize_rows(rows),
                  'limitations': 'Stream elapsed includes scheduling/host gaps; not kernel-only. '
                  'Host sync is nested in forward. Transfers cover pipeline-level Tensor.to/cpu only; '
                  'not model-internal copies. Stage synchronization perturbs timing.'}
        (directory / 'cuda_events.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
