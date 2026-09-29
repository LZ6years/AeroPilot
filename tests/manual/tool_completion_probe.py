"""Offline fake-tool persistence/performance probe; no models or simulators.

Run with flyagent: -m tests.manual.tool_completion_probe --output <new directory>
The baseline is the pre-fix Runtime method from fbbec23, not a simulated writer.
Normal timings and allocation/telemetry diagnostics are separate runs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
import types

from src.env_client import ActionResult
from src.memory.context.media_refs import image_media_ref
from tests.test_tool_completion_persistence import setup_runtime, run


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def baseline_method():
    import src.runtime.runners.task_runtime as module
    source = subprocess.check_output(['git', 'show', 'fbbec23:src/runtime/runners/task_runtime.py'],
                                     text=True, encoding='utf-8')
    tree = ast.parse(source.lstrip('\ufeff'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TaskRuntime')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_native_read_calls')
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<fbbec23-runtime>', 'exec'), namespace)
    return namespace['_execute_native_read_calls'], hashlib.sha256(source.encode()).hexdigest()


def sample(directory, workers, baseline, diagnostics=False):
    import psutil
    runtime, state, scene, calls = setup_runtime(directory, workers)
    if baseline is not None:
        runtime._execute_native_read_calls = types.MethodType(baseline, runtime)
    returned_at = {}
    def dispatch(*args, **kwargs):
        # Constructed local payload, not a benchmark observation or inference.
        ident = kwargs['runtime_context']['tool_call_id']
        result = ActionResult(True, 'knowledge.load', 'constructed', {'values': list(range(100))})
        returned_at[ident] = time.perf_counter()
        return result
    runtime.tools.call = dispatch
    if not diagnostics:
        runtime.tool_telemetry.debug_event = lambda *a: None
    if diagnostics:
        tracemalloc.start()
    rss_before = psutil.Process().memory_info().rss if diagnostics else None
    began = time.perf_counter()
    results = run(runtime, state, scene, calls)
    elapsed = time.perf_counter() - began
    peak = tracemalloc.get_traced_memory()[1] if diagnostics else None
    if diagnostics:
        tracemalloc.stop()
    events = runtime.recorder.events
    request = None
    if diagnostics:
        # Capture the actual Planner payload after timing; no HTTP client runs.
        from tests.test_model_action_planner import StubPlanner, response_with_calls
        planner = StubPlanner(response_with_calls(('next', 'uav_turn_relative', {'yaw_deg': 10})))
        planner.plan_turn(runtime.agent_transcript, runtime.tools.registry)
        request = planner.payloads[0]
    latencies = [payload['monotonic_sec'] - returned_at[payload['tool_call_id']]
                 for name, payload in events if name == 'read_result_persisted']
    return {'seconds': elapsed, 'result': results,
            'files_bytes': sum(p.stat().st_size for p in directory.rglob('*.jsonl')),
            'receipt_bytes': (directory / 'memory/tool_completions.jsonl').stat().st_size if
                (directory / 'memory/tool_completions.jsonl').exists() else 0,
            'received_to_saved_seconds': latencies, 'python_peak_bytes': peak,
            'rss_before_bytes': rss_before,
            'rss_after_bytes': psutil.Process().memory_info().rss if diagnostics else None,
            'events': events if diagnostics else None, 'request': request}


def crash_child(directory):
    from PIL import Image
    runtime, state, scene, calls = setup_runtime(directory, 4)
    image = directory / 'evidence.png'
    Image.new('RGB', (96, 64), 'orange').save(image)
    persisted = threading.Event()
    original = runtime.memory_writer.write_tool_completion
    def save(**kwargs):
        result = original(**kwargs)
        persisted.set()
        return result
    runtime.memory_writer.write_tool_completion = save
    def dispatch(*args, **kwargs):
        if args[3]['knowledge_id'] == 'a':
            if not persisted.wait(10):
                raise RuntimeError('completion not saved')
            os._exit(23)  # Intentional native-style exit; parent must survive.
        return ActionResult(True, 'knowledge.load', 'synthetic image evidence', {
            '_flyagent_model_inputs': [image_media_ref(
                image,
                observation_id='constructed_tool_completion',
                frame_id='constructed_evidence',
                role='tool_evidence',
            ).to_model_input()]})
    runtime.tools.call = dispatch
    run(runtime, state, scene, calls)
    raise RuntimeError('expected abnormal exit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    expected = Path('F:/conda_env/flyagent/python.exe').resolve()
    if Path(sys.executable).resolve() != expected:
        raise RuntimeError(f'Use {expected}; got {sys.executable}')
    if args.child:
        crash_child(args.output)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base, base_hash = baseline_method()
    files = ['src/runtime/control/read_only_executor.py', 'src/runtime/runners/task_runtime.py',
             'src/memory/stores/event_store.py', 'src/runtime/middleware/tool_telemetry.py']
    write(root / 'manifest.json', {'python': sys.executable, 'baseline': 'fbbec23',
        'baseline_source_sha256': base_hash, 'input': 'two constructed local read calls',
        'source_hashes': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        'config': {'workers': [1, 4], 'production_default_changed': False},
        'status': 'running'})
    rows, summaries = [], []
    for workers in (1, 4):
        for iteration in range(31):
            pair = {}
            for mode in (('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')):
                directory = root / f'w{workers}' / f'{iteration:02d}' / mode
                directory.mkdir(parents=True)
                row = sample(directory, workers, base if mode == 'baseline' else None)
                row.update(workers=workers, iteration=iteration, mode=mode, first=iteration == 0)
                pair[mode] = row['result']
                rows.append(row)
            if pair['baseline'] != pair['candidate']:
                write(root / 'mismatch.json', pair)
                raise RuntimeError('Canonical results differ')
        for mode in ('baseline', 'candidate'):
            values = sorted(row['seconds'] * 1000 for row in rows if row['workers'] == workers
                            and row['mode'] == mode and not row['first'])
            summaries.append({'workers': workers, 'mode': mode, 'warm_count': len(values),
                              'p50_ms': statistics.median(values), 'p95_ms': values[28]})
    write(root / 'timings.json', rows)
    diagnostic = []
    for workers in (1, 4):
        for mode in ('baseline', 'candidate'):
            directory = root / 'diag' / f'{workers}_{mode}'
            directory.mkdir(parents=True)
            diagnostic.append({'workers': workers, 'mode': mode, **sample(
                directory, workers, base if mode == 'baseline' else None, diagnostics=True)})
    write(root / 'diagnostics.json', diagnostic)
    for workers in (1, 4):
        requests = [row['request'] for row in diagnostic if row['workers'] == workers]
        if requests[0] != requests[1]:
            raise RuntimeError('Captured Planner requests differ')
    from tests.manual.process_diagnostics import supervise
    process = supervise([sys.executable, '-u', '-X', 'faulthandler', '-m',
                         'tests.manual.tool_completion_probe', '--child', '--output', str(root / 'crash')],
                        root / 'process', timeout=30)
    if process['exit_code'] != 23:
        raise RuntimeError(f'Unexpected crash probe outcome: {process}')
    from evaluation.benchmarks.core.audit import BenchmarkAuditReader, EpisodeAuditCursor
    from evaluation.benchmarks.core.review import BenchmarkCaseReview, BenchmarkReviewRenderer
    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(root / 'crash'))
    assert len(audit.uncommitted_tool_completions()) == 1
    write(root / 'crash_audit.json', audit.to_dict())
    case = BenchmarkCaseReview('01', 'Fake tool persistence', 'offline',
        'Constructed inputs; intentional process exit; no navigation score.', {}, {}, audit, ())
    review_root = root / 'review'
    for detailed in (False, True):
        BenchmarkReviewRenderer(detailed_audit=detailed).write_case(review_root, case)
    body = '<h1>Offline tool completion probe</h1><p>Constructed inputs; no model, simulator or GPU calls.</p>'
    body += '<p><a href="cases/01/review.html">Timeline</a> · <a href="cases/01/audit_review.html">Audit</a></p>'
    body += '<pre>' + html.escape(json.dumps(summaries, indent=2)) + '</pre>'
    body += '<p>First call excluded; P95 uses nearest-rank on 30 warm calls. Diagnostics are separate.</p>'
    body += '<p>Baseline uses the old Runtime batch-save method with the unchanged no-callback scheduler interface.</p>'
    (review_root / 'review.html').write_text('<!doctype html><meta charset="utf-8">' + body, encoding='utf-8')
    write(root / 'summary.json', {'status': 'completed', 'performance': summaries,
        'canonical_results_equal': True, 'intentional_child_exit': process,
        'uncommitted_receipts_preserved': 1})
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    manifest['status'] = 'completed'
    write(root / 'manifest.json', manifest)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
