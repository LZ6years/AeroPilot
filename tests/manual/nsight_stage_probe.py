"""Use an installed Nsight Systems CLI for one warm batch per child.

No dependency installation, CPU sampling, WDDM tracing or privilege escalation.
Project Python children retain sys.executable. Raw reports are exported by the
native CLI; SQLite is analyzed using stdlib sqlite3, not Nsight's bundled Python.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.manual import gpu_variance_probe as variance
from tests.manual.process_diagnostics import write_json
from tests.manual.read_only_concurrency_probe import child, read, sha
from tests.manual.export_benchmark_review import validate_bundle
from tests.manual.task_profile import TaskProfile


def script_target(command):
    """Avoid nested Python -c quoting through the Windows Nsight launcher."""
    idx = next(i for i,v in enumerate(command) if v.endswith('read_only_concurrency_probe.py'))
    return command[:4] + [str(Path(command[idx]).with_name('nsight_stage_probe.py')), *command[idx+1:]]


def should_capture(index, total, previous_sec, already_captured, on_slow):
    """One bounded diagnostic sample, never a repeat-until-success benchmark."""
    return not already_captured and (index == total or
        (on_slow and index >= 3 and previous_sec is not None and previous_sec > 3.0))


def summarize_sqlite(path):
    result = {}
    with sqlite3.connect(path) as database:
        tables = {r[0] for r in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
            if table not in tables:
                result[table] = None
                continue
            count, elapsed = database.execute(f'SELECT COUNT(*), SUM(end-start) FROM {table}').fetchone()
            result[table] = {'count': count, 'sum_sec': (elapsed or 0) / 1e9}
            if table == 'CUPTI_ACTIVITY_KIND_RUNTIME' and 'StringIds' in tables:
                result[table]['top'] = list(database.execute(
                    f'SELECT s.value, COUNT(*), SUM(a.end-a.start)/1e9 FROM {table} a JOIN StringIds s ON a.nameId=s.id GROUP BY s.value ORDER BY SUM(a.end-a.start) DESC LIMIT 15'))
        if not result['CUPTI_ACTIVITY_KIND_KERNEL'] or not result['CUPTI_ACTIVITY_KIND_KERNEL']['count']:
            raise ValueError('Nsight export contains no actual CUDA kernel events')
    return result


def traced_child(directory, sample):
    import torch
    from src.runtime.control.read_only_executor import ReadOnlyBatchExecutor
    manifest = read(directory.parent / 'manifest.json')
    target = manifest['repeats']
    count = 0
    captured = None
    previous_sec = None
    original_execute, original_span = ReadOnlyBatchExecutor.execute, TaskProfile.span

    @contextmanager
    def nvtx_span(self, name):
        with torch.cuda.nvtx.range(name), original_span(self, name):
            yield

    def execute(executor, jobs):
        nonlocal count, captured, previous_sec
        count += 1
        if not should_capture(count, target, previous_sec, captured is not None, manifest.get('nsight_capture_on_slow', False)):
            began = time.perf_counter()
            result = original_execute(executor, jobs)
            previous_sec = time.perf_counter() - began
            return result
        captured = count
        write_json(directory / 'capture_selection.json', {'repeat': count, 'previous_batch_sec': previous_sec,
            'reason': 'final_batch' if count == target else 'previous_non_cold_batch_over_3_seconds'})
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        try:
            with patch.object(TaskProfile, 'span', nvtx_span):
                return original_execute(executor, jobs)
        finally:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()

    with patch.object(ReadOnlyBatchExecutor, 'execute', execute):
        child(directory, sample, 1)
    # Metadata classification only: completed tool results remain unchanged.
    rows = read(directory / 'results.json')
    rows[captured - 1]['profiled'] = True
    rows[captured - 1]['nsight_captured'] = True
    write_json(directory / 'results.json', rows)
    write_json(directory / f'r{captured}' / 'results.json', rows[captured - 1])


def run(args):
    nsys = args.nsys.resolve()
    if not nsys.is_file():
        raise ValueError('Requested existing Nsight CLI is missing')
    original_supervise, original_freeze = variance.supervise, variance.freeze

    def freeze(*a, **kw):
        manifest = original_freeze(*a, **kw)
        manifest['nsight'] = {'executable': str(nsys), 'sha256': sha(nsys),
                              'version': subprocess.run([str(nsys), '--version'], capture_output=True, text=True, check=True).stdout.strip()}
        manifest['nsight_capture_on_slow'] = args.capture_on_slow
        return manifest

    def supervise(command, directory, **kw):
        # Nsight's Windows launcher strips quotes inside Python -c source.
        # Invoke a real script and let it set cwd; no nested command-language quoting.
        command = script_target(command)
        wrapped = [str(nsys), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--show-output=true',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                   '--output=' + str(directory / 'capture'), '--force-overwrite=false', *command]
        status = original_supervise(wrapped, directory, **kw)
        if status['exit_code']:
            return status
        export = subprocess.run([str(nsys), 'export', '--type=sqlite', '--output=' + str(directory / 'capture.sqlite'),
                                 str(directory / 'capture.nsys-rep')], capture_output=True, text=True)
        write_json(directory / 'export.json', {'exit_code': export.returncode, 'stdout': export.stdout, 'stderr': export.stderr})
        if export.returncode:
            raise RuntimeError('Nsight SQLite export failed; preserve raw report')
        write_json(directory / 'nsight_summary.json', summarize_sqlite(directory / 'capture.sqlite'))
        return status

    with patch.object(variance, 'supervise', supervise), patch.object(variance, 'freeze', freeze):
        variance.run(args)
    summary = read(args.root / 'summary.json')
    evidence = []
    for condition in summary['conditions']:
        directory = args.root / condition['directory']
        record = read(directory / 'nsight_summary.json')
        evidence.append({'trial': condition['trial'], **record})
        for filename in ('nsight_summary.json', 'capture.nsys-rep', 'capture.sqlite'):
            shutil.copyfile(directory / filename, args.review_root / (directory.name + '_' + filename))
    write_json(args.root / 'nsight_summary.json', evidence)
    write_json(args.review_root / 'nsight_summary.json', evidence)
    index = args.review_root / 'index.html'
    index.write_text('<a href="nsight_summary.json">Actual CUDA event summary</a>' + index.read_text(encoding='utf-8'), encoding='utf-8')
    write_json(args.review_root / 'resource_validation.json', validate_bundle(args.review_root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'child'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--input-manifest', type=Path)
    parser.add_argument('--review-root', type=Path)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--nsys', type=Path)
    parser.add_argument('--capture-on-slow', action='store_true', help='Capture once after a >3 s non-cold batch, otherwise final batch')
    args = parser.parse_args()
    if args.mode == 'child':
        os.chdir(ROOT)
        traced_child(args.root, args.sample)
    else:
        run(args)
