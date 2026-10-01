"""Fixed-input, fixed-thread GPU variance experiment; no simulator or LLM.

Uses the existing frozen-input replay, backend hooks, process supervision and
portable review exporter. Performance sampling is observational, not a GPU
clock/power policy change. No result tolerance or model setting is changed.
"""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.manual.gpu_probe_environment import check_interpreter, cache_sources, verify_weights, other_project_jobs
from tests.manual.gpu_resource_monitor import sample_gpu_detailed
from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.read_only_concurrency_probe import (
    freeze, read, sha, condition_summary, validate_condition, compare_with_serial, render,
)
from tests.manual.task_profile import summarize
from tests.manual.export_benchmark_review import validate_bundle
from tests.manual.review_browser_probe import inspect


def batch_resources(directory):
    """Join sampled device state to completed batch intervals; never fill gaps.

    NVIDIA's three queries are sequential and coarse (~1 Hz), not simultaneous
    kernel counters. Visible compute processes exclude neither WDDM graphics
    load nor hidden driver activity. Correlation alone does not establish cause.
    """
    spans = summarize(directory / 'timings.jsonl')['spans']
    samples = [json.loads(line) for line in (directory / 'gpu_resources.jsonl').read_text().splitlines()]
    pid = read(directory / 'process.json')['pid']
    rows = []
    for batch in (s for s in spans if s['name'] == 'experiment.batch'):
        nearby = [r for r in samples if batch['time'] <= r['monotonic_sec'] <= batch['ended']]
        forward = [s for s in spans if s['decision'] == batch['decision'] and s['name'].endswith('.forward')]
        performance = [p for r in nearby for p in r.get('performance', {}).get('rows', [])]
        devices = [d for r in nearby for d in r.get('devices', [])]
        external = {int(p['pid']) for r in nearby for p in r.get('compute_processes', {}).get('rows', [])
                    if p.get('pid') is not None and int(p['pid']) != pid}
        ranges = {}
        for key, source in [('sm_mhz', performance), ('power_w', performance), ('temperature_c', performance),
                            ('used_mib', devices), ('gpu_percent', devices)]:
            values = [r[key] for r in source if r.get(key) is not None]
            ranges[key] = [min(values), max(values)] if values else None
        rows.append({'repeat': batch['decision'], 'wall_sec': batch['inclusive_sec'],
                     'grounding_sec': sum(s['inclusive_sec'] for s in forward if '.grounding.' in s['name']),
                     'sam_sec': sum(s['inclusive_sec'] for s in forward if '.sam.' in s['name']),
                     'sample_count': len(nearby), 'ranges': ranges, 'visible_external_compute_pids': sorted(external)})
    return rows


def run(args):
    check_interpreter()
    if not 1 <= args.sample <= 4 or args.trials < 1 or args.repeats < 2:
        raise ValueError('Need sample 1..4, positive trials and at least two repeats')
    if getattr(args, 'cuda_events', False) and (getattr(args, 'cuda_trace', False) or args.repeats < 5):
        raise ValueError('Event mode needs >=5 repeats and must be separate from profiler mode')
    if getattr(args, 'sam_coordinate_ab', False) and (getattr(args, 'cuda_events', False) or getattr(args, 'cuda_trace', False)):
        raise ValueError('Coordinate A/B must run without intrusive event/profiler instrumentation')
    blockers = other_project_jobs(ROOT)
    if blockers:
        raise RuntimeError(f'Other project experiments active: {blockers}')
    root, review = args.root.resolve(), args.review_root.resolve()
    manifest = freeze(root, review, input_manifest=args.input_manifest.resolve(), require_cuda=True)
    snapshot = Path(manifest['source_snapshot'])
    conditions, results = [], {}
    try:
        manifest.update(repeats=args.repeats, workers=[1], trials=args.trials, selected_sample=args.sample,
                        windows_gpu_memory=getattr(args, 'windows_memory', False),
                        experiment='fixed-thread GPU variance',
                        limitations='Only selected sample executed. Fixed one worker, serial device gate; no navigation. '
                        'Coarse sequential NVIDIA sampling cannot prove causation or GPU exclusivity.')
        if getattr(args, 'cuda_trace', False):
            manifest['cuda_trace_repeats'] = [args.repeats]
            manifest['limitations'] += ' Last batch is profiled and excluded from ordinary warm statistics.'
        if getattr(args, 'cuda_events', False):
            manifest['cuda_event_repeats'] = list(range(args.repeats - 2, args.repeats + 1))
            manifest['sam_module_events'] = getattr(args, 'sam_modules', False)
            manifest['limitations'] += ' Last three batches use CUDA Events and are excluded from normal warm statistics.'
        if getattr(args, 'sam_coordinate_ab', False):
            manifest['sam_coordinate_ab'] = True
            manifest['limitations'] += ' Experimental coordinate placement A/B; ordinary warm aggregate mixes variants; use per-variant report.'
        manifest['weight_sources'] = cache_sources(read(snapshot / 'plugins/capabilities/visual_grounding/measure_visible_depth/capability.json')['settings'])
        write_json(root / 'manifest.json', manifest)
        for trial in range(1, args.trials + 1):
            if other_project_jobs(ROOT):
                raise RuntimeError('Other experiment started; pausing')
            if any(sha(snapshot / name) != value for name, value in manifest['snapshot_hashes'].items()):
                raise ValueError('Frozen source changed')
            sample = manifest['samples'][args.sample - 1]
            if any(sha(path) != value for path, value in sample['input_hashes'].items()):
                raise ValueError('Frozen input changed')
            verify_weights(manifest['weight_sources'])
            render(root, manifest, conditions, state='running')
            directory = root / f'trial{trial}'
            command = [sys.executable, '-u', '-X', 'faulthandler', '-c',
                       'import os,runpy,sys; os.chdir(sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1),run_name="__main__")',
                       str(snapshot), str(snapshot / 'tests/manual/read_only_concurrency_probe.py'),
                       'child', '--root', str(directory), '--sample', str(args.sample), '--workers', '1']
            memory_options = {'windows_gpu_memory': True} if getattr(args, 'windows_memory', False) else {}
            status = supervise(command, directory, monitor_gpu=True, gpu_sampler=sample_gpu_detailed, **memory_options)
            condition = dict(sample=args.sample, workers=1, kind='comparison', directory=directory.name,
                             trial=trial, exit_code=status['exit_code'], process_elapsed_sec=status['elapsed_seconds'],
                             **condition_summary(directory))
            conditions.append(condition)
            if getattr(args, 'windows_memory', False):
                from tests.manual.windows_gpu_memory import summarize_process_memory
                condition['windows_gpu_memory'] = summarize_process_memory(directory, status['pid'])
                if condition['windows_gpu_memory']['samples'] == 0:
                    raise RuntimeError('No Windows GPU memory samples for target process; inspect monitor logs')
            if status['exit_code']:
                raise RuntimeError(f'{directory.name}: child failure')
            results[trial] = read(directory / 'results.json')
            validate_condition(condition, results[trial], require_cuda=True, expected_batches=args.repeats)
            compare_with_serial(results)
            if getattr(args, 'sam_coordinate_ab', False):
                from tests.manual.read_only_concurrency_probe import percentile
                import statistics
                condition['coordinate_variants'] = {}
                for variant in ('original', 'device'):
                    variant_rows = [r for r in results[trial] if not r['cold'] and r['coordinate_variant'] == variant]
                    values = [r['wall_sec'] for r in variant_rows]
                    condition['coordinate_variants'][variant] = dict(count=len(values), p50=statistics.median(values), p95=percentile(values, .95))
            condition['batch_resources'] = batch_resources(directory)
            if getattr(args, 'cuda_trace', False):
                condition['cuda_timeline'] = read(directory / f'r{args.repeats}' / 'cuda_summary.json')
            if getattr(args, 'cuda_events', False):
                condition['cuda_events'] = [read(directory / f'r{n}' / 'cuda_events.json') for n in manifest['cuda_event_repeats']]
            if trial == 1:
                manifest['environment'] = read(directory / 'environment.json')
                write_json(root / 'manifest.json', manifest)
            print(json.dumps({k: condition[k] for k in ('trial', 'cold_sec', 'warm_p50_sec', 'warm_p95_sec')}), flush=True)
        render(root, manifest, conditions, state='passed')
        (review / 'browser_validation').mkdir(exist_ok=True)
        pages = [f'cases/{args.sample:02d}/{name}.html' for name in ('review', 'audit_review')]
        for name, executable in [('chrome', 'C:/Program Files/Google/Chrome/Application/chrome.exe'),
                                 ('edge', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')]:
            write_json(review / 'browser_validation' / f'{name}.json', inspect(Path(executable), review, pages, name))
        table = '<table border="1"><tr><th>Trial</th><th>Cold s</th><th>Warm P50 s</th><th>Warm P95 s</th></tr>'
        for c in conditions:
            table += f'<tr><td>{c["trial"]}</td><td>{c["cold_sec"]:.4f}</td><td>{c["warm_p50_sec"]:.4f}</td><td>{c["warm_p95_sec"]:.4f}</td></tr>'
        (review / 'performance.html').write_text('<!doctype html><meta charset="utf-8"><h1>Fixed-thread GPU variance</h1><a href="index.html">Review</a>' +
            table + '</table><p>' + html.escape(manifest['limitations']) + '</p><pre>' +
            html.escape(json.dumps(conditions, indent=2)) + '</pre>', encoding='utf-8')
        index = review / 'index.html'
        if getattr(args, 'cuda_events', False):
            import shutil
            links = []
            for c in conditions:
                for n in manifest['cuda_event_repeats']:
                    name = f'{c["directory"]}_r{n}_cuda_events.json'
                    shutil.copyfile(root / c['directory'] / f'r{n}' / 'cuda_events.json', review / name)
                    links.append(f'<a href="{name}">{name}</a>')
            index.write_text(index.read_text(encoding='utf-8') + '<p>' + ' | '.join(links) + '</p>', encoding='utf-8')
        if getattr(args, 'cuda_trace', False):
            import shutil
            links = []
            for c in conditions:
                for name in ('cuda_trace.json', 'cuda_summary.json'):
                    exported = c['directory'] + '_' + name
                    shutil.copyfile(root / c['directory'] / f'r{args.repeats}' / name, review / exported)
                    links.append(f'<a href="{exported}">{exported}</a>')
            index.write_text(index.read_text(encoding='utf-8') + '<p>' + ' | '.join(links) + '</p>', encoding='utf-8')
        index.write_text(index.read_text(encoding='utf-8').replace('<h1>', '<a href="performance.html">Performance and sampled GPU state</a><h1>', 1), encoding='utf-8')
        write_json(review / 'resource_validation.json', validate_bundle(review))
    except BaseException as exc:
        render(root, manifest, conditions, state='paused', error={'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--cuda-trace', action='store_true', help='Profile final batch; exclude it from warm statistics')
    parser.add_argument('--cuda-events', action='store_true', help='Stage event timing for last 3 batches; single thread only')
    parser.add_argument('--sam-modules', action='store_true', help='Also measure SAM module boundaries in event batches')
    parser.add_argument('--sam-coordinate-ab', action='store_true', help='Experimental same-process original/device coordinate ABBA comparison')
    parser.add_argument('--windows-memory', action='store_true', help='Read-only per-process Windows GPU memory counters')
    run(parser.parse_args())
