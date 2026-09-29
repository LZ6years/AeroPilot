"""Frozen, resumable 10+10 benchmark experiment using the production runners.

``prepare`` writes a manifest without flying. ``run`` supervises one case per
process and stops on infrastructure or cleanup failure. ``child`` is private to
the supervisor. No case selection or model behavior enters production code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.aerial_vln.selection import _scene_balanced_sample
from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner
from evaluation.benchmarks.travel_uav.selection import _length_spread
from src.agent.config import load_model_config
from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.control import RemoteSceneControl
from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.supervised_flight_probe import cleanup_owned, write_review
from tests.manual.task_progress_simulation_probe import AuditedPlanner

AERIAL_HASH = '29b52c9b7a5554376410f5d1d0631f563b29df730c20fb1b7adc23f36c004ac3'
TRAVEL_HASH = '8a2dbe59a182774f147869269a4a612546031498d5e40fac0697227412152e87'
ANCHOR = '39GHHAVOMOP8ZV27ILH5P0TV5F6J4A'
TRAVEL_HISTORICAL = (
    'Carla_Town05__27940e4d-36b6-4021-a727-9c457d173c79',
    'Carla_Town10HD__53e4e305-2f7f-46ef-ad07-554ba86b380f',
    'ModernCityMap__a2625ae8-d423-4708-a6d3-cc33606a7e6e',
    'NYCEnvironmentMegapa__bc86986b-c79a-43a0-ae6b-6408e568589f',
    'NYCEnvironmentMegapa__c3f9fc36-7c55-4a48-8213-5adce9b35b12',
    'NewYorkCity__2a070f7b-50d0-490b-a648-633759e51e3b',
    'TropicalIsland__b2665c19-3db6-45ff-be2d-795004aaeb5b',
)
AERIAL_CONFIG = ROOT / 'evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_full.example.json'
TRAVEL_CONFIG = ROOT / 'evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json'
TRAVEL_MAP = ROOT / 'configs/maps/travel_uav_remote.local'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


from tests.manual.experiment_sources import source_hashes


class SelectedAerialRunner(AerialVlnBenchmarkRunner):
    """Filter an exact ID only after full val_seen difficulty classification."""

    def __init__(self, config, case_id):
        super().__init__(config)
        self.case_id = case_id

    def prepare_evaluation(self, *, require_execution_ready):
        prepared = super().prepare_evaluation(require_execution_ready=False)
        cases = tuple(case for case in prepared.cases if case.episode_id == self.case_id)
        selected = tuple(item for item in prepared.selection.assignments if item.episode_id == self.case_id)
        if len(cases) != 1 or len(selected) != 1:
            raise ValueError(f'Frozen AerialVLN case missing: {self.case_id}')
        selection = replace(prepared.selection, selected_assignments=selected)
        prepared = replace(prepared, cases=cases, selection=selection,
                           execution_readiness=self._execution_readiness(selection.selected_scene_ids))
        if require_execution_ready:
            self._validate_dataset_for_execution(prepared)
            if not prepared.execution_readiness['ready']:
                raise ValueError(f'AerialVLN scene unavailable: {self.case_id}')
        return prepared

    def _make_runtime(self, run_dir, recorder, *, planner=None):
        return super()._make_runtime(run_dir, recorder,
                                    planner=planner or AuditedPlanner(self._model_config()))


def aerial_config(*, output=None, run_root=None):
    config = AerialVlnEvalConfig.from_file(str(AERIAL_CONFIG))
    return replace(config, expected_split_sha256=AERIAL_HASH, limit_per_difficulty=None,
                   difficulties=('low', 'medium', 'high'), output_dir=str(output or config.output_dir),
                   run_root=str(run_root or config.run_root))


def travel_config(*, output=None, run_root=None, case_id=None):
    config = TravelUavEvalConfig.from_file(str(TRAVEL_CONFIG))
    return replace(config, expected_index_sha256=TRAVEL_HASH,
                   map_config_path=str(TRAVEL_MAP), limit_per_difficulty=None,
                   case_ids=(case_id,) if case_id else None,
                   output_dir=str(output or config.output_dir), run_root=str(run_root or config.run_root))


def historical_aerial_ids() -> set[str]:
    """Only previously finished benchmark cases are eligible for the 10."""
    result: set[str] = set()
    base = ROOT / 'evaluation/benchmarks/aerial_vln/outputs'
    for path in base.rglob('summary.json'):
        try:
            summary = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if summary.get('dataset', {}).get('split') != 'val_seen':
            continue
        for row in summary.get('cases', []):
            if row.get('status') in ('ok', 'stopped'):
                result.add(row['episode_id'])
    return result


def historical_travel_evidence(case_id: str) -> dict:
    paths = sorted([*(ROOT / 'evaluation/benchmarks/travel_uav/outputs').rglob(
                   f'cases/{case_id}/case_result.json'),
                   *(ROOT / 'tests/manual_outputs').rglob(f'cases/{case_id}/case_result.json')])
    for path in paths:
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('dataset_split') == 'test_seen' and row.get('difficulty') == 'easy':
            return {'path': path.relative_to(ROOT).as_posix(), 'status': row['status'],
                    'infrastructure_error': row.get('infrastructure_error')}
    raise ValueError(f'Historical TravelUAV easy case unavailable: {case_id}')


def freeze_manifest() -> dict:
    """Classify full populations, then use historical IDs and existing samplers."""
    prepared = AerialVlnBenchmarkRunner(aerial_config()).prepare_evaluation(require_execution_ready=False)
    if prepared.provenance.split_sha256 != AERIAL_HASH or prepared.selection.population_case_count != 333:
        raise ValueError('AerialVLN full-split identity changed')
    history = historical_aerial_ids()
    if ANCHOR not in history:
        raise ValueError('AerialVLN anchor lacks a completed historical run')
    selected = []
    for label, count in (('low', 4), ('medium', 3), ('high', 3)):
        bucket = [item for item in prepared.selection.assignments
                  if item.difficulty == label and item.episode_id in history and item.episode_id != ANCHOR]
        chosen = list(_scene_balanced_sample(bucket, count - (label == 'low')))
        if label == 'low':
            chosen.insert(0, prepared.selection.assignment_by_episode[ANCHOR])
        if len(chosen) != count:
            raise ValueError(f'Insufficient historical {label} AerialVLN cases')
        selected.extend({'benchmark': 'aerial', 'case_id': item.episode_id,
                         'difficulty': label, 'scene': item.scene_id,
                         'reference_path_length_m': item.reference_path_length_m,
                         'historical': True} for item in chosen)
    travel, provenance = load_travel_uav_index(travel_config().index_path, split='test_seen')
    if provenance.index_sha256 != TRAVEL_HASH:
        raise ValueError('TravelUAV index identity changed')
    by_id = {case.episode_id: case for case in travel}
    evidence = {case_id: historical_travel_evidence(case_id) for case_id in TRAVEL_HISTORICAL}
    hard = _length_spread([case for case in travel if case.difficulty == 'hard'], 3)
    if len(hard) != 3 or any(case.episode_id in TRAVEL_HISTORICAL for case in hard):
        raise ValueError('TravelUAV hard sample is invalid')
    for case in [*(by_id[case_id] for case_id in TRAVEL_HISTORICAL), *hard]:
        selected.append({'benchmark': 'travel', 'case_id': case.episode_id,
                         'difficulty': case.difficulty, 'scene': case.map_name,
                         'reference_path_length_m': case.reference_path_length_m,
                         'historical': case.episode_id in TRAVEL_HISTORICAL,
                         'history': evidence.get(case.episode_id)})
    validate_manifest_cases(selected)
    model = load_model_config('.env')
    if model.task_progress_reminder_interval != 5:
        raise ValueError('Expected the existing five-turn reminder')
    if aerial_config().runtime_loop_guard_windows != 300 or travel_config().runtime_loop_guard_windows != 300:
        raise ValueError('Expected original 300 physical-action windows')
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'model': model.model_name,
            'dataset_sha256': {'aerial': AERIAL_HASH, 'travel': TRAVEL_HASH},
            'sampling': {'aerial': 'historical 4 low / 3 medium / 3 high; full val_seen classification',
                         'travel': '7 historical easy / 3 new hard; length spread'},
            'aerial_thresholds_m': {'q33': prepared.selection.q33_m, 'q67': prepared.selection.q67_m},
            'source_hashes': source_hashes(), 'cases': selected}


def import_aerial_manifest(source: Path) -> dict:
    """Freeze a paired Aerial-only run without resampling or modifying its source."""
    previous = json.loads(source.read_text(encoding='utf-8'))
    selected = [dict(item) for item in previous['cases'] if item['benchmark'] == 'aerial']
    validate_manifest_cases(selected)
    prepared = AerialVlnBenchmarkRunner(aerial_config()).prepare_evaluation(require_execution_ready=False)
    for item in selected:
        actual = prepared.selection.assignment_by_episode[item['case_id']]
        if actual.difficulty != item['difficulty'] or actual.scene_id != item['scene']:
            raise ValueError('Historical classification or scene changed')
    if prepared.provenance.split_sha256 != previous['dataset_sha256']['aerial']:
        raise ValueError('Historical dataset changed')
    config = load_model_config('.env')
    if (config.task_progress_reminder_interval, config.max_tool_rounds,
            config.max_consecutive_protocol_rejections, aerial_config().runtime_loop_guard_windows) != (5, 10, 6, 300):
        raise ValueError('Historical task budgets or reminder interval changed')
    current = source_hashes()
    old = previous['source_hashes']
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'model': config.model_name,
            'model_config': config.redacted_summary(), 'cases': selected, 'browser_gate': True,
            'dataset_sha256': {'aerial': previous['dataset_sha256']['aerial']},
            'aerial_thresholds_m': previous['aerial_thresholds_m'],
            'source_hashes': current, 'baseline_manifest': str(source.resolve()),
            'requires_admission': True,
            'baseline_manifest_sha256': sha256(source),
            'source_changes_from_baseline': [k for k in sorted(set(old) | set(current)) if old.get(k) != current.get(k)],
            'comparison_limitations': 'Historical Qwen provider defaults versus explicit DeepSeek non-thinking; not equal internal reasoning budgets.'}


def validate_manifest_cases(cases) -> None:
    if len(cases) not in (10, 20) or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Frozen manifest must have 20 distinct cases')
    aerial, travel = cases[:10], cases[10:]
    if any(case['benchmark'] != 'aerial' or not case['historical'] for case in aerial):
        raise ValueError('AerialVLN cases must all be historical')
    if any(case['benchmark'] != 'travel' for case in travel):
        raise ValueError('TravelUAV cases must follow AerialVLN cases')
    if [case['difficulty'] for case in aerial] != ['low'] * 4 + ['medium'] * 3 + ['high'] * 3:
        raise ValueError('AerialVLN difficulty order is wrong')
    if not travel:
        return
    if [case['difficulty'] for case in travel] != ['easy'] * 7 + ['hard'] * 3:
        raise ValueError('TravelUAV difficulty order is wrong')
    if [case['case_id'] for case in travel[:7]] != list(TRAVEL_HISTORICAL):
        raise ValueError('TravelUAV historical identity changed')
    if any(not case['historical'] for case in travel[:7]):
        raise ValueError('TravelUAV historical identity changed')
    if any(case['historical'] for case in travel[7:]):
        raise ValueError('TravelUAV hard cases must be new')


def child(directory: Path, item: dict, expected_hash: str) -> None:
    """Run exactly one frozen case in an isolated child process."""
    output, runs = directory / 'evaluation_output', directory / 'runs'
    if item['benchmark'] == 'aerial':
        runner = SelectedAerialRunner(aerial_config(output=output, run_root=runs), item['case_id'])
        runner._loaded_model_config = load_model_config('.env')
        map_config = runner.config.map_config_path
        actual_hash = sha256(Path(runner.config.dataset_root) / 'val_seen.json')
    else:
        runner = TravelUavBenchmarkRunner(travel_config(output=output, run_root=runs,
                                                        case_id=item['case_id']),
                                          planner_factory=AuditedPlanner)
        runner._model_config_value = load_model_config('.env')
        map_config = runner.config.map_config_path
        actual_hash = sha256(Path(runner.config.index_path))
    frozen = json.loads((directory.parents[1] / 'manifest.json').read_text(encoding='utf-8'))
    if frozen.get('model_config') and runner._model_config().redacted_summary() != frozen['model_config']:
        raise ValueError('Frozen model configuration changed')
    if actual_hash != expected_hash or runner._model_config().model_name != frozen['model']:
        raise ValueError('Frozen dataset or requested model changed')
    runner.prepare_evaluation(require_execution_ready=True)
    write_json(directory / 'experiment.json', {'case': item, 'config': runner.config.to_dict(),
               'model': runner._model_config().redacted_summary(), 'dataset_sha256': actual_hash,
               'map_config_path': map_config, 'source_hashes': source_hashes(),
               'thinking_policy': runner._model_config().thinking_mode or 'provider default'})
    output.mkdir(parents=True, exist_ok=False)
    result = runner.run(eval_id='evaluation')
    write_json(directory / 'result.json', result)


def preflight(root: Path, manifest: dict) -> dict:
    """Check frozen source, local routing, SSH and each remote executable."""
    if manifest['source_hashes'] != source_hashes():
        raise ValueError('Executable source changed after the manifest was frozen')
    for name, path in (('aerial', Path(aerial_config().dataset_root) / 'val_seen.json'),
                       ('travel', Path(travel_config().index_path))):
        if name not in manifest['dataset_sha256']:
            continue
        if sha256(path) != manifest['dataset_sha256'][name]:
            raise ValueError(f'{name} dataset hash changed after freezing')
    control = RemoteSceneControl()
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'scenes': []}
    for name, config in (('aerial', aerial_config()), ('travel', travel_config())):
        packages, _ = load_map_config(config.map_config_path)
        by_id = {package.map_id: package for package in packages}
        for scene in sorted({item['scene'] for item in manifest['cases'] if item['benchmark'] == name}):
            mapped = config.scene_map[scene]
            if mapped not in by_id:
                raise ValueError(f'Missing local map package: {name}/{scene}')
            package = by_id[mapped]
            try:
                remote = control.preflight(package)
                result['scenes'].append({'benchmark': name, 'scene': scene,
                                         'map_id': mapped, 'remote': remote})
            except Exception as exc:
                result['error'] = {'benchmark': name, 'scene': scene,
                                   'type': type(exc).__name__, 'message': str(exc)}
                write_json(root / 'preflight.json', result)
                raise
    write_json(root / 'preflight.json', result)
    return result


def case_dir(root: Path, index: int, item: dict) -> Path:
    return root / 'cases' / f'{index + 1:02d}_{item["benchmark"]}_{item["case_id"]}'


def latest_case_dir(root: Path, index: int, item: dict) -> Path:
    """Recovery attempts are new directories; the original evidence stays intact."""
    original = case_dir(root, index, item)
    attempts = sorted(original.parent.glob(original.name + '_recovery_*'))
    return attempts[-1] if attempts else original


def audit_metrics(directory: Path, item: dict) -> dict:
    path = directory / 'evaluation_output/evaluation/cases' / item['case_id'] / 'audit.json'
    if not path.exists():
        return {}
    audit = json.loads(path.read_text(encoding='utf-8'))
    calls = Counter(call.get('tool_name') for entry in audit.get('agent_transcript', [])
                    if entry.get('kind') == 'assistant_turn'
                    for call in entry.get('payload', {}).get('tool_calls', []))
    feedback = [entry.get('payload', {}).get('message', '')
                for entry in audit.get('agent_transcript', [])
                if entry.get('kind') == 'protocol_feedback']
    usage = Counter()
    for decision in audit.get('planner_debug', []):
        for model_round in decision.get('model_rounds', []):
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                usage[key] += ((model_round.get('response') or {}).get('usage') or {}).get(key, 0) or 0
    return {'plan_calls': calls['runtime.update_plan'], 'knowledge_calls': calls['knowledge.load'],
            'token_usage': dict(usage), 'protocol_rejections': len(feedback),
            'invalid_argument_rejections': sum('JSON' in message or 'argument' in message.lower()
                                               for message in feedback),
            'terminal_reference_rejections': sum('evidence_ref' in message
                                                 for message in feedback),
            'first_flight_preceded_by_plan': _first_flight_preceded_by_plan(audit),
            'feedback_messages': feedback}


def _first_flight_preceded_by_plan(audit: dict) -> bool | None:
    for entry in audit.get('agent_transcript', []):
        if entry.get('kind') != 'assistant_turn':
            continue
        calls = [call.get('tool_name') for call in entry.get('payload', {}).get('tool_calls', [])]
        if 'runtime.update_plan' in calls:
            return True
        if any(name and name.startswith('uav.') for name in calls):
            return False
    return None


def classify_case(directory: Path, supervision: dict, item: dict) -> dict:
    result_path = directory / 'result.json'
    benchmark_result = None
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding='utf-8'))
        rows = payload.get('results') or []
        if len(rows) == 1 and rows[0].get('episode_id') == item['case_id']:
            benchmark_result = rows[0]
    cleanup = supervision.get('cleanup') or []
    clean = bool(cleanup) and all(entry['remote_stop'].get('cleanup_verified')
                                 and entry['endpoint'].get('status') == 'unreachable'
                                 for entry in cleanup)
    if supervision.get('cleanup_error') or not clean:
        status = 'cleanup_failed'
    elif supervision['exit_code'] != 0 or benchmark_result is None:
        status = 'incomplete'
    elif benchmark_result.get('infrastructure_error'):
        status = 'infrastructure_error'
    elif benchmark_result.get('status') == 'error':
        status = 'model_error'
    else:
        status = 'completed'
    return {'status': status, 'benchmark_status': benchmark_result.get('status') if benchmark_result else None,
            'infrastructure_error': benchmark_result.get('infrastructure_error') if benchmark_result else None,
            'metrics': benchmark_result.get('metrics') if benchmark_result else {},
            'elapsed_seconds': supervision['elapsed_seconds'], 'exit_code': supervision['exit_code'],
            'cleanup_verified': clean, **audit_metrics(directory, item)}


def run_one(root: Path, index: int, item: dict, *, supervisor=supervise, cleaner=cleanup_owned,
            directory: Path | None = None) -> dict:
    directory = directory or case_dir(root, index, item)
    if directory.exists():
        raise ValueError(f'Existing case output must not be overwritten: {directory}')
    command = [sys.executable, '-u', '-X', 'faulthandler', str(Path(__file__).resolve()),
               'child', '--root', str(directory), '--case-json', json.dumps(item),
               '--dataset-hash', AERIAL_HASH if item['benchmark'] == 'aerial' else TRAVEL_HASH]
    result = supervisor(command, directory)
    map_config = aerial_config().map_config_path if item['benchmark'] == 'aerial' else str(TRAVEL_MAP)
    try:
        result['cleanup'] = cleaner(directory / 'runs', map_config, result['descendants'])
    except Exception as exc:
        result['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    write_json(directory / 'supervision.json', result)
    write_review(directory, result)  # Partial HTML is always available after a native child exit.
    status = classify_case(directory, result, item)
    write_json(directory / 'status.json', status)
    return status


def _cell(value) -> str:
    return html.escape('—' if value is None else str(value))


def report(root: Path, manifest: dict) -> dict:
    """Refresh navigable HTML even for a paused or partially complete batch."""
    rows = []
    export_record = root / 'review_export.json'
    short_root = Path(json.loads(export_record.read_text(encoding='utf-8'))['output_dir']) if export_record.exists() else None
    exported_rows = {}
    if short_root and (short_root / 'export_summary.json').exists():
        exported_rows = {row['index']: row for row in json.loads((short_root / 'export_summary.json').read_text(encoding='utf-8'))['cases']}
    for index, item in enumerate(manifest['cases']):
        directory = latest_case_dir(root, index, item)
        status_path = directory / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'status': 'pending'}
        native_html = directory / 'evaluation_output/evaluation/cases' / item['case_id'] / 'review.html'
        exported = exported_rows.get(index + 1, {}).get('attempts', [])
        if exported:
            native_html = short_root / exported[-1]['review']
        row = {**item, **status, 'index': index + 1,
               'attempt_dir': directory.relative_to(root).as_posix(),
               'html': native_html if native_html.exists() else directory / 'review.html'}
        rows.append(row)
    summary = {'model': manifest['model'], 'dataset_sha256': manifest['dataset_sha256'],
               'cases': [{**row, 'html': Path(os.path.relpath(row['html'], root)).as_posix()} for row in rows],
               'counts': dict(Counter(row['status'] for row in rows))}
    write_json(root / 'summary.json', summary)
    if manifest.get('baseline_manifest'):
        baseline_path = Path(manifest['baseline_manifest']).parent / 'summary.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        previous = {r['case_id']: r for r in baseline['cases']}
        comparison = {'limitations': manifest['comparison_limitations'],
                      'baseline_summary_sha256': sha256(baseline_path),
                      'source_changes': manifest['source_changes_from_baseline'],
                      'cases': [{'case_id': r['case_id'], 'baseline': previous.get(r['case_id']),
                                 'current': r} for r in summary['cases']]}
        write_json(root / 'comparison.json', comparison)
        (root / 'comparison.html').write_text(
            '<!doctype html><meta charset="utf-8"><h1>Historical paired comparison</h1>'
            '<p>Pending cases are not navigation failures. Tool counts alone do not establish better behavior.</p><pre>'
            + html.escape(json.dumps(comparison, ensure_ascii=False, indent=2)) + '</pre>', encoding='utf-8')
    for name in dict.fromkeys(row['benchmark'] for row in rows):
        group = [row for row in rows if row['benchmark'] == name]
        completed = [row for row in group if row['status'] == 'completed']
        scored = [row for row in completed if row.get('metrics') and 'success' in row['metrics']]
        by_level = {}
        for label in ('low', 'medium', 'high') if name == 'aerial' else ('easy', 'hard'):
            subset = [row for row in scored if row['difficulty'] == label]
            by_level[label] = {'planned': sum(row['difficulty'] == label for row in group),
                               'scored': len(subset),
                               'success': sum(bool(row['metrics'].get('success')) for row in subset),
                               'success_rate': (sum(bool(row['metrics'].get('success')) for row in subset)
                                                / len(subset) if subset else None)}
        by_level['overall'] = {'planned': len(group), 'scored': len(scored),
                               'success': sum(bool(row['metrics'].get('success')) for row in scored),
                               'success_rate': (sum(bool(row['metrics'].get('success')) for row in scored)
                                                / len(scored) if scored else None)}
        title = 'AerialVLN val_seen' if name == 'aerial' else 'TravelUAV test_seen'
        cells = []
        for row in group:
            metrics = row.get('metrics') or {}
            link = Path(os.path.relpath(row['html'], root)).as_posix()
            error = metrics.get('distance_to_goal' if name == 'aerial' else 'navigation_error')
            token = (row.get('token_usage') or {}).get('total_tokens')
            cells.append('<tr>' + ''.join(f'<td>{_cell(value)}</td>' for value in
                         (row['index'], row['case_id'], row['scene'], row['difficulty'], row['status'],
                          metrics.get('success'), error, metrics.get('uav_action_steps'),
                          round(row.get('elapsed_seconds', 0), 1) if row.get('elapsed_seconds') else None,
                          token, row.get('plan_calls'), row.get('knowledge_calls'),
                          row.get('first_flight_preceded_by_plan'),
                          row.get('invalid_argument_rejections'), row.get('terminal_reference_rejections'),
                          row.get('infrastructure_error') or (row.get('history') or {}).get('infrastructure_error'),
                          row.get('exit_code'))) +
                         f'<td><a href="{html.escape(link, quote=True)}">HTML</a></td></tr>')
        body = ('<!doctype html><meta charset="utf-8"><title>' + title + '</title><h1>' + title + '</h1>' +
                '<p>Only completed, scored navigation cases enter success rates. Pending and infrastructure failures are separate.</p>' +
                '<pre>' + html.escape(json.dumps(by_level, ensure_ascii=False, indent=2)) + '</pre>' +
                '<table border="1"><tr><th>#</th><th>Case</th><th>Scene</th><th>Difficulty</th><th>Status</th>' +
                '<th>Success</th><th>Endpoint error (m)</th><th>Actions</th><th>Seconds</th><th>Tokens</th>' +
                '<th>Plan</th><th>Knowledge</th><th>Plan before flight</th><th>Invalid args</th><th>Terminal rejects</th>' +
                '<th>Infrastructure/current or historical</th><th>Exit</th><th>Review</th></tr>' +
                ''.join(cells) + '</table>')
        (root / f'{name}_summary.html').write_text(body, encoding='utf-8')
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Frozen FlyAgent evaluation</h1>'
                                    '<p><a href="aerial_summary.html">AerialVLN summary</a></p>'
                                    + ('<p><a href="travel_summary.html">TravelUAV summary</a></p>' if any(r['benchmark'] == 'travel' for r in rows) else '') +
                                    '<p><a href="summary.json">All cases and status</a></p>', encoding='utf-8')
    return summary


def prepare(root: Path, *, source_manifest: Path | None = None) -> dict:
    if root.exists():
        raise ValueError(f'Output exists; refusing to replace frozen manifest: {root}')
    if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
        raise ValueError('Less than 10 GiB free before batch preparation')
    manifest = import_aerial_manifest(source_manifest) if source_manifest else freeze_manifest()
    root.mkdir(parents=True)
    write_json(root / 'manifest.json', manifest)
    report(root, manifest)
    return manifest


def refresh_reviews(root: Path, review_root: Path | None = None) -> None:
    """Presentation failures pause scheduling without changing navigation results."""
    from tests.manual.export_benchmark_review import export_batch
    destination = (review_root or root.parent / 'reviews' / root.name).resolve()
    try:
        export_batch(root, destination)
    except Exception as exc:
        write_json(root / 'review_failure.json', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    write_json(root / 'review_export.json', {'output_dir': str(destination)})


def execute(root: Path, *, resume: bool = False, review_root: Path | None = None) -> dict:
    if not (root / 'manifest.json').exists():
        prepare(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    validate_manifest_cases(manifest['cases'])
    if load_model_config('.env').model_name != manifest['model']:
        raise ValueError('Local model no longer matches frozen model')
    if manifest.get('model_config') and load_model_config('.env').redacted_summary() != manifest['model_config']:
        raise ValueError('Frozen model configuration changed')
    if manifest.get('requires_admission'):
        admission_path = root / 'admission' / 'admission.json'
        admission = json.loads(admission_path.read_text(encoding='utf-8')) if admission_path.exists() else {}
        if admission.get('status') != 'passed' or admission.get('config') != manifest['model_config']:
            raise ValueError('Matching successful API admission required before any scene starts')
    preflight(root, manifest)
    refresh_reviews(root, review_root)
    for index, item in enumerate(manifest['cases']):
        directory = latest_case_dir(root, index, item)
        status_file = directory / 'status.json'
        if status_file.exists():
            previous = json.loads(status_file.read_text(encoding='utf-8'))
            if previous['status'] in ('completed', 'model_error'):
                continue
            if not resume:
                print(f'PAUSED at {index + 1}: existing {previous["status"]}; inspect before --resume', flush=True)
                break
            if not previous.get('cleanup_verified'):
                print(f'PAUSED at {index + 1}: cleanup must be verified before --resume', flush=True)
                break
            attempts = list(directory.parent.glob(case_dir(root, index, item).name + '_recovery_*'))
            directory = directory.parent / f'{case_dir(root, index, item).name}_recovery_{len(attempts) + 1:02d}'
        if directory.exists():
            print(f'PAUSED at {index + 1}: unclassified existing output', flush=True)
            break
        if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
            print(f'PAUSED at {index + 1}: low free disk space', flush=True)
            break
        status = run_one(root, index, item, directory=directory)
        refresh_reviews(root, review_root)
        report(root, manifest)
        print(json.dumps({'case': index + 1, 'id': item['case_id'], **status}, ensure_ascii=False), flush=True)
        if status['status'] not in ('completed', 'model_error'):
            print(f'PAUSED at {index + 1}: {status["status"]}', flush=True)
            break
        # First two from each benchmark are a mandatory artifact/cleanup gate.
        if index in (1, 11):
            if manifest.get('browser_gate'):
                import subprocess
                destination = (review_root or root.parent / 'reviews' / root.name).resolve()
                subprocess.run([sys.executable, str(ROOT / 'tests/manual/review_browser_probe.py'),
                                '--review-root', str(destination), '--cases', f'{index:02d}', f'{index + 1:02d}'], check=True)
            for gate in ((0, 1) if index == 1 else (10, 11)):
                gate_item = manifest['cases'][gate]
                gate_dir = latest_case_dir(root, gate, gate_item)
                if not (gate_dir / 'evaluation_output/evaluation/cases' /
                        gate_item['case_id'] / 'review.html').exists():
                    raise ValueError(f'Pilot case HTML missing: {gate_item["case_id"]}')
                if not json.loads((gate_dir / 'status.json').read_text(encoding='utf-8'))['cleanup_verified']:
                    raise ValueError(f'Pilot case cleanup failed: {gate_item["case_id"]}')
    return report(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'child'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, help='Import the exact historical AerialVLN ten cases')
    parser.add_argument('--case-json', help=argparse.SUPPRESS)
    parser.add_argument('--dataset-hash', help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true', help='Retry only an infrastructure-interrupted case in a new directory')
    parser.add_argument('--review-root', type=Path, help='Short independent review directory; defaults to ../reviews/<batch>')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root.resolve(), source_manifest=args.source_manifest)
        print(json.dumps({'manifest': str(args.root / 'manifest.json'),
                          'cases': len(result['cases'])}, ensure_ascii=False))
    elif args.command == 'run':
        result = execute(args.root.resolve(), resume=args.resume, review_root=args.review_root)
        print(json.dumps({'counts': result['counts'], 'index': str(args.root / 'index.html')}, ensure_ascii=False))
    else:
        if not args.case_json or not args.dataset_hash:
            parser.error('child requires --case-json and --dataset-hash')
        child(args.root.resolve(), json.loads(args.case_json), args.dataset_hash)


if __name__ == '__main__':
    main()
