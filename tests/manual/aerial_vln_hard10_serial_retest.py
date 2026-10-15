"""Serial hard-ten replay of an immutable parallel navigation version.

Copies the complete previous source byte-for-byte. Only an independently
frozen controller admits one task at a time on its original GPU/port lane.
Existing interface admissions are reused with explicit provenance; resource
and environment checks are fresh. No scores or navigation failures are retried.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from tests.manual import aerial_vln_plan_context_probe as probe
from tests.manual.aerial_vln_hard10_parallel_probe import resource_check
from evaluation.benchmarks.aerial_vln import parallel_campaign as controller
from evaluation.benchmarks.aerial_vln.parallel_recovery import ParallelRecoveryQueue

PREVIOUS = Path('H:/FlyAgent-Eval/avln_hard10_parallel_20261012_210310')
ADMISSIONS = ('admission.json', 'historical_request_replay.json', 'pixel_place_admission.json', 'place_memory_admission.json')


def serial_manifest(original, previous, root, state):
    """Retain all navigation settings and bind each case to its prior lane."""
    manifest = copy.deepcopy(original)
    if state['status'] != 'seen_complete' or len(original['tasks']) != 10:
        raise ValueError('A complete fixed ten-case parallel baseline is required')
    assignments = {}
    for task in manifest['tasks']:
        item = state['tasks'][ParallelRecoveryQueue.key(task)]
        if item['status'] != 'scored' or not item['attempts'][-1].get('cleanup_verified'):
            raise ValueError('Baseline case must be scored and cleaned')
        assignments[task['episode_id']] = item['attempts'][-1]['lane']
    config = manifest['config']
    config.update(map_config_path=str(root/'source/configs/maps/aerial_vln_s_remote.example.json'),
        context_config_path=str(root/'source/configs/context.json'), output_dir=str(root/'evaluation'), run_root=str(root/'runs'))
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), configs={'val_seen':config},
        previous_root=str(previous), previous_progress_sha256=probe.digest(previous/'progress.json'),
        overall_primary_baseline=str(previous), secondary_root=original['previous_root'], reference_roots=[],
        dispatch_concurrency=1, task_lane_assignments=assignments,
        retest={'kind':'serial_same_navigation_source', 'source_root':str(previous),
                'source_manifest_sha256':probe.digest(previous/'manifest.json'),
                'shared_host_workers':2, 'active_navigation_limit':1,
                'post_navigation_photo_repair_included':False},
        limits='Same frozen navigation source, same per-case GPU lane, one active task. One repeat cannot establish causality.')
    for cohort in manifest['cohorts']:
        cohort.update(primary_baseline=str(previous), secondary_baseline=original['previous_root'])
    manifest.pop('fingerprint',None)
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return manifest


def prepare(root, previous=PREVIOUS):
    if root.exists():
        raise ValueError('Output already exists; use run to resume')
    original = probe.guard(previous)
    state = probe.read(previous/'progress.json')
    root.mkdir(parents=True)
    for relative, expected in original['source_hashes'].items():
        source, target = previous/'source'/relative, root/'source'/relative
        if probe.digest(source) != expected:
            raise ValueError('Frozen navigation source changed: '+relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    manifest = serial_manifest(original,previous,root,state)
    probe.write(root/'manifest.json',manifest)
    ParallelRecoveryQueue(root/'progress.json',manifest['tasks'],manifest['fingerprint'],terminal_provider_errors=True)
    reused = []
    for filename in ADMISSIONS:
        value = probe.read(previous/filename)
        passed = probe.admission_passed(value,manifest['plan_mode']) if filename == 'admission.json' else (
            value.get('status') == 'passed' if filename == 'place_memory_admission.json' else value.get('passed'))
        if not passed:
            raise ValueError('Previous admission did not pass: '+filename)
        shutil.copy2(previous/filename,root/filename)
        reused.append({'file':filename,'source':str(previous/filename),'sha256':probe.digest(previous/filename)})
    # Reused replay reports contain relative links to their original evidence.
    # Preserve those bytes too; copying the aggregate JSON alone breaks review.
    for source in sorted((previous/'context_replay').rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(previous)
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        reused.append({'file':relative.as_posix(), 'source':str(source), 'sha256':probe.digest(source)})
    probe.write(root/'admission_reuse.json',{'scope':'Unchanged frozen navigation source and model; no new real interface calls implied.',
        'source_hashes_identical':True,'records':reused})
    probe.write(root/'environment.json',probe.environment_diagnostics())
    resource_check(root)
    controller.prepare_extension(root,manifest)
    report(root)
    return root


def report(root):
    probe.report(root)
    manifest = probe.read(root/'manifest.json')
    overlay_path = root/'parallel/execution_overlay.json'
    if overlay_path.exists():
        controller.verify_overlay(probe.read(overlay_path),manifest)
    # Host status is persisted on every polling cycle. Preserve an independent
    # maximum-active audit so two available workers cannot be mistaken for two
    # simultaneously flying tasks.
    rows=[]
    for file in sorted(root.glob('parallel/sessions/*/dispatch_audit.jsonl')):
        rows.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip())
    violations=[row for row in rows if len(row['active'])>1]
    previous = Path(manifest['previous_root'])
    source_equal = manifest['source_hashes'] == probe.read(previous/'manifest.json')['source_hashes']
    probe.write(root/'serial_isolation.json',{'source_hashes_identical':source_equal,'dispatch_events':len(rows),
        'max_active_tasks':max((len(row['active']) for row in rows),default=0), 'violations':violations,
        'task_lane_assignments':manifest['task_lane_assignments'],
        'passed':source_equal and bool(rows) and not violations})


def run(root):
    manifest=probe.guard(root)
    resource_check(root)
    overlay=probe.read(root/'parallel/execution_overlay.json')
    controller.verify_overlay(overlay,manifest)
    subprocess.run([sys.executable,'-u',overlay['controller_script'],'run','--output',str(root),'--workers','2'],check=True)
    report(root)
    if probe.read(root/'progress.json')['status'] != 'seen_complete':
        raise RuntimeError('Serial experiment stopped before all ten terminal outcomes')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.command](args.root.resolve())
