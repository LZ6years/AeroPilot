"""Freeze a complete Seen campaign and outcome-independent audit sample.

This evaluation-only entry validates released split identities, uses installed
Linux assets, and delegates execution/persistence to the shared campaign. It
does not launch a simulator, make model requests, or change flight semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from evaluation.benchmarks.core.campaign.prepare import prepare
from evaluation.benchmarks.core.campaign.retention import select_full_audit
from evaluation.benchmarks.core.report_io import write_json
from evaluation.benchmarks.travel_uav.dataset import (
    _deduplicate_json_paths, _safe_episode_id, load_travel_uav_index,
)

ROOT = Path(__file__).resolve().parents[3]


def audit_source(index, source_split):
    """Compare every released episode identity, rather than paper totals alone."""
    cases, provenance = load_travel_uav_index(index, split='test_seen')
    rows = json.loads(Path(source_split).read_text(encoding='utf-8-sig'))
    paths = _deduplicate_json_paths(rows, Path(source_split))
    identities = [_safe_episode_id(*path.split('/')[-3:-1]) for path in paths]
    if len(set(identities)) != len(identities):
        raise ValueError('Released Seen paths collapse to duplicate episode identities')
    indexed = {case.episode_id for case in cases}
    missing, extra = sorted(set(identities) - indexed), sorted(indexed - set(identities))
    digest = hashlib.sha256(Path(source_split).read_bytes()).hexdigest()
    record = dict(index_sha256=provenance.index_sha256, source_sha256=digest,
        expected_source_sha256=provenance.source_split_sha256,
        source_hash_matches=digest == provenance.source_split_sha256,
        frame_rows=len(rows), released_unique_episodes=len(identities), indexed_episodes=len(cases),
        missing_from_index=missing, extra_in_index=extra, published_seen_count=1410,
        published_count_matches=len(cases) == 1410,
        interpretation='Actual released split identities are authoritative; paper-count discrepancy is retained, not corrected by filtering.')
    if missing or extra:
        raise ValueError('Seen index differs from the released episode identities: ' + json.dumps(record))
    return cases, record


def build_catalog(cases, env_root, settings_path, port):
    """Resolve official close-loop and CARLA Town launcher layouts."""
    prototype = json.loads((ROOT / 'configs/maps/travel_uav_linux_local.example.json').read_text())['maps'][0]
    packages = []
    for scene in sorted({case.map_name for case in cases}):
        item = deepcopy(prototype)
        if scene.startswith('Carla_'):
            directory = Path(env_root) / 'carla_town_envs/carla_town_envs' / scene.removeprefix('Carla_') / 'LinuxNoEditor'
            launcher = 'CarlaUE4.sh'
        else:
            directory = Path(env_root) / 'closeloop_envs/closeloop_envs'
            launcher = scene + '.sh'
        if not (directory / launcher).is_file():
            raise FileNotFoundError(directory / launcher)
        item.update(map_id='travel_uav_local_' + scene, scene_id=scene, root=str(directory.resolve()))
        item['launch'].update(linux=launcher, ready_timeout_sec=120)
        item['launch']['args'] = [arg for arg in item['launch']['args'] if not arg.lower().startswith('-graphicsadapter')]
        item['airsim'].update(port=port, settings_template=str(settings_path))
        packages.append(item)
    return dict(maps=packages, port_policy=dict(control_port_start=port, max_scene_num=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--source-split', type=Path, required=True)
    parser.add_argument('--expected-index-sha256', required=True)
    parser.add_argument('--env-root', type=Path, required=True)
    parser.add_argument('--env-path', type=Path, default=ROOT / '.env')
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--allow-shared-gpu', action='store_true')
    parser.add_argument('--port', type=int, default=39141)
    parser.add_argument('--image-budget', type=int, default=30)
    parser.add_argument('--full-audit-count', type=int, default=100)
    parser.add_argument('--full-seed-bytes', type=int, required=True,
                        help='Measured conservative initial full-record size; replaced by observed receipts')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh campaign directory')
    cases, source_audit = audit_source(args.index, args.source_split)
    if source_audit['index_sha256'] != args.expected_index_sha256:
        raise ValueError('Unexpected TravelUAV index SHA256')
    tasks = [dict(dataset='travel_uav', split=c.split, scene_id=c.map_name,
                  difficulty=c.difficulty, episode_id=c.episode_id) for c in cases]
    full = select_full_audit(tasks, args.full_audit_count)
    trial = []
    for difficulty in ('easy', 'hard'):
        ranked = sorted((t for t in full if t['difficulty'] == difficulty),
                        key=lambda t: hashlib.sha256(('42:' + t['episode_id']).encode()).hexdigest())
        if len(ranked) < len(args.gpus):
            raise ValueError('Audit sample cannot supply both difficulty trial waves')
        trial.extend(t['episode_id'] for t in ranked[:len(args.gpus)])
    inputs = args.output.resolve() / 'inputs'
    inputs.mkdir(parents=True, mode=0o700)
    write_json(inputs / 'source_split_audit.json', source_audit)
    context = json.loads((ROOT / 'configs/context.json').read_text())
    context['history_projection']['media']['max_request_images'] = args.image_budget
    write_json(inputs / 'context.json', context)
    settings = inputs / 'settings.json'
    shutil.copy2(ROOT / 'configs/airsim/travel_uav_official.settings.example.json', settings)
    maps = build_catalog(cases, args.env_root, settings, args.port)
    map_path = inputs / 'maps.json'
    write_json(map_path, maps)
    response = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = {str(int(row.split(',')[0])): row.split(',')[1].strip() for row in response.splitlines()}
    execution = dict(mode='local', gpu_indices=args.gpus,
        gpu_uuids={str(gpu): devices[str(gpu)] for gpu in args.gpus}, require_cuda=False,
        report_profile='paper_process', allow_shared_gpu=args.allow_shared_gpu,
        minimum_free_gpu_mib=4096, graphics_context_limits_mib={})
    write_json(inputs / 'execution.json', execution)
    env = inputs / '.env'
    shutil.copy2(args.env_path, env)
    env.chmod(0o600)
    config = json.loads((ROOT / 'evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json').read_text())
    config.update(index_path=str(args.index.resolve()), expected_index_sha256=args.expected_index_sha256,
        limit_per_difficulty=None, case_ids=None, env_path=str(env),
        context_config_path=str(inputs / 'context.json'), map_config_path=str(map_path),
        scene_map={p['scene_id']: p['map_id'] for p in maps['maps']})
    config_path = inputs / 'test_seen.json'
    write_json(config_path, config)
    prepare(args.output, [config_path], 'travel_uav', root=ROOT,
        plugin_config=ROOT / 'configs/plugins.pixel_map.json', execution=execution,
        full_audit_count=args.full_audit_count, full_seed_bytes=args.full_seed_bytes,
        technical_trial_ids=trial)
    write_json(inputs / 'audit_selection.json', dict(seed=42, method='scene_difficulty_largest_remainder_then_sha256',
        tasks=full, technical_trial_ids=trial,
        population_by_scene=dict(Counter(t['scene_id'] for t in tasks)),
        full_by_scene=dict(Counter(t['scene_id'] for t in full))))
    print(json.dumps(dict(output=str(args.output), tasks=len(tasks), full=len(full),
                          compact=len(tasks) - len(full), technical_trial=len(trial))))


if __name__ == '__main__':
    main()
