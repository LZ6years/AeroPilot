"""Prepare a reproducible Linux-local scene cohort; never start full evaluation.

Uses canonical split files, selects two lexicographically first episode IDs per
scene and delegates freezing/queue/report ownership to the shared campaign.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from evaluation.benchmarks.core.campaign.prepare import prepare
from evaluation.benchmarks.core.campaign.support import ROOT
from evaluation.benchmarks.core.report_io import write_json


def select_scene_episodes(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[str(episode['scene_id'])].append(str(episode['episode_id']))
    selected = {}
    for scene in sorted(groups, key=int):
        identities = sorted(set(groups[scene]))
        if len(identities) < 2:
            raise ValueError('Scene needs two distinct episodes: ' + scene)
        selected[scene] = identities[:2]
    return selected


def prepare_stability(output, dataset_root, env_root, template, execution_config, *, root=ROOT):
    """Write explicit deployment inputs, then freeze the effective run."""
    output, root = Path(output).resolve(), Path(root).resolve()
    if output.exists():
        raise ValueError('Use a new stability output directory')
    settings = json.loads(Path(template).read_text(encoding='utf-8'))
    for capture in settings['Vehicles']['Drone_1']['Cameras']['front_0']['CaptureSettings']:
        if capture['ImageType'] in (0, 2, 3) and (capture['Width'], capture['Height'], capture['FOV_Degrees']) != (256, 256, 90):
            raise ValueError('Unexpected camera template')
    selections = {}
    for split in ('val_seen', 'val_unseen'):
        path = Path(dataset_root) / (split + '.json')
        selections[split] = (select_scene_episodes(json.loads(path.read_text())['episodes']),
                             hashlib.sha256(path.read_bytes()).hexdigest())
    scenes = sorted({scene for selected, _ in selections.values() for scene in selected}, key=int)
    for scene in scenes:
        script = Path(env_root) / f'env_{scene}/env_{scene}/LinuxNoEditor/AirVLN.sh'
        if not script.is_file():
            raise FileNotFoundError(script)
    inputs = output / 'inputs'
    inputs.mkdir(parents=True)
    settings['ApiServerPort'] = 39131
    settings_path = inputs / 'settings.json'
    write_json(settings_path, settings)
    prototype = json.loads((root / 'configs/maps/aerial_vln_linux_local.example.json').read_text())['maps'][0]
    maps = []
    for scene in scenes:
        package = deepcopy(prototype)
        package.update(map_id='aerial_vln_local_' + scene, scene_id=scene,
                       root=str(Path(env_root) / f'env_{scene}/env_{scene}/LinuxNoEditor'))
        package['airsim']['settings_template'] = str(settings_path)
        maps.append(package)
    map_path = inputs / 'maps.json'
    write_json(map_path, {'port_policy': {'control_port_start': 39131, 'max_scene_num': 1}, 'maps': maps})
    paths = []
    selection_report = {}
    for split, (selected, digest) in selections.items():
        config = json.loads((root / 'evaluation/benchmarks/aerial_vln/configs/linux_local_smoke.example.json').read_text())
        config.update(dataset_root=str(Path(dataset_root).resolve()), split=split,
            episode_ids=[identity for identities in selected.values() for identity in identities],
            scene_map={scene: 'aerial_vln_local_' + scene for scene in selected}, map_config_path=str(map_path),
            env_path=str(root / '.env'), context_config_path=str(root / 'configs/context.json'),
            dataset_version='canonical-aerialvln-s-' + split, expected_split_sha256=digest)
        path = inputs / (split + '.json')
        write_json(path, config)
        paths.append(path)
        selection_report[split] = selected
    write_json(inputs / 'selection.json', selection_report)
    execution = json.loads(Path(execution_config).read_text(encoding='utf-8'))
    result = prepare(output, paths, 'aerial_vln', root=root,
                     plugin_config=root / 'configs/plugins.pixel_map.json', execution=execution)
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    write_json(output / 'deployment.json', {'commit': revision, 'selection': selection_report,
        'settings_sha256': hashlib.sha256(settings_path.read_bytes()).hexdigest(), 'phase': 'scene_stability',
        'full_evaluation_authorized': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--env-root', type=Path, required=True)
    parser.add_argument('--settings-template', type=Path, required=True)
    parser.add_argument('--execution-config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_stability(args.output, args.dataset_root, args.env_root,
                                      args.settings_template, args.execution_config), ensure_ascii=False))


if __name__ == '__main__':
    main()
