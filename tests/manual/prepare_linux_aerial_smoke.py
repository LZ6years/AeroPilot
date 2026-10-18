"""Freeze exactly two AerialVLN-S tasks with maps for all scene capture checks.

No UE or model calls. Dataset and executable layout must already exist.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks.core.campaign.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--env-root', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, default=39131)
    parser.add_argument('--graphics-context-limits', type=json.loads, default={},
                        help='Explicitly approved auxiliary G-only limits, JSON physical-index to MiB')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Choose a fresh output directory')
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, mode=0o700)
    def write(name, data):
        path = inputs / name
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return path
    data = {}
    for split in ['val_seen', 'val_unseen']:
        path = args.dataset_root / (split + '.json')
        data[split] = (json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest())
    scenes = sorted({str(e['scene_id']) for d, _ in data.values() for e in d['episodes']}, key=int)
    template = json.loads((ROOT / 'configs/airsim/aerialvln_env_1.settings.example.json').read_text())
    template.update(ApiServerPort=args.port, ClockSpeed=1)
    camera = template['Vehicles']['Drone_1']['Cameras']['front_0']
    camera.update(X=0.5, Y=0, Z=0, Pitch=0, Roll=0, Yaw=0)
    for capture in camera['CaptureSettings']:
        capture.update(Width=256, Height=256, FOV_Degrees=90)
    settings = write('settings.json', template)
    prototype = json.loads((ROOT / 'configs/maps/aerial_vln_linux_local.example.json').read_text())['maps'][0]
    maps = []
    for scene in scenes:
        item = deepcopy(prototype)
        directory = args.env_root / f'env_{scene}/env_{scene}/LinuxNoEditor'
        if not (directory / 'AirVLN.sh').is_file():
            raise FileNotFoundError(directory / 'AirVLN.sh')
        item.update(map_id='aerial_vln_local_' + scene, scene_id=scene, root=str(directory.resolve()))
        item['airsim'].update(port=args.port, settings_template=str(settings))
        maps.append(item)
    map_path = write('maps.json', dict(maps=maps, port_policy=dict(control_port_start=args.port, max_scene_num=1)))
    execution = dict(mode='local', gpu_indices=[args.gpu], gpu_uuids={str(args.gpu): args.gpu_uuid},
        require_cuda=False, report_profile='audit', graphics_context_limits_mib=args.graphics_context_limits,
        allow_shared_gpu=True, minimum_free_gpu_mib=4096)
    write('execution.json', execution)
    # Credentials stay private, outside source and Git; explicitly select protocol.
    env_path = inputs / '.env'
    env_path.write_text((ROOT / '.env').read_text() + '\nFLYAGENT_MODEL_API_PROTOCOL=chat_completions\n', encoding='utf-8')
    env_path.chmod(0o600)
    configs, identities = [], []
    for split, scene in [('val_seen', '2'), ('val_unseen', '7')]:
        dataset, digest = data[split]
        identity = sorted(str(e['episode_id']) for e in dataset['episodes'] if str(e['scene_id']) == scene)[0]
        config = json.loads((ROOT / 'evaluation/benchmarks/aerial_vln/configs/linux_local_smoke.example.json').read_text())
        config.update(dataset_root=str(args.dataset_root.resolve()), split=split, episode_ids=[identity],
            scene_map={s: 'aerial_vln_local_' + s for s in scenes}, map_config_path=str(map_path),
            env_path=str(env_path), context_config_path=str(ROOT / 'configs/context.json'),
            dataset_version='canonical-aerialvln-s-' + split, expected_split_sha256=digest)
        configs.append(write(split + '.json', config))
        identities.append(dict(split=split, scene=scene, episode_id=identity))
    prepare(output, configs, 'aerial_vln', root=ROOT, plugin_config=ROOT / 'configs/plugins.pixel_map.json', execution=execution)
    manifest = json.loads((output / 'manifest.json').read_text())
    if len(manifest['tasks']) != 2:
        raise RuntimeError('Refusing a queue with anything other than two tasks')
    write('test_scope.json', dict(tasks=identities, scenes=scenes, full_evaluation_authorized=False))
    print(json.dumps(identities))


if __name__ == '__main__':
    main()
