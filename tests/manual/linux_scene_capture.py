"""Probe every configured Linux scene's vehicle, RGB/depth and GPU; no model API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.core.campaign.parallel import REMOTE_GPU_INSPECTION, read_json, gpu_violation, local_gpu_idle, execution_settings, execution_lanes
from evaluation.benchmarks.core.campaign.support import environment_diagnostics
from evaluation.benchmarks.core.report_io import write_json
from evaluation.benchmarks.core.supervision import cleanup_verified
from evaluation.benchmarks.core.campaign.local_execution import cleanup_local
from src.runtime.configs.map_config import load_map_config
from src.sim_server.server import SimulationServer
from src.adapters.airsim.client import close_airsim_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--execution-config', type=Path)
    parser.add_argument('--lane', default='A')
    parser.add_argument('--scene', action='append', help='Probe only selected scenes from the complete catalog')
    parser.add_argument('--travel-index', type=Path, help='Verify Seen target assets and one placement per map')
    args = parser.parse_args()
    execution = read_json(args.execution_config) if args.execution_config else None
    limits = execution.get('graphics_context_limits_mib', {}) if execution else {}
    if execution and execution_settings({'execution': execution})['gpu_indices'][execution_lanes(execution).index(args.lane)] != args.gpu:
        raise ValueError('Capture GPU must match execution policy')
    import airsim
    packages, policy = load_map_config(args.maps)
    if args.scene:
        packages = [p for p in packages if str(p.scene_id) in args.scene]
        if {str(p.scene_id) for p in packages} != set(args.scene):
            raise ValueError('Capture requested an unknown scene')
    cases = []
    if args.travel_index:
        from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
        cases, _ = load_travel_uav_index(args.travel_index, split='test_seen')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'environment.json', environment_diagnostics())
    results = []
    for package in packages:
        while execution and not local_gpu_idle(execution, args.lane):
            print('Waiting for GPU admission before scene ' + str(package.scene_id), flush=True)
            time.sleep(30)
        directory = args.output / str(package.scene_id)
        server = SimulationServer([package], policy, run_dir=directory, show_display=False)
        scene, client = None, None
        record = {'scene_id': package.scene_id, 'passed': False}
        started = time.monotonic()
        try:
            scene = server.start_scene(package.map_id)
            record['startup_seconds'] = time.monotonic() - started
            sample = json.loads(subprocess.run([sys.executable, '-c', REMOTE_GPU_INSPECTION,
                json.dumps([{'pid': scene.process_id}])], capture_output=True, text=True, check=True, timeout=15).stdout)
            record['gpu'] = sample
            allowed = execution['gpu_indices'] if execution else [args.gpu]
            if args.gpu not in sample['owned'][0]['observed_gpus'] or gpu_violation(sample, allowed, limits):
                raise RuntimeError('UE rendering GPU does not match authorized physical card')
            client = airsim.VehicleClient(ip='127.0.0.1', port=scene.control_port, timeout_value=30)
            record['vehicles'] = client.listVehicles()
            vehicle = package.airsim['vehicle_name']
            camera = package.airsim['camera_name']
            if vehicle not in record['vehicles']:
                raise RuntimeError('Expected vehicle missing')
            settings = read_json(Path(package.airsim['settings_template']))
            captures = settings['Vehicles'][vehicle]['Cameras'][camera]['CaptureSettings']
            types = [int(c['ImageType']) for c in captures if int(c['ImageType']) in {0, 2, 3}]
            if 0 not in types or not set(types) & {2, 3}:
                raise ValueError('Configured camera requires RGB and metric depth')
            images = client.simGetImages([airsim.ImageRequest(camera, t, t != 0, False) for t in types], vehicle_name=vehicle)
            record['images'] = [{'type': i.image_type, 'width': i.width, 'height': i.height,
                                 'values': len(i.image_data_float) if i.pixels_as_float else len(i.image_data_uint8)} for i in images]
            expected = {int(c['ImageType']): (c['Width'], c['Height']) for c in captures if int(c['ImageType']) in types}
            if (len(images) != len(types) or any((i.width, i.height) != expected[i.image_type] for i in images)
                    or any(not i['values'] for i in record['images'])):
                raise RuntimeError('Invalid RGB/depth capture')
            if cases:
                from evaluation.benchmarks.travel_uav.adapter import TravelUavScenarioPayload
                from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController
                local_cases = sorted((c for c in cases if c.map_name == package.scene_id), key=lambda c: c.episode_id)
                required = sorted({c.target.asset_name for c in local_cases})
                available = set(client.simListAssets())
                record['target_assets'] = dict(required=required, missing=sorted(set(required) - available))
                if record['target_assets']['missing']:
                    raise RuntimeError('Required target assets are missing')
                controller = TravelUavScenarioController(cleanup_strategy='destroy_object')
                payload = TravelUavScenarioPayload(local_cases[0].target)
                try:
                    controller.prepare_episode(scene, payload)
                    record['target_placement_verified'] = True
                finally:
                    controller.cleanup_episode(scene, payload)
            record['passed'] = True
        except Exception as exc:
            record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        finally:
            if client is not None:
                close_airsim_client(client)
            if scene is not None:
                server.stop_scene(scene.scene_instance_id)
            record['cleanup'] = cleanup_local(directory)
            record['passed'] = record['passed'] and cleanup_verified(record['cleanup'])
            record['total_seconds'] = time.monotonic() - started
            results.append(record)
            write_json(args.output / 'capture_validation.json', {
                'passed': all(r['passed'] for r in results) and len(results) == len(packages),
                'map_sha256': hashlib.sha256(args.maps.read_bytes()).hexdigest(),
                'graphics_context_limits_mib': limits, 'scenes': results})
            print(json.dumps({key: record[key] for key in ('scene_id', 'passed', 'total_seconds')}, ensure_ascii=False), flush=True)
        if not record['passed']:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
