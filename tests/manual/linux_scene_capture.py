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

from evaluation.benchmarks.core.campaign.parallel import REMOTE_GPU_INSPECTION, read_json, gpu_violation, local_gpu_idle, execution_settings
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
    args = parser.parse_args()
    execution = read_json(args.execution_config) if args.execution_config else None
    limits = execution.get('graphics_context_limits_mib', {}) if execution else {}
    if execution and execution_settings({'execution': execution})['gpu_indices'] != [args.gpu]:
        raise ValueError('Capture GPU must match execution policy')
    import airsim
    packages, policy = load_map_config(args.maps)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'environment.json', environment_diagnostics())
    results = []
    for package in packages:
        while execution and not local_gpu_idle(execution):
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
            if args.gpu not in sample['owned'][0]['observed_gpus'] or gpu_violation(sample, [args.gpu], limits):
                raise RuntimeError('UE rendering GPU does not match authorized physical card')
            client = airsim.VehicleClient(ip='127.0.0.1', port=scene.control_port, timeout_value=30)
            record['vehicles'] = client.listVehicles()
            if 'Drone_1' not in record['vehicles']:
                raise RuntimeError('Expected vehicle missing')
            images = client.simGetImages([airsim.ImageRequest('front_0', 0, False, False),
                airsim.ImageRequest('front_0', 2, True, False), airsim.ImageRequest('front_0', 3, True, False)], vehicle_name='Drone_1')
            record['images'] = [{'type': i.image_type, 'width': i.width, 'height': i.height,
                                 'values': len(i.image_data_float) if i.pixels_as_float else len(i.image_data_uint8)} for i in images]
            if len(images) != 3 or any(i.width != 256 or i.height != 256 for i in images) or any(not i['values'] for i in record['images']):
                raise RuntimeError('Invalid RGB/depth capture')
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
