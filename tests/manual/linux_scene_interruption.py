"""Validate owned UE process-group recovery after killing its launch parent.

Uses one map from a previously verified capture catalog; no model/API requests.
An independent CPU sentinel must survive cleanup. Never signals unowned groups.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks.core.campaign.local_execution import cleanup_local
from evaluation.benchmarks.core.campaign.parallel import local_gpu_idle, REMOTE_GPU_INSPECTION, gpu_violation
from evaluation.benchmarks.core.supervision import cleanup_verified
from src.runtime.configs.map_config import load_map_config
from src.sim_server.server import SimulationServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--execution-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution_config.read_text())
    while not local_gpu_idle(execution):
        print('Waiting for GPU admission', flush=True)
        time.sleep(30)
    packages, policy = load_map_config(args.maps)
    args.output.mkdir(parents=True, exist_ok=False)
    server = SimulationServer([packages[0]], policy, run_dir=args.output, show_display=False)
    sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True)
    record, scene = {'passed': False}, None
    try:
        scene = server.start_scene(packages[0].map_id)
        sample = json.loads(subprocess.check_output([sys.executable, '-c', REMOTE_GPU_INSPECTION,
            json.dumps([{'pid': scene.process_id}])], text=True, timeout=15))
        if gpu_violation(sample, execution['gpu_indices'], execution['graphics_context_limits_mib']):
            raise RuntimeError('Unexpected GPU ownership')
        # The server registered this launch PID and persisted a random ownership
        # token. Recovery independently verifies every member before signaling.
        record['launch_pid'] = scene.process_id
        os.kill(scene.process_id, signal.SIGKILL)
        record['cleanup'] = cleanup_local(args.output)
        record['sentinel_survived'] = sentinel.poll() is None
        record['passed'] = cleanup_verified(record['cleanup']) and record['sentinel_survived']
    except Exception as exc:
        record['error'] = str(exc)
    finally:
        if scene is not None:
            try:
                server.stop_scene(scene.scene_instance_id)
            except Exception as exc:
                record['stop_error'] = str(exc)
                record['passed'] = False
        record['final_cleanup'] = cleanup_local(args.output)
        record['passed'] = record['passed'] and cleanup_verified(record['final_cleanup'])
        sentinel.terminate()
        sentinel.wait(timeout=10)
        (args.output / 'interruption_validation.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
