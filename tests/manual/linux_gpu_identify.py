"""Continuous, explicitly selected Linux GPU load; Ctrl+C stops owned workers.

Standalone: copy this file to the server and run it with a CUDA-enabled PyTorch
environment. Requires torch, psutil and nvidia-smi; imports no project modules.
No duration/temperature cutoff, model weights, API calls or driver tuning.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def parse_gpus(value):
    """Interpret CLI numbers as nvidia-smi indices, never CUDA ordinals."""
    parts = value.split(',')
    if not parts or any(not part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError('Use comma-separated nonnegative indices, e.g. 0,2,3')
    indices = [int(part) for part in parts]
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError('GPU indices must not repeat')
    return indices


def smi(arguments):
    result = subprocess.run(['nvidia-smi', *arguments], capture_output=True,
                            text=True, timeout=10, check=True)
    return result.stdout


def inventory():
    rows = csv.reader(io.StringIO(smi(['--query-gpu=index,uuid,pci.bus_id,name',
                                     '--format=csv,noheader,nounits'])))
    result = {}
    for row in rows:
        index, uuid, pci, name = (part.strip() for part in row)
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid):
            raise RuntimeError('Expected a physical GPU UUID: ' + uuid)
        result[int(index)] = {'index': int(index), 'uuid': uuid, 'pci': pci, 'name': name}
    if not result:
        raise RuntimeError('No NVIDIA GPUs found')
    return result


def resolve_devices(indices, available):
    missing = set(indices) - set(available)
    if missing:
        raise ValueError('GPU indices do not exist: ' + str(sorted(missing)))
    return [available[index] for index in indices]


def numeric(text):
    try:
        return float((text or '').split()[0])
    except (ValueError, IndexError):
        return None


def parse_snapshot(xml):
    """Read graphics AND compute ownership; absent ownership data is an error."""
    result = {}
    for gpu in ET.fromstring(xml).findall('gpu'):
        uuid = gpu.findtext('uuid')
        processes = gpu.find('processes')
        if not uuid or processes is None:
            raise RuntimeError('GPU identity/process monitoring unavailable')
        if (processes.text or '').strip():
            raise RuntimeError('GPU process monitoring returned unsupported data')
        owners = []
        for process in processes.findall('process_info'):
            owners.append({'pid': int(process.findtext('pid')), 'type': process.findtext('type'),
                           'name': process.findtext('process_name', '')})
        values = {'temperature': numeric(gpu.findtext('temperature/gpu_temp')),
                  'utilization': numeric(gpu.findtext('utilization/gpu_util')),
                  'memory_mib': numeric(gpu.findtext('fb_memory_usage/used')),
                  'power_w': numeric(gpu.findtext('gpu_power_readings/power_draw') or
                                     gpu.findtext('power_readings/power_draw')),
                  'owners': owners}
        if any(values[key] is None for key in ('temperature', 'utilization', 'memory_mib')):
            raise RuntimeError('Required GPU telemetry unavailable: ' + uuid)
        result[uuid] = values
    return result


def check_occupancy(devices, snapshot, own_pids=()):
    """Desktop exceptions are explicit; all other external owners block load."""
    for device in devices:
        if device['uuid'] not in snapshot:
            raise RuntimeError('Selected GPU disappeared: ' + device['uuid'])
        for owner in snapshot[device['uuid']]['owners']:
            if owner['pid'] in own_pids:
                continue
            desktop = owner['type'] == 'G' and Path(owner['name']).name in {'Xorg', 'Xwayland'}
            if desktop:
                continue
            raise RuntimeError(f"GPU {device['index']} occupied: PID {owner['pid']} "
                               f"type={owner['type']} name={owner['name']}")


def worker_command(device, parent_pid, parent_created):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', device['uuid'],
               '--parent-pid', str(parent_pid), '--parent-created', str(parent_created)]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': device['uuid'], 'PYTHONUNBUFFERED': '1'}
    return command, environment


def parent_alive(pid, created):
    import psutil
    try:
        parent = psutil.Process(pid)
        return parent.create_time() == created and parent.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def worker(uuid, parent_pid, parent_created):
    """The child alone imports torch and creates its single CUDA context."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('Worker UUID isolation missing')
    import torch
    available = torch.cuda.is_available()
    print(json.dumps({'pid': os.getpid(), 'uuid': uuid, 'python': sys.executable,
                      'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'cuda_available': available}), flush=True)
    if not available or torch.cuda.device_count() != 1:
        raise RuntimeError('Worker requires exactly one visible CUDA GPU; no CPU fallback')
    if not parent_alive(parent_pid, parent_created):
        return
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    # Three float32 9216-square buffers use 972 MiB, plus CUDA/library overhead.
    # Reuse the output to avoid allocator churn or growing memory usage.
    with torch.inference_mode():
        a = torch.rand((9216, 9216), device='cuda:0', dtype=torch.float32)
        b = torch.rand_like(a)
        out = torch.empty_like(a)
        torch.cuda.synchronize()
        print(f'READY uuid={uuid} allocated_mib={torch.cuda.memory_allocated() / 2**20:.1f}', flush=True)
        while parent_alive(parent_pid, parent_created):
            for _ in range(8):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()


def cleanup_workers(workers):
    """Popen handles belong only to this invocation; never kill by GPU/name."""
    for process in workers:
        if process.poll() is None:
            process.terminate()
    errors = []
    for process in workers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                errors.append(process.pid)
    if errors:
        raise RuntimeError('Workers did not exit: ' + str(errors))


def run(indices):
    """Supervise load until an explicit stop, worker failure or monitoring error."""
    import psutil
    devices = resolve_devices(indices, inventory())
    snapshot = parse_snapshot(smi(['-q', '-x']))
    check_occupancy(devices, snapshot)
    for device in devices:
        print(json.dumps(device, ensure_ascii=False), flush=True)
        desktops = snapshot[device['uuid']]['owners']
        if desktops:
            print('Existing desktop processes: ' + json.dumps(desktops), flush=True)
    print('Continuous load; no time/temperature cutoff. Press Ctrl+C to stop.', flush=True)
    stop = threading.Event()
    previous = {}
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, 'SIGHUP'):
        signals.append(signal.SIGHUP)
    for sig in signals:
        previous[sig] = signal.signal(sig, lambda *_: stop.set())
    workers = []
    try:
        parent_created = psutil.Process().create_time()
        for device in devices:
            if stop.is_set():
                break
            command, environment = worker_command(device, os.getpid(), parent_created)
            workers.append(subprocess.Popen(command, env=environment, start_new_session=True))
        while not stop.is_set():
            failed = [(p.pid, p.returncode) for p in workers if p.poll() is not None]
            if failed:
                raise RuntimeError('GPU worker exited: ' + str(failed))
            snapshot = parse_snapshot(smi(['-q', '-x']))
            check_occupancy(devices, snapshot, {p.pid for p in workers})
            for device in devices:
                sample = snapshot[device['uuid']]
                print(f"GPU {device['index']} | util={sample['utilization']}% | "
                      f"temp={sample['temperature']}C | power={sample['power_w']}W | "
                      f"memory={sample['memory_mib']}MiB", flush=True)
            stop.wait(2)
    finally:
        try:
            cleanup_workers(workers)
            print('All owned load workers exited.', flush=True)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=parse_gpus, help='Required physical nvidia-smi indices, e.g. 0,2,3')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--parent-created', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and args.gpus is None:
        parser.error('--gpus is required')
    if sys.platform != 'linux':
        parser.error('GPU load execution is supported only on Linux')
    try:
        if args.worker:
            if args.parent_pid is None or args.parent_created is None:
                parser.error('Worker requires parent identity')
            worker(args.worker, args.parent_pid, args.parent_created)
        else:
            run(args.gpus)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
