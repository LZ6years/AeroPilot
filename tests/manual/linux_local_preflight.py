"""Read-only Linux local deployment diagnostics. No scene or model API calls.

Run from the repository root with the server's explicit flyagent interpreter.
JSON is printed to stdout; redirect it to a diagnostic artifact if desired.
GPU identification is separate from UE adapter selection, which must be checked
against Vulkan enumeration / UE logs before starting a real test.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.configs.map_config import load_map_config
from src.sim_server.port_manager import PortManager
from src.sim_server.process_control import LocalSceneControl
from src.sim_server.process_manager import ProcessManager


def inspect_config(config_path: Path) -> dict:
    """Validate selected local maps and assets without reserving or launching."""
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    maps, policy = load_map_config(config['map_config_path'])
    checks = []
    for p in maps:
        if p.connection_mode != 'local' or p.connection.get('data_transport') != 'direct':
            raise ValueError('Preflight requires local/direct maps')
        executable = LocalSceneControl.resolve_executable(p)
        command = ProcessManager().build_command(executable, p.launch.get('args', []),
                    p.launch.get('method', 'direct'), '/tmp/preflight settings.json',
                    p.launch.get('settings_arg_style'))
        template = p.airsim.get('settings_template')
        checks.append(dict(map_id=p.map_id, command=command, template=template,
                           template_exists=bool(template and Path(template).is_file())))
    asset = config.get('dataset_root') or config.get('index_path')
    ports = list(range(policy.control_port_start, policy.control_port_start + policy.max_scene_num))
    free = {str(p): PortManager._is_available(p) for p in ports}
    return dict(config=str(config_path), maps=checks, dataset=asset,
                dataset_exists=bool(asset and Path(asset).exists()), free_ports=free,
                valid=all(x['template_exists'] for x in checks) and bool(asset and Path(asset).exists()) and all(free.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', required=True)
    parser.add_argument('--gpu', type=int, required=True, help='Physical nvidia-smi index, not CUDA ordinal')
    args = parser.parse_args()
    report = dict(sys_executable=sys.executable, python=sys.version, platform=platform.platform(),
                  cwd=str(Path.cwd()), disk_free_bytes=shutil.disk_usage('.').free,
                  dependencies={}, configs=[], errors=[])
    if platform.system() != 'Linux':
        report['errors'].append('This deployment preflight requires Linux')
    for name in ['numpy', 'PIL', 'cv2', 'openai', 'airsim', 'msgpackrpc', 'unrealcv', 'pytest']:
        try:
            mod = importlib.import_module(name)
            report['dependencies'][name] = str(getattr(mod, '__version__', 'imported'))
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    try:
        import torch
        report['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda, available=torch.cuda.is_available())
    except ImportError:
        # UE uses Vulkan; absence of optional torch must not be mislabeled as
        # successful CUDA perception, or force installation of a CPU fallback.
        report['torch'] = dict(version=None, cuda=None, available=False, status='not installed; no Python GPU inference verified')
    try:
        command = ['nvidia-smi', f'--id={args.gpu}',
                   '--query-gpu=index,uuid,pci.bus_id,name,memory.used,memory.total', '--format=csv,noheader,nounits']
        report['physical_gpu'] = subprocess.check_output(command, text=True, timeout=15).strip()
        report['compute_processes'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report['errors'].append(f'GPU query: {exc}')
    for path in args.config:
        try:
            checked = inspect_config(Path(path))
            report['configs'].append(checked)
            if not checked['valid']:
                report['errors'].append(f'Unavailable path or port: {path}')
        except Exception as exc:
            report['errors'].append(f'{path}: {type(exc).__name__}: {exc}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
