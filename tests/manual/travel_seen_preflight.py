"""Validate prepared Linux TravelUAV assets and every frozen execution lane.

No model calls. Repairs only missing owner execute bits when explicitly asked;
asset bytes, existing processes and active evaluation snapshots stay intact.
Each capture child uses this process's flyagent interpreter and own output tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks.core.campaign.parallel import read_json, verify_frozen, verify_overlay, validate_local_capture
from evaluation.benchmarks.core.report_io import write_json


def validate_resources(catalog, *, repair_execute_bits=False):
    """Check the exact launcher-referenced ELF, not guessed shipping names."""
    records = []
    for item in catalog['maps']:
        root = Path(item['root']).resolve()
        script = (root / item['launch']['linux']).resolve()
        if root not in script.parents or not script.is_file():
            raise ValueError('Launcher is outside the configured asset package')
        content = script.read_text()
        binaries = [b for b in root.glob('*/Binaries/Linux/*') if b.is_file()
                    and b.relative_to(root).as_posix() in content
                    and b.open('rb').read(4) == b'\x7fELF']
        if len(binaries) != 1:
            raise ValueError('Expected one launcher-referenced executable: ' + str(script))
        permissions = []
        for path in [script, *binaries]:
            before = stat.S_IMODE(path.stat().st_mode)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not os.access(path, os.X_OK) and repair_execute_bits:
                if path.stat().st_uid != os.getuid():
                    raise PermissionError('Asset is not owned by the current account: ' + str(path))
                path.chmod(before | stat.S_IXUSR)
            if not os.access(path, os.X_OK):
                raise PermissionError('Missing execute permission: ' + str(path))
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Asset content changed during permission check')
            permissions.append(dict(path=str(path), mode_before=before,
                                    mode_after=stat.S_IMODE(path.stat().st_mode), sha256=digest))
        libraries = subprocess.run(['ldd', str(binaries[0])], capture_output=True, text=True, check=True)
        missing = [line.strip() for line in libraries.stdout.splitlines() if 'not found' in line]
        if missing:
            raise RuntimeError('Missing shared libraries: ' + json.dumps(missing))
        records.append(dict(scene_id=item['scene_id'], files=permissions, missing_libraries=missing))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repair-execute-bits', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = read_json(out / 'manifest.json')
    overlay = read_json(out / 'parallel/execution_overlay.json')
    verify_frozen(out, manifest)
    verify_overlay(overlay, manifest)
    config = manifest['configs'][manifest['primary_config_key']]
    catalog = read_json(config['map_config_path'])
    record = dict(passed=False, resources=validate_resources(catalog, repair_execute_bits=args.repair_execute_bits), lane_captures={})
    write_json(out / 'capture/resource_validation.json', record['resources'])
    representative = None
    for lane, info in overlay['lanes'].items():
        destination = out / 'capture' / lane
        command = [sys.executable, '-u', str(out / 'source/tests/manual/linux_scene_capture.py'),
                   '--output', str(destination), '--maps', info['map_config_path'], '--gpu', str(info['remote_gpu']),
                   '--lane', lane, '--execution-config', str(out / 'inputs/execution.json'),
                   '--travel-index', config['index_path']]
        if representative is not None:
            command += ['--scene', representative]
        result = subprocess.run(command, cwd=out / 'source')
        receipt = read_json(destination / 'capture_validation.json')
        record['lane_captures'][lane] = receipt
        if result.returncode or not receipt.get('passed'):
            write_json(out / 'capture/capture_validation.json', record)
            return 1
        # The complete first lane proves all package contents; a fast verified
        # package then checks each additional port/GPU without repeating all maps.
        representative = min(receipt['scenes'], key=lambda scene: scene['startup_seconds'])['scene_id']
    record['passed'] = True
    validate_local_capture(record, overlay, manifest['execution'])
    write_json(out / 'capture/capture_validation.json', record)
    print(json.dumps(dict(passed=True, lanes=list(record['lane_captures']), maps=len(catalog['maps']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
