"""Build an audited remote deployment snapshot without changing navigation source.

This operator helper patches only the pinned controller's deployment boundaries.
It preserves the 600-task manifest and all old receipts, and blocks dispatch of
datasets not enabled in the new deployment. Activation is a separate operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Pinned controller patch does not match: ' + before[:100])
    return text.replace(before, after, 1)


def patch_controller(text):
    """Narrow deployment-only adaptations to the previously reviewed snapshot."""
    text = replace_once(text, 'SCRIPT = Path(__file__).resolve()', '''SCRIPT = Path(__file__).resolve()
DEPLOYMENT = json.loads((SCRIPT.parent / "deployment.json").read_text(encoding="utf-8"))
GPU_INDICES = tuple(DEPLOYMENT["gpu_indices"])
LANE_GPUS = tuple(zip(("A", "B"), GPU_INDICES))

def stage_enabled(stage):
    return not isinstance(stage, tuple) or stage[1] in DEPLOYMENT["enabled_datasets"]
''')
    text = replace_once(text, 'from evaluation.benchmarks.core.campaign.contracts import task_config, primary_config', '''from evaluation.benchmarks.core.campaign.contracts import task_config, primary_config as original_primary_config

def primary_config(manifest):
    data = dict(original_primary_config(manifest))
    data["map_config_path"] = DEPLOYMENT["catalog_path"]
    return data
''')
    text = text.replace('if index not in (0,2): raise ValueError(\'Only GPU 0 and 2 are authorized\')',
                        "if index < 0: raise ValueError('GPU index must be nonnegative')")
    text = replace_once(text, 'icds=[]', 'icds=[]\ndriver_libraries=set()')
    text = replace_once(text, "  if 'nvidia' in manifest.get('ICD',{}).get('library_path','').lower(): icds.append(str(path))", """  driver=manifest.get('ICD',{}).get('library_path','')
  if 'nvidia' in driver.lower() and driver not in driver_libraries:
   icds.append(str(path))
   driver_libraries.add(driver)""")
    text = text.replace('{0, 2}', 'set(GPU_INDICES)')
    text = text.replace('(("A", 0), ("B", 2))', 'LANE_GPUS')
    text = text.replace('0 if lane == "A" else 2', 'dict(LANE_GPUS)[lane]')
    text = text.replace('overlay["forbidden_remote_gpus"] != [1]',
                        'overlay["forbidden_remote_gpus"] != DEPLOYMENT["forbidden_remote_gpus"]')
    text = replace_once(text, 'if any(1 in item["observed_gpus"] for item in observations):',
                        'if any(set(item["observed_gpus"]) - set(GPU_INDICES) for item in observations):')
    text = replace_once(text, '    for lane, info in overlay["lanes"].items():', '''    for filename, digest in DEPLOYMENT["asset_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Deployment asset changed: " + filename)
    for lane, info in overlay["lanes"].items():''')
    text = replace_once(text, '        group_field, _ = compat.group_policy(manifest)', '''        if not stage_enabled(budget):
            atomic_write_text(output / "PAUSE", "Waiting for dataset readiness: " + str(budget))
            queue.pause("dataset_not_ready")
            return
        group_field, _ = compat.group_policy(manifest)''')
    text = replace_once(text, '                if next_budget != budget:', '''                if not stage_enabled(next_budget):
                    atomic_write_text(output / "PAUSE", "Waiting for dataset readiness: " + str(next_budget))
                    fatal = "dataset_not_ready"
                    break
                if next_budget != budget:''')
    # Capture streams must not inherit an unusable Windows console input handle.
    text = replace_once(text, 'child = subprocess.Popen(command, cwd=str(output / "source"), stdout=stdout, stderr=stderr,',
                        'child = subprocess.Popen(command, cwd=str(output / "source"), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,')
    return text


def patch_recovery(text):
    """Keep historical failed cleanup intact, but retire its old-server scope."""
    return replace_once(text, '            if item["status"] != "running" and not unfinished_cleanup:', '''            retirement = item["attempts"][-1].get("cleanup_retirement") if item["attempts"] else None
            if (item["status"] != "running" and retirement
                    and retirement.get("deployment_revision") == self.state.get("deployment_revision")):
                # The operator verified a different server and empty new lane
                # ports. Do not send old process IDs to the replacement server.
                unfinished_cleanup = False
            if item["status"] != "running" and not unfinished_cleanup:''')


def build(root):
    from evaluation.benchmarks.core.campaign import parallel as c
    old = c.read_json(root / 'parallel/execution_overlay.json')
    prepared = root / 'parallel/server_readiness_10245/remaining_prepared'
    directory = root / 'parallel/server_migration_10245'
    directory.mkdir(exist_ok=True)
    bindings = c.read_json(prepared / 'gpu_bindings.json')
    if [bindings[lane]['remote_gpu'] for lane in ('A', 'B')] != [0, 1]:
        raise ValueError('GPU 0/1 must be verified before migration')
    assets = {}
    def write_asset(path, data):
        content = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Refuse to overwrite a prepared deployment asset')
        path.write_bytes(content)
        assets[str(path)] = hashlib.sha256(content).hexdigest()
    catalog = c.read_json(prepared / 'maps_A.json')
    for package in catalog['maps']:
        previous = Path(package['airsim']['settings_template'])
        target = directory / previous.name
        write_asset(target, c.read_json(previous))
        package['airsim']['settings_template'] = str(target)
    write_asset(directory / 'catalog.json', catalog)
    lanes = {}
    for lane in ('A', 'B'):
        data = c.lane_catalog(catalog, lane, bindings[lane], {'mode': 'remote', 'gpu_indices': [0, 1]})
        path = directory / ('maps_' + lane + '.json')
        write_asset(path, data)
        lanes[lane] = {'map_config_path': str(path), 'sha256': assets[str(path)],
                       'local_port': 35001 if lane == 'A' else 35002,
                       'remote_port': 30001 if lane == 'A' else 30002, 'remote_gpu': bindings[lane]['remote_gpu']}
    deployment = {'catalog_path': str(directory / 'catalog.json'), 'gpu_indices': [0, 1],
                  'forbidden_remote_gpus': [2, 3], 'enabled_datasets': ['aerial_vln'],
                  'shared_gpu_authorized': True, 'asset_hashes': assets}
    previous = Path(old['controller_script']).parent
    data = {name: (previous / name).read_bytes() for name in old['source_hashes']}
    for name in ('parallel_campaign.py', 'campaign/parallel.py'):
        data[name] = patch_controller(data[name].decode('utf-8')).encode('utf-8')
    for name in ('parallel_recovery.py', 'campaign/recovery.py'):
        data[name] = patch_recovery(data[name].decode('utf-8')).encode('utf-8')
    data['deployment.json'] = json.dumps(deployment, sort_keys=True, indent=2).encode('utf-8')
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    snapshot = root / 'parallel/controller_source' / revision[:16]
    for name, content in data.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Immutable controller collision')
        path.write_bytes(content)
        if name.endswith('.py'):
            compile(content, str(path), 'exec')
    candidate = {**old, 'controller_script': str(snapshot / 'parallel_campaign.py'),
                 'controller_revision': revision, 'source_hashes': hashes, 'lanes': lanes,
                 'remote_gpu_bindings': bindings, 'forbidden_remote_gpus': [2, 3],
                 'deployment_revision': revision, 'recovery_revision': 'server10245_' + revision[:16],
                 'recovery_validation_required': True}
    c.write_json(directory / 'candidate_overlay.json', candidate)
    c.write_json(directory / 'build_receipt.json', {'changed_controller_files': [n for n in data if hashes[n] != old['source_hashes'].get(n)],
                 'execution_source_unchanged': candidate['execution_source'] == old['execution_source'],
                 'previous_controller': old['controller_script'], 'revision': revision})
    print(json.dumps({'candidate': str(directory / 'candidate_overlay.json'), 'revision': revision}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output.resolve())
