"""Supervise a fixed-case benchmark child and verify only its owned scene cleanup."""

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psutil
from tests.manual.process_diagnostics import supervise, write_json
from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.control import RemoteSceneControl


def cleanup_owned(run_root, map_config, descendants, control=None, process_factory=psutil.Process):
    """Require both isolated run ownership and remote registry identity before stopping."""
    control = control or RemoteSceneControl()
    packages, _ = load_map_config(map_config)
    packages = {p.map_id: p for p in packages}
    results = []
    for path in Path(run_root).glob('*/state/scene_instance.json'):
        scene = json.loads(path.read_text(encoding='utf-8'))
        if scene.get('process_ownership') != 'remote_managed':
            continue
        package = packages[scene['map_id']]
        scene_id = scene['scene_instance_id']
        status = control.status(package, scene_id)
        if status.get('pid') != scene.get('remote_process_id') or status.get('scene_instance_id') != scene_id:
            raise ValueError('Remote ownership mismatch; refusing cleanup')
        stopped = control.stop(package, scene_id)
        settings = json.loads((path.parents[1]/'scenes'/scene_id/'settings.json').read_text(encoding='utf-8'))
        probe = control.probe_prepared_port(package, scene_id, '127.0.0.1', settings['ApiServerPort'])
        tunnel_results = []
        for child in descendants:
            command = child['command']
            if not command or Path(command[0]).stem.lower() != 'ssh' or '-L' not in command:
                continue
            if not any(arg.startswith(f"127.0.0.1:{scene['control_port']}:") for arg in command):
                continue
            try:
                process = process_factory(child['pid'])
                if process.create_time() != child['created'] or process.cmdline() != command:
                    raise ValueError('Tunnel process identity changed; refusing cleanup')
                process.terminate()
                process.wait(timeout=10)
                tunnel_results.append({'pid':child['pid'], 'status':'terminated'})
            except psutil.NoSuchProcess:
                tunnel_results.append({'pid':child['pid'], 'status':'already_exited'})
        results.append({'scene':scene_id, 'remote_stop':stopped, 'endpoint':probe, 'tunnels':tunnel_results})
    return results


def run(benchmark, directory):
    directory = Path(directory).resolve()
    runs = directory/'runs'
    result = supervise([sys.executable, '-u', '-X', 'faulthandler',
                        str(ROOT/'tests/manual/task_progress_simulation_probe.py'), '--benchmark',benchmark,
                        '--group','guided','--model','qwen3.8-flash', '--output-dir',str(directory/'evaluation_output'),
                        '--run-root',str(runs),'--trace-dir',str(directory/'trace')], directory)
    experiment = directory/'evaluation_output/experiment.json'
    try:
        if experiment.exists():
            config = json.loads(experiment.read_text(encoding='utf-8'))['config']
            result['cleanup'] = cleanup_owned(runs, config['map_config_path'], result['descendants'])
    except Exception as exc:
        result['cleanup_error'] = {'type':type(exc).__name__, 'message':str(exc)}
    write_json(directory/'supervision.json',result)
    write_review(directory, result)
    print(json.dumps(result), flush=True)


def write_review(directory, result):
    """Keep a navigable partial audit even when the benchmark cannot finalize."""
    directory = Path(directory)
    parts = ['<!doctype html><meta charset="utf-8"><title>Supervised flight</title>',
             '<h1>Supervised flight diagnostic</h1>',
             '<p>Child exit and cleanup status; an interrupted run has no final benchmark score.</p>',
             '<pre>'+html.escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre>']
    for name in ('stdout.log', 'stderr.log', 'resources.jsonl', 'supervision.json'):
        parts.append(f'<p><a href="{name}">{name}</a></p>')
    for run in (directory/'runs').glob('*'):
        for path in sorted(run.glob('debug/tool_loop/*/decision_*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            responses = [item.get('response') for item in data.get('model_rounds', [])]
            parts.append('<h2>'+html.escape(path.stem)+'</h2><pre>'+
                         html.escape(json.dumps(responses, ensure_ascii=False, indent=2))+'</pre>')
        for path in sorted(run.glob('artifacts/*/rgb/*.png')):
            relative = path.relative_to(directory).as_posix()
            parts.append(f'<p>{html.escape(path.name)}</p><img width="256" src="{html.escape(relative, quote=True)}">')
    (directory/'review.html').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('aerial','travel'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.benchmark,args.output_dir)
