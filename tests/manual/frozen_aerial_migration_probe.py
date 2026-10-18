"""Run one isolated navigation check against a campaign's unchanged frozen source.

Requires an explicit single-episode config. Never writes the campaign queue or
starts its watchdog. STOP in the probe directory cancels provider recovery.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.campaign.resolve(), args.output.resolve()
    config_path = args.config.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    overlay = json.loads((root / 'parallel/execution_overlay.json').read_text(encoding='utf-8'))
    source = Path(overlay['execution_source'])
    for name, digest in overlay['execution_source_hashes'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + name)
    sys.path.insert(0, str(source))
    os.chdir(source)
    import torch
    environment = {'sys.executable': sys.executable, 'torch.__version__': torch.__version__,
                   'torch.version.cuda': torch.version.cuda, 'torch.cuda.is_available': torch.cuda.is_available(),
                   'execution_source': str(source), 'recovery_revision': overlay['recovery_revision']}
    save('environment.json', environment)
    if Path(sys.executable).resolve() != Path('F:/conda_env/flyagent/python.exe').resolve() or not torch.cuda.is_available():
        raise RuntimeError('The required flyagent CUDA environment is unavailable')
    from evaluation.benchmarks.aerial_vln.config import AerialVlnEvalConfig
    from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
    from src.llm.recovery import RecoveryPolicy, recovery_scope
    from src.sim_server.remote.ssh import ssh_observer
    from src.agent.config import load_model_config
    config = AerialVlnEvalConfig.from_file(config_path)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if load_model_config(config.env_path).redacted_summary() != manifest['model']:
        raise ValueError('Decision model configuration changed')
    if len(config.episode_ids or ()) != 1:
        raise ValueError('Exactly one explicit episode is required')
    started = time.monotonic()
    save('status.json', {'status': 'running', 'episode_id': config.episode_ids[0], 'pid': os.getpid()})
    try:
        policy = RecoveryPolicy(max_attempts=6, deadline_sec=600,
            progress=lambda data: save('provider_recovery.json', data),
            cancelled=lambda: (output / 'STOP').exists() or time.monotonic() - started >= 7200)
        def ssh_progress(value):
            with (output / 'ssh_operations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value) + '\n')
        with recovery_scope(policy), ssh_observer(ssh_progress):
            result = AerialVlnBenchmarkRunner(config).run(eval_id='server_migration_one_case')
        save('result.json', result)
        valid = len(result['results']) == 1 and not any(
            item.get('error') or item.get('infrastructure_error') for item in result['results'])
        save('status.json', {'status': 'passed' if valid else 'technical_failure', 'elapsed_seconds': time.monotonic()-started,
                            'eval_dir': str(result['eval_dir']), 'counted_in_campaign': False})
    except BaseException as exc:
        save('status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
