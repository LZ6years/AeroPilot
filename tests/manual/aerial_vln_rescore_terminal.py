"""Rescore retained AerialVLN outcomes without modifying active campaigns.

Agent-reported blocking is a valid navigation stop, not an infrastructure error.
The independent report retains original outcomes and links to original replays.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.benchmarks.core.campaign.reporting import write_report
from evaluation.benchmarks.core.report_io import write_json


def rescore(result):
    result = copy.deepcopy(result)
    distance = result.get('metrics', {}).get('distance_to_goal')
    valid = (not result.get('error') and not result.get('infrastructure_error')
             and result.get('status') in {'ok', 'stopped'}
             and ((result.get('completed') and not result.get('stopped'))
                  or result.get('stop_source') == 'runtime.report_blocked'))
    if valid and isinstance(distance, (int, float)) and math.isfinite(distance):
        metrics = result['metrics']
        metrics.update(success=distance <= 20, success_25m=distance < 25,
                       marginal_20_25m=20 < distance < 25)
        metrics['sdtw'] = float(metrics['success']) * metrics['ndtw']
    return result


def refresh(source, destination):
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    state = json.loads((source / 'progress.json').read_text(encoding='utf-8'))
    changes = []
    for key, item in state['tasks'].items():
        if item['status'] != 'scored':
            continue
        old = item['result']
        item['result'] = rescore(old)
        if item['result']['metrics'] != old['metrics']:
            changes.append({'task': key, 'stop_source': old.get('stop_source'),
                            'distance': old['metrics']['distance_to_goal'],
                            'old_success': old['metrics']['success'],
                            'new_success': item['result']['metrics']['success'],
                            'new_success_25m': item['result']['metrics']['success_25m']})
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'rescore_policy.json', {
        'policy': 'agent_blocked_is_valid_terminal_distance_v1',
        'source': str(source), 'updated_at': time.time(), 'changes': changes,
        'replay_note': 'Original replay metrics retain the frozen scoring; this report contains rescored metrics.'})
    return write_report(destination, manifest, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        records = [refresh(source, args.output / source.name) for source in args.source]
        print(json.dumps([{'groups': r['groups']} for r in records]), flush=True)
        if not args.watch or all(r['valid_executions'] == r['planned_executions'] for r in records):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
