"""Compare fixed recorded decisions under three guidance treatments; no tools run.

Example: python tests/manual/task_progress_guidance_probe.py --run-dir <aerial>
--run-dir <travel> --output-dir <new-dir> [--validate-only]. Each run contributes
decisions 1 and 6. Three treatments and three repetitions total 36 requests.
Recorded images/history are authoritative, never reconstructed from future state.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.manual.tool_selection_ablation_probe import restore_request, digest, wire_check
from src.llm import ModelRequest, create_model_client
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import public_response_snapshot
from src.agent.config import load_model_config
from src.tools.registry import default_registry

BASELINE = ROOT / 'tests/fixtures/task_progress_baseline.json'
PROMPTS = ROOT / 'src/agent/planning/prompts'
GROUPS = ('baseline', 'guidance', 'guidance_reminder')


def model_override(config, model):
    """Override one experiment's model without changing environment or source capture."""
    if model is None:
        return config
    if not model.strip():
        raise ValueError('Model override must not be empty')
    return replace(config, model_name=model.strip())


def schema_contract(value):
    """Compare schema structure, allowing prose updates while replaying captured prose verbatim."""
    if isinstance(value, list):
        return [schema_contract(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: ({name:schema_contract(schema) for name,schema in item.items()}
                  if key in ('properties', '$defs', 'definitions', 'patternProperties') else schema_contract(item))
            for key,item in value.items() if key != 'description'}


def treatments(payload, decision_index):
    """Change only the declared system guidance/tool prose and transient reminder."""
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    system = (PROMPTS / 'agent_tool_loop_system.txt').read_text(encoding='utf-8').strip()
    reminder = (PROMPTS / 'task_progress_reminder.txt').read_text(encoding='utf-8').strip()
    current = {t['function']['name']: t for t in
               [s.to_provider_tool() for s in default_registry().model_specs()]}
    old = {t['function']['name']: t for t in baseline['tools']}
    result = {}
    for group in GROUPS:
        value = copy.deepcopy(payload)
        slots = [m for m in value['messages'] if m['role'] == 'system'
                 and isinstance(m['content'], str)
                 and m['content'].startswith("You are FlyAgent's mission tool-loop agent.")]
        if len(slots) != 1 or slots[0]['content'] != baseline['system_prompt']:
            raise ValueError('Expected exactly one captured baseline task prompt')
        slots[0]['content'] = baseline['system_prompt'] if group == 'baseline' else system
        for tool in value['tools']:
            name = tool['function']['name']
            if tool['function']['description'] != old[name]['function']['description']:
                raise ValueError('Captured tool description differs from saved baseline')
            # Captured runtime safety bounds can differ from default_registry's
            # standalone defaults. Preserve the entire captured schema verbatim.
            if schema_contract(old[name]['function']['parameters']) != schema_contract(current[name]['function']['parameters']):
                raise ValueError('Current tool schema changed since baseline capture')
            if group != 'baseline':
                tool['function']['description'] = current[name]['function']['description']
        if group == 'guidance_reminder' and decision_index > 1 and (decision_index - 1) % 5 == 0:
            value['messages'].append({'role': 'system', 'content': reminder})
        result[group] = value
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def render(output, manifest, results, status):
    """Self-contained public response comparison, without hidden reasoning."""
    parts = ['<!doctype html><meta charset="utf-8"><title>Task guidance comparison</title>',
             '<style>body{font:16px system-ui;max-width:1200px;margin:auto}pre{white-space:pre-wrap}</style>',
             '<h1>Task guidance comparison</h1><p>No tools executed; single-response comparisons.</p>',
             f'<p>{html.escape(status)}: {len(results)}/{manifest["planned_requests"]}</p>']
    for row in results:
        parts.append('<details><summary>' + html.escape(f'{row["sample"]} / {row["group"]} / {row["repeat"]}: '
                     + ', '.join(row.get('wire_check', {}).get('tool_names', []))) + '</summary><pre>'
                     + html.escape(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>')
    (output / 'review.html').write_text('\n'.join(parts), encoding='utf-8')


def run(run_dirs, output, *, validate_only=False, client=None, model=None):
    """Validate every capture before paid requests; never retry to improve a score."""
    if output.exists():
        raise ValueError('Output directory must not exist')
    if len(run_dirs) != 2 or len({p.resolve() for p in run_dirs}) != 2:
        raise ValueError('Supply exactly two distinct captured runs')
    configured = load_model_config()
    settings = model_override(configured, model).to_provider_settings()
    samples = []
    for run_dir in run_dirs:
        records = {}
        for path in run_dir.glob('debug/tool_loop/*/decision_*.json'):
            decision = json.loads(path.read_text(encoding='utf-8'))
            records[decision['decision_index']] = (decision['model_rounds'][0], path)
        for index in (1, 6):
            record, path = records[index]
            payload = restore_request(record['request'], record['request_media'], run_dir)
            variants = treatments(payload, index)
            if model is None and payload['model'] != settings.model_name:
                raise ValueError('Configured model differs from recorded model')
            for value in variants.values():
                value['model'] = settings.model_name
            snapshots = {}
            for group, value in variants.items():
                snapshot, media = snapshot_request(value)
                if [m['sha256'] for m in media] != [m['sha256'] for m in record['request_media']]:
                    raise ValueError('Treatment changed captured images')
                snapshots[group] = {'request': snapshot, 'sha256': digest(snapshot)}
            samples.append({'name': f'{run_dir.name}/decision_{index}', 'source': str(path.resolve()),
                            'source_model': payload['model'],
                            'run_dir': str(run_dir.resolve()), 'media': record['request_media'],
                            'snapshots': snapshots, 'variants': variants})
    output.mkdir(parents=True)
    manifest = {'mode': 'single_response_no_execution', 'planned_requests': 36,
                'baseline_sha256': digest(json.loads(BASELINE.read_text(encoding='utf-8'))),
                'model': settings.model_name, 'configured_model': configured.model_name,
                'thinking_policy': 'preserve captured parameters; unspecified settings remain provider defaults',
                'samples': [{k: v for k, v in s.items() if k != 'variants'} for s in samples]}
    write_json(output / 'manifest.json', manifest)
    results, status = [], 'validated_only' if validate_only else 'running'
    try:
        if not validate_only:
            client = client or create_model_client(settings)
            for index, sample in enumerate(samples):
                for repeat in range(3):
                    offset = (index + repeat) % 3
                    for group in GROUPS[offset:] + GROUPS[:offset]:
                        value = sample['variants'][group]
                        req = ModelRequest(messages=value['messages'], tools=value['tools'],
                                           parameters={k:v for k,v in value.items() if k not in ('model','messages','tools')})
                        started = time.monotonic()
                        response = client.request(req)
                        public = public_response_snapshot(response.data)
                        row = {'sample': sample['name'], 'group': group, 'repeat': repeat + 1,
                               'source_model': sample['source_model'], 'requested_model': settings.model_name,
                               'returned_model': response.data.get('model'),
                               'elapsed_seconds': time.monotonic() - started,
                               'response': public, 'transport': response.transport_trace,
                               'wire_check': wire_check(public, value), 'tools_executed': False}
                        results.append(row)
                        write_json(output / 'results.json', {'status': status, 'results': results})
                        render(output, manifest, results, status)
                        # Preserve the first incompatible response before stopping;
                        # never weaken image/tool input to make a provider work.
                        if model is not None and row['wire_check']['issues']:
                            raise ValueError('Native tool protocol validation failed; see saved response')
                        print(json.dumps({'completed': len(results), 'group': group,
                                          'tools': row['wire_check']['tool_names']}), flush=True)
            status = 'completed'
    except Exception as exc:
        status = 'failed'
        write_json(output / 'failure.json', {'error_type': type(exc).__name__, 'completed': len(results),
                   'category': getattr(exc, 'category', None), 'status_code': getattr(exc, 'status_code', None),
                   'requested_model': settings.model_name})
        raise
    finally:
        write_json(output / 'results.json', {'status': status, 'results': results})
        render(output, manifest, results, status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--model', help='Experiment-only model override; keeps captured inputs unchanged')
    args = parser.parse_args()
    run(args.run_dir, args.output_dir, validate_only=args.validate_only, model=args.model)
