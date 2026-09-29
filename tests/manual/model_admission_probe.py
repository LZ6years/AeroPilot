"""Two real multimodal protocol requests; no tool or flight is executed.

The second request returns explicit not-executed results for actual model calls,
never invented successful tool evidence. Failures are durable and stop admission.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm import DefaultModelClient, ModelRequest
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import public_response_snapshot
from src.agent.config import load_model_config
from tests.manual.tool_selection_ablation_probe import restore_request, wire_check, write_json, digest


def run(source: Path, output: Path, *, client=None,
        expected_base_url='https://api.deepseek.com', expected_model='deepseek-flash'):
    config = load_model_config()
    if config.base_url != expected_base_url or config.model_name != expected_model or config.thinking_mode != 'disabled':
        raise ValueError('Configured endpoint/model differs from explicit admission target, or thinking is not disabled')
    output.mkdir(parents=True, exist_ok=False)
    record = json.loads(source.read_text(encoding='utf-8'))['model_rounds'][0]
    run_dir = next(parent for parent in source.parents if (parent / 'debug').is_dir())
    payload = restore_request(record['request'], record['request_media'], run_dir)
    source_digest = digest(record['request'])
    payload['model'] = config.model_name
    payload['thinking'] = {'type': config.thinking_mode}
    client = client or DefaultModelClient(config.to_provider_settings())
    rows = []
    status = {'status': 'running', 'source': str(source), 'source_request_sha256': source_digest,
              'config': config.redacted_summary(), 'tools_executed': False,
              'media': record['request_media'], 'source_run': str(run_dir), 'rounds': rows}
    try:
        for index in range(2):
            snapshot, media = snapshot_request(payload)
            row = {'request': snapshot, 'media_sha256': [m['sha256'] for m in media]}
            rows.append(row)
            write_json(output / 'admission.json', status)
            started = time.monotonic()
            response = client.request(ModelRequest(messages=payload['messages'], tools=payload['tools'],
                parameters={k: v for k, v in payload.items() if k not in ('model', 'messages', 'tools')}))
            public = public_response_snapshot(response.data)
            row.update(response=public, returned_model=response.data.get('model'),
                       elapsed_seconds=time.monotonic() - started, transport=response.transport_trace)
            row['wire_check'] = wire_check(public, payload)
            if row['wire_check']['issues']:
                raise ValueError(str(row['wire_check']['issues']))
            if response.data['choices'][0]['message'].get('reasoning_content'):
                raise ValueError('Unexpected reasoning content despite thinking disabled')
            if index == 0:
                message = copy.deepcopy(public['message'])
                payload['messages'].append(message)
                for call in message['tool_calls']:
                    payload['messages'].append({'role': 'tool', 'tool_call_id': call['id'],
                        'content': json.dumps({'status': 'rejected', 'message':
                            'Offline API admission probe: this call was not executed. No flight or tool side effects occurred.'})})
        status['status'] = 'passed'
    except Exception as exc:
        status.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        write_json(output / 'admission.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-base-url', default='https://api.deepseek.com')
    parser.add_argument('--expected-model', default='deepseek-flash')
    args = parser.parse_args()
    print(run(args.source.resolve(), args.output.resolve(),
              expected_base_url=args.expected_base_url, expected_model=args.expected_model)['status'])
