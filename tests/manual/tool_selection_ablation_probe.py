"""Single-response prompt ablation of recorded requests; NEVER executes tools.

Example (repeat --case-dir for each finalized case):
  python tests/manual/tool_selection_ablation_probe.py --case-dir <case> \
      --output-dir <new-directory> [--validate-only] [--env-path .env]

All selections and media are validated before the first paid request. Captured
messages/tools are authoritative: this diagnostic neither reconstructs future
states nor invokes a Planner, Runtime, simulator, detector, or knowledge loader.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.llm import DefaultModelClient, ModelRequest
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import public_response_snapshot
from src.agent.config import load_model_config
from src.tools.validation import validate_tool_args

PREFIXES = ('FlyAgent identity and operating rules:\n', 'User preferences:\n')
GROUPS = {'A': (), 'B': (0,), 'C': (1,), 'D': (0, 1)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    """Canonical content hash, not a token estimate."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value


def restore_request(snapshot, media, run_dir):
    """Fail closed on missing, duplicated, mislocated or tampered media.

    Restore before ablation because removing system messages changes indices.
    Read assets only within the recorded run directory, including symlink checks.
    """
    payload = copy.deepcopy(snapshot)
    root = Path(run_dir).resolve(strict=True)
    expected = {path for path, value in leaves(payload)
                if isinstance(value, str) and value.startswith('[provider_media ')}
    seen = set()
    for item in media:
        path = tuple(item['request_path'])
        if path in seen or path not in expected:
            raise ValueError('Duplicate or unmatched media request_path')
        seen.add(path)
        asset = (root / item['asset_ref']).resolve(strict=True)
        if not asset.is_relative_to(root) or not asset.is_file():
            raise ValueError('Media asset is outside the recorded run directory')
        raw = asset.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Media SHA256 mismatch')
        if item['mime_type'] not in ('image/png', 'image/jpeg', 'image/jpg', 'image/webp'):
            raise ValueError('Unsupported captured image MIME type')
        parent = payload
        for key in path[:-1]:
            parent = parent[key]
        if parent[path[-1]] != f"[provider_media sha256={item['sha256']}]":
            raise ValueError('Media placeholder and digest do not match')
        parent[path[-1]] = f"data:{item['mime_type']};base64," + base64.b64encode(raw).decode('ascii')
    if seen != expected or not seen:
        raise ValueError('Missing captured request media')
    # Do not permit a missing/unrecorded image to silently fall back to an URL.
    image_paths = {path for path, _ in leaves(payload) if path[-2:] == ('image_url', 'url')}
    if image_paths != seen:
        raise ValueError('Request contains uncaptured image URLs')
    return payload


def variant(payload, group):
    """Remove exactly the selected whole system messages; no rewritten content."""
    found = []
    for prefix in PREFIXES:
        indices = [i for i, m in enumerate(payload['messages'])
                   if m['role'] == 'system' and isinstance(m.get('content'), str)
                   and m['content'].startswith(prefix)]
        if len(indices) != 1:
            raise ValueError('Expected exactly one identity and one preference system message')
        found.append(indices[0])
    drop = {found[i] for i in GROUPS[group]}
    result = copy.deepcopy(payload)
    result['messages'] = [m for i, m in enumerate(result['messages']) if i not in drop]
    if not any(m['role'] == 'system' and isinstance(m.get('content'), str)
               and m['content'].startswith("You are FlyAgent's mission tool-loop agent.")
               for m in result['messages']):
        raise ValueError('Missing captured tool-loop system prompt')
    return result


def select_samples(case_dir):
    """Select first, first post-grounding and pre-blocked actual requests."""
    case_dir = Path(case_dir).resolve(strict=True)
    path = case_dir / 'audit.json'
    audit = read_json(path)
    rounds = [r for d in audit['planner_debug'] for r in d['model_rounds']]
    if not rounds:
        raise ValueError('No captured model requests')
    successful = next(e['payload']['tool_call_id'] for e in audit['agent_transcript']
                      if e['kind'] == 'tool_result'
                      and e['payload']['tool_name'] == 'skill.visual_grounding.locate'
                      and e['payload']['status'] == 'ok')
    after = next(i for i, r in enumerate(rounds)
                 if any(m['role'] == 'tool' and m.get('tool_call_id') == successful
                        for m in r['request']['messages']))
    blocked = [i for i, r in enumerate(rounds)
               if any(c['tool_name'] == 'runtime.report_blocked'
                      for c in r.get('canonical_turn', {}).get('tool_calls', []))]
    if len(blocked) != 1 or len({0, after, blocked[0]}) != 3:
        raise ValueError('Three distinct diagnostic snapshots cannot be identified')
    samples = []
    for kind, index in (('initial', 0), ('post_grounding', after), ('pre_blocked', blocked[0])):
        r = rounds[index]
        request = r['request']
        if request.get('tool_choice') != 'auto' or request.get('temperature') != 0:
            raise ValueError('Captured request does not use auto / temperature=0')
        restored = restore_request(request, r['request_media'], audit['run_dir'])
        samples.append({'case_id':case_dir.name, 'kind':kind, 'round_index':index,
                        'case_dir':str(case_dir), 'run_dir':str(Path(audit['run_dir']).resolve()),
                        'audit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'captured_request_sha256':digest(request),
                        'media':copy.deepcopy(r['request_media']), 'payload':restored})
    return samples


def schedule(sample_count, repeats):
    """Deterministic rotating treatment order within every matched repetition."""
    for index in range(sample_count):
        for repeat in range(repeats):
            order = list(GROUPS)
            offset = (index + repeat) % len(order)
            for group in order[offset:] + order[:offset]:
                yield index, repeat + 1, group


def wire_check(response, request):
    """Read-only response diagnostics, NOT Runtime/Safety or semantic approval."""
    schemas = {t['function']['name']:t['function']['parameters'] for t in request['tools']}
    calls = response['message'].get('tool_calls')
    issues, names, ids = [], [], set()
    if not isinstance(calls, list) or not calls:
        return {'issues':['No tool calls'], 'tool_names':[], 'tools_executed':False}
    prior = {c['id'] for m in request['messages'] for c in m.get('tool_calls', [])}
    for call in calls:
        try:
            call_id = call['id']
            if not call_id or call_id in ids or call_id in prior:
                raise ValueError('Missing or reused tool call ID')
            ids.add(call_id)
            if call.get('type') != 'function':
                raise ValueError('Unexpected tool call type')
            function = call['function']
            name = function['name']
            names.append(name)
            if name not in schemas:
                raise ValueError('Tool not exposed in captured request')
            args = json.loads(function['arguments'])
            validate_tool_args(args, schemas[name])
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f'{type(exc).__name__}: {exc}')
    return {'issues':issues, 'tool_names':names, 'tools_executed':False,
            'scope':'wire IDs/names/argument schema only; Runtime/Safety not executed'}


def render_review(output, manifest, results, status):
    """Human comparison of public responses, with no automated semantic grades."""
    esc = html.escape
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Prompt ablation</title>',
             '<style>body{font:16px/1.5 system-ui;max-width:1400px;margin:24px auto}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ddd;padding:10px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{margin:32px 0}summary{cursor:pointer}</style>',
             '<h1>同模型常驻提示消融</h1><p>Single next response only. NO tools executed. NOT a flight or success evaluation.</p>',
             f'<p>Status: {esc(status)} · Responses: {len(results)}/{manifest["planned_requests"]}</p>',
             '<p>A: all; B: no FLYAGENT; C: no USER; D: neither. Original history remains fixed, including previous mistakes.</p>',
             '<p><a href="manifest.json">Input manifest</a> · <a href="results.json">Public results</a></p>']
    for index, sample in enumerate(manifest['samples']):
        parts.append(f'<section><h2>{esc(sample["case_id"])} / {esc(sample["kind"])} (round {sample["round_index"]+1})</h2>')
        parts.append('<table><tr><th>Repetition</th>' + ''.join(f'<th>{g}</th>' for g in GROUPS) + '</tr>')
        for repeat in range(1, manifest['repeats']+1):
            parts.append(f'<tr><td>{repeat}</td>')
            for group in GROUPS:
                item = next((r for r in results if (r['sample_index'],r['repeat'],r['group'])==(index,repeat,group)), None)
                if item is None:
                    parts.append('<td>Not run</td>')
                    continue
                summary = ', '.join(item.get('wire_check', {}).get('tool_names', [])) or item['status']
                parts.append(f'<td>{esc(summary)}<details><summary>Public response / diagnostics</summary><pre>{esc(json.dumps(item,ensure_ascii=False,indent=2))}</pre></details></td>')
            parts.append('</tr>')
        parts.append('</table></section>')
    (output/'review.html').write_text(''.join(parts)+'</html>', encoding='utf-8')


def run(case_dirs, output, *, validate_only=False, repeats=3, env_path='.env', client_factory=None):
    """Validate all inputs, then independently request each treatment once.

    client_factory is a deterministic test seam. No response is ever appended to
    another sample. A provider/decode failure stops the run; malformed tool output
    is recorded without a semantic repair request or a best-of retry.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; no resume or overwrite')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    resolved = [Path(p).resolve(strict=True) for p in case_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Duplicate case directory')
    samples = [s for path in resolved for s in select_samples(path)]
    prepared, public_samples = [], []
    for sample in samples:
        requests, snapshots = {}, {}
        for group in GROUPS:
            payload = variant(sample['payload'], group)
            requests[group] = payload
            snapshot, media = snapshot_request(payload)
            if [m['sha256'] for m in media] != [m['sha256'] for m in sample['media']]:
                raise ValueError('Image order changed during ablation')
            mapped = [{**old, 'request_path':new['request_path']} for old,new in zip(sample['media'],media)]
            snapshots[group] = {'request':snapshot, 'media':mapped, 'sha256':digest(snapshot)}
        prepared.append(requests)
        public_samples.append({k:v for k,v in sample.items() if k not in ('payload','media')} | {'variants':snapshots})
    manifest = {'mode':'single_response_prompt_ablation_NO_EXECUTION', 'repeats':repeats,
                'planned_requests':len(samples)*len(GROUPS)*repeats, 'samples':public_samples,
                'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'groups':GROUPS, 'created_at':datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    results = []
    status = 'validated_only' if validate_only else 'running'

    def persist():
        write_json(output/'results.json', {'status':status, 'results':results})
        render_review(output, manifest, results, status)

    persist()
    if validate_only:
        return {'status':status, 'planned_requests':manifest['planned_requests']}
    try:
        if client_factory is None:
            settings = load_model_config(env_path).to_provider_settings()
            if not settings.api_key or any(s['payload']['model'] != settings.model_name for s in samples):
                raise ValueError('Missing credentials or model differs from captured model')
            manifest['provider'] = {'provider':settings.provider, 'model_name':settings.model_name,
                                    'timeout_sec':settings.timeout_sec,
                                    'timeout_retries':settings.provider_timeout_retries,
                                    'connection_retries':settings.provider_connection_retries}
            write_json(output/'manifest.json', manifest)
            client = DefaultModelClient(settings)
        else:
            client = client_factory()
        for index, repeat, group in schedule(len(samples), repeats):
            payload = prepared[index][group]
            request = ModelRequest(messages=payload['messages'], tools=payload['tools'],
                                   parameters={k:v for k,v in payload.items() if k not in ('model','messages','tools')})
            item = {'sample_index':index, 'repeat':repeat, 'group':group,
                    'request_sha256':public_samples[index]['variants'][group]['sha256'],
                    'started_at':datetime.now(timezone.utc).isoformat(), 'tools_executed':False}
            transport = {}
            try:
                response = client.request(request, transport_trace=transport)
                public = public_response_snapshot(response.data)
                item.update(status='response_received', response=public,
                            transport=response.transport_trace, wire_check=wire_check(public,payload))
            except Exception as exc:
                # Avoid leaking URLs, response bodies or credentials in exception text.
                item.update(status='provider_failed', error_type=type(exc).__name__,
                            category=getattr(exc,'category',None), transport=transport)
                results.append(item)
                raise
            item['finished_at'] = datetime.now(timezone.utc).isoformat()
            results.append(item)
            persist()
            print(json.dumps({'completed':len(results), 'total':manifest['planned_requests'],
                              'sample':index, 'repeat':repeat, 'group':group,
                              'tools':item['wire_check']['tool_names']}, ensure_ascii=False), flush=True)
        status = 'completed'
    except Exception as exc:
        status = 'failed'
        write_json(output/'failure.json', {'error_type':type(exc).__name__, 'completed_responses':len(results)})
    finally:
        persist()
    return {'status':status, 'responses':len(results), 'planned_requests':manifest['planned_requests']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-path', default='.env')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    report = run(args.case_dir,args.output_dir,validate_only=args.validate_only,
                 repeats=args.repeats,env_path=args.env_path)
    print(json.dumps(report), flush=True)
    return 1 if report['status']=='failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
