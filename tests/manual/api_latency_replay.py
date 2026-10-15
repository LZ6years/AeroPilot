"""Export saved provider requests and replay identical bytes with curl.

Standalone diagnostic: standard-library Python and curl only. No flight tools,
SDK retries, credentials in artifacts, or production imports. Each request uses
a new curl process (cold connection); compare locations using this SAME runner.
TTFB includes upload and server work, not pure inference. Recent curl versions
also expose time_posttransfer (last upload byte); older versions leave it null.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore_request(record, root):
    """Restore image placeholders only after verifying recorded SHA-256 hashes."""
    payload = copy.deepcopy(record['request'])
    for item in record.get('request_media', []):
        asset = (root / item['asset_ref']).resolve()
        if not asset.is_relative_to(root.resolve()):
            raise ValueError('Media path escapes run directory')
        content = asset.read_bytes()
        if digest(content) != item['sha256']:
            raise ValueError('Media hash mismatch')
        target = payload
        path = item['request_path']
        for part in path[:-1]:
            target = target[part]
        expected = '[provider_media sha256=' + item['sha256'] + ']'
        if target[path[-1]] != expected:
            raise ValueError('Unexpected media placeholder')
        target[path[-1]] = 'data:' + item['mime_type'] + ';base64,' + base64.b64encode(content).decode('ascii')
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if b'[provider_media ' in encoded or b'[image_data_url]' in encoded:
        raise ValueError('Unresolved request media')
    if any(k.lower() in {'api_key', 'authorization', 'access_token'} for k in payload):
        raise ValueError('Credential field in request')
    return encoded


def export_bundle(args):
    """Sample evenly across valid saved decisions; never reconstruct missing prompts."""
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'requests').mkdir()
    manifest = []
    for run_arg in args.run:
        root = Path(run_arg).resolve()
        candidates = []
        for path in sorted(root.glob('debug/tool_loop/*/decision_*.json')):
            doc = json.loads(path.read_text(encoding='utf-8'))
            for index, record in enumerate(doc.get('model_rounds', [])):
                if record.get('request') and not record.get('error_type') and record.get('transport', {}).get('final_status') == 'success':
                    candidates.append((path, index, record))
        if not candidates:
            raise ValueError('No complete successful request snapshots: ' + str(root))
        count = min(args.per_run, len(candidates))
        indices = sorted({round(i * (len(candidates)-1) / max(1, count-1)) for i in range(count)})
        for i in indices:
            path, index, record = candidates[i]
            data = restore_request(record, root)
            name = f'requests/request_{len(manifest)+1:03d}.json'
            (out / name).write_bytes(data)
            manifest.append(dict(file=name, sha256=digest(data), bytes=len(data),
                                 images=len(record.get('request_media', [])),
                                 model=record['request'].get('model'),
                                 source_run=root.name, source_decision=path.name, source_round=index))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    shutil.copyfile(__file__, out / 'replay.py')
    print(json.dumps({'bundle': str(out), 'requests': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)}))


def response_info(body):
    """Check Chat Completions completion, including errors returned as HTTP 200."""
    text = body.decode('utf-8', errors='replace')
    usage = None
    terminal = False
    error = False
    choices_seen = False
    messages = []
    if text.lstrip().startswith('{'):
        try:
            messages = [json.loads(text)]
        except ValueError:
            return {'valid': False, 'usage': None}
    else:
        for line in text.splitlines():
            if line.startswith('data:'):
                value = line[5:].strip()
                if value == '[DONE]':
                    continue  # DONE alone is insufficient evidence of a usable turn.
                try:
                    messages.append(json.loads(value))
                except ValueError:
                    error = True
    for msg in messages:
        if not isinstance(msg, dict):
            error = True
            continue
        error = error or bool(msg.get('error'))
        usage = msg.get('usage') or usage
        for choice in msg.get('choices', []):
            choices_seen = True
            terminal = terminal or choice.get('finish_reason') is not None
    return {'valid': bool(choices_seen and terminal and not error), 'usage': usage}


def run_bundle(args):
    root = Path(args.bundle).resolve()
    entries = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for entry in entries:
        if digest((root / entry['file']).read_bytes()) != entry['sha256']:
            raise ValueError('Request hash mismatch: ' + entry['file'])
    if args.check:
        print(f'OK: {len(entries)} request files verified; no API calls made.')
        return
    if args.limit:
        entries = entries[:args.limit]
    url = os.environ.get('API_URL') or input('Full API URL (including /v1/chat/completions): ').strip()
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query or not parsed.path.endswith('/chat/completions'):
        raise ValueError('Use an HTTPS Chat Completions URL without credentials/query parameters')
    import getpass
    key = os.environ.get('API_KEY') or getpass.getpass('API key (hidden): ')
    if not key or any(c in key for c in '\r\n'):
        raise ValueError('Invalid API key')
    curl = shutil.which('curl.exe' if os.name == 'nt' else 'curl')
    if not curl:
        raise RuntimeError('curl is required')
    version = subprocess.check_output([curl, '--version'], text=True).splitlines()[0]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root / 'results' / (args.label + '_' + stamp)
    out.mkdir(parents=True)
    rows = []
    # Key goes through stdin curl configuration, not argv, disk or output.
    escaped_key = key.replace('\\', '\\\\').replace('"', '\\"')
    config = f'header = "Authorization: Bearer {escaped_key}"\n'
    for repeat in range(args.repeat):
        for entry in entries:
            with tempfile.TemporaryDirectory() as temp:
                body = Path(temp) / 'response.bin'
                command = [curl, '--disable', '--config', '-', '--silent', '--show-error',
                           '--connect-timeout', '30', '--max-time', str(args.timeout),
                           '--request', 'POST', '--header', 'Content-Type: application/json',
                           '--header', 'Expect:', '--data-binary', '@' + str(root / entry['file']),
                           '--output', str(body), '--write-out', '%{json}', url]
                result = subprocess.run(command, input=config, text=True, capture_output=True)
                try:
                    timings = json.loads(result.stdout)
                except ValueError:
                    timings = {}
                info = response_info(body.read_bytes()) if body.exists() else {'valid': False, 'usage': None}
            status = timings.get('http_code', 0)
            row = dict(request=entry['file'], sha256=entry['sha256'], repeat=repeat+1,
                       utc=datetime.now(timezone.utc).isoformat(), curl_exit=result.returncode,
                       http_status=status, valid=result.returncode == 0 and 200 <= status < 300 and info['valid'],
                       usage=info['usage'], request_bytes=entry['bytes'])
            for field in ['time_namelookup', 'time_connect', 'time_appconnect', 'time_pretransfer',
                          'time_posttransfer', 'time_starttransfer', 'time_total', 'size_upload',
                          'size_download', 'remote_ip', 'http_version']:
                row[field] = timings.get(field)
            rows.append(row)
            with (out / 'requests.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            print(f"{entry['file']} repeat={repeat+1} HTTP={status} valid={row['valid']} total={row['time_total']}s", flush=True)
            if not row['valid']:
                print('Stopped after invalid response. Check endpoint/key/model; no automatic retry.')
                break
        if rows and not rows[-1]['valid']:
            break
    valid = [r for r in rows if r['valid']]
    summary = dict(label=args.label, curl=version, endpoint=url, cold_connection=True,
                   proxy_environment_present=any(os.environ.get(k) for k in ['HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy']),
                   attempted=len(rows), successful=len(valid),
                   mean_total_seconds=statistics.mean(r['time_total'] for r in valid) if valid else None,
                   median_total_seconds=statistics.median(r['time_total'] for r in valid) if valid else None,
                   note='TTFB includes upload, network and server work. No pure inference measurement. Usage absent if provider did not return it.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Results:', out)
    if not valid or len(valid) != len(rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run', action='append', required=True)
    export.add_argument('--per-run', type=int, default=6)
    export.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--bundle', default=str(Path(__file__).resolve().parent))
    run.add_argument('--label', choices=['local', 'aws'], required=True)
    run.add_argument('--limit', type=int, default=0)
    run.add_argument('--repeat', type=int, default=1)
    run.add_argument('--timeout', type=int, default=300)
    run.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.mode == 'export':
        if args.per_run < 1:
            parser.error('--per-run must be positive')
        export_bundle(args)
    else:
        if args.limit < 0 or args.repeat < 1 or args.timeout < 1:
            parser.error('Invalid limit/repeat/timeout')
        run_bundle(args)


if __name__ == '__main__':
    main()
