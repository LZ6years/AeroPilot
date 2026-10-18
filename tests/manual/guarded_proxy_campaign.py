"""Gate an authorized campaign on disk capacity and its explicit HTTP proxy.

The child uses the existing campaign watchdog and ownership cleanup. This guard
never signals UE processes or changes frozen model/configuration files. A lost
proxy writes PAUSE; resuming after a runtime failure remains an operator action.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit


def proxy_ready(proxy, origin):
    """Check CONNECT through the selected proxy without a model/API request."""
    address, target = urlsplit(proxy), urlsplit(origin)
    if address.scheme != 'http' or not address.hostname or not target.hostname:
        raise ValueError('Expected an HTTP proxy and API origin')
    authority = f'{target.hostname}:{target.port or 443}'
    try:
        with socket.create_connection((address.hostname, address.port or 80), timeout=10) as sock:
            sock.sendall(f'CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n'.encode('ascii'))
            response = b''
            while b'\r\n' not in response and len(response) < 8192:
                data = sock.recv(1024)
                if not data:
                    break
                response += data
            return response.split(b'\r\n', 1)[0].split()[1:2] == [b'200']
    except OSError:
        return False


def proxy_environment(proxy):
    """Keep local simulator traffic direct; proxy failure cannot fall back."""
    env = dict(os.environ)
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env[name] = proxy
    for name in ('NO_PROXY', 'no_proxy'):
        env[name] = '127.0.0.1,localhost,::1'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-bytes', type=int, required=True)
    parser.add_argument('--proxy', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    origin = manifest['model']['base_url']
    child = None
    with (out / 'watchdog.log').open('a', encoding='utf-8') as log:
        while True:
            free = shutil.disk_usage(out).free
            connected = proxy_ready(args.proxy, origin)
            required = args.minimum_free_bytes if child is None else 10 * 1024**3
            reason = 'proxy_unavailable' if not connected else 'storage_budget' if free < required else None
            if child is not None and reason and not (out / 'PAUSE').exists():
                (out / 'PAUSE').write_text('Guard: ' + reason + '\n', encoding='utf-8')
            status = 'waiting_' + reason if reason else 'paused' if (out / 'PAUSE').exists() else 'monitoring'
            if child is None and reason is None and not (out / 'PAUSE').exists():
                command = [sys.executable, '-u', '-m', 'evaluation.benchmarks.core.campaign.parallel',
                           'watch', '--output', str(out), '--workers', '1']
                child = subprocess.Popen(command, cwd=out / 'source', env=proxy_environment(args.proxy),
                                         stdout=log, stderr=subprocess.STDOUT)
            record = dict(status=status, reason=reason, free_bytes=free, required_bytes=required,
                          proxy_connected=connected, child_pid=child.pid if child else None, updated_at=time.time())
            if child is not None and child.poll() is not None:
                record.update(status='exited', exit_code=child.returncode)
            temp = out / 'guard_status.tmp'
            temp.write_text(json.dumps(record, indent=2), encoding='utf-8')
            temp.replace(out / 'guard_status.json')
            if record['status'] == 'exited':
                return child.returncode
            time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
