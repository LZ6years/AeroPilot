"""Check real file:// image decoding and scroll screenshots in Chrome and Edge.

Uses an isolated headless browser profile and CDP; no simulator/model requests.
Uses websocket-client when installed, or Windows' native .NET WebSocket client.
Writes validation evidence inside the review bundle; never installs packages.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import time
from urllib.request import urlopen

import psutil


class NativeWebSocket:
    """Small JSON-lines adapter to the existing Windows .NET WebSocket API."""

    def __init__(self, endpoint):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        if os.name != 'nt' or not shell:
            raise RuntimeError('CDP requires websocket-client or Windows PowerShell/.NET')
        self.process = subprocess.Popen([shell, '-NoProfile', '-NonInteractive', '-File',
            str(Path(__file__).with_name('cdp_websocket.ps1')), '-Endpoint', endpoint],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            if not self._read().get('ready'):
                raise RuntimeError('Native CDP transport not ready')
        except BaseException:
            self.process.kill()
            self.process.communicate(timeout=5)
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Native CDP transport exited without a reply')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def _command(self, op, **values):
        self.process.stdin.write(json.dumps({'op': op, **values}) + '\n')
        self.process.stdin.flush()
        return self._read()

    def send(self, payload):
        self._command('send', payload=payload)

    def recv(self):
        return self._command('recv')['data']

    def close(self):
        try:
            if self.process.poll() is None:
                self._command('close')
                self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
            self.process.communicate(timeout=5)


def connect_cdp(endpoint):
    try:
        import websocket
    except ModuleNotFoundError as exc:
        if exc.name != 'websocket':
            raise
        return NativeWebSocket(endpoint)
    return websocket.create_connection(endpoint, suppress_origin=True, timeout=30)


def close_owned_browser(profile, executable):
    """Edge may relaunch its parent; ownership follows the unique test profile."""
    profile_path = Path(profile).resolve()
    if not profile_path.is_relative_to(Path(tempfile.gettempdir()).resolve()) or not profile_path.name.startswith('fly_review_browser_'):
        raise ValueError('Unexpected browser profile ownership')
    flag = '--user-data-dir=' + str(profile_path)
    owned = []
    for process in psutil.process_iter(['exe', 'cmdline']):
        try:
            if flag in (process.info['cmdline'] or []) and Path(process.info['exe'] or '').resolve() == executable.resolve():
                process.terminate()
                owned.append(process)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for process in alive:
        process.kill()
    psutil.wait_procs(alive, timeout=5)


def inspect(executable: Path, root: Path, pages: list[str], name: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix='fly_review_browser_') as profile:
        process = subprocess.Popen([str(executable), '--headless', '--disable-gpu', '--no-first-run',
            '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        connection = None
        try:
            port_file = Path(profile) / 'DevToolsActivePort'
            deadline = time.monotonic() + 20
            while not port_file.exists():
                # Edge's launcher may exit successfully after handing the same
                # profile to another process; the DevTools port is authoritative.
                if time.monotonic() > deadline:
                    raise RuntimeError('Headless browser did not start')
                time.sleep(.1)
            port = port_file.read_text().splitlines()[0]
            with urlopen(f'http://127.0.0.1:{port}/json/list') as response:
                target = next(t for t in json.load(response) if t['type'] == 'page')
            connection = connect_cdp(target['webSocketDebuggerUrl'])
            sequence = 0

            def call(method, params=None):
                nonlocal sequence
                sequence += 1
                connection.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
                while True:
                    reply = json.loads(connection.recv())
                    if reply.get('id') == sequence:
                        if 'error' in reply:
                            raise RuntimeError(reply['error'])
                        return reply.get('result', {})

            call('Page.enable')
            call('Emulation.setDeviceMetricsOverride', {'width': 1280, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
            results = []
            for relative in pages:
                uri = (root / relative).resolve().as_uri()
                call('Page.navigate', {'url': uri})
                deadline = time.monotonic() + 20
                while True:
                    state = call('Runtime.evaluate', {'expression': '({url:location.href,ready:document.readyState})', 'returnByValue': True})['result'].get('value', {})
                    if state.get('url') == uri and state.get('ready') == 'complete':
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('Review page load timeout')
                    time.sleep(.1)
                expression = """(async () => {
                    const imgs = [...document.images];
                    imgs.forEach(i => i.loading = 'eager');
                    await Promise.all(imgs.map(i => i.decode().catch(() => null)));
                    const frame = document.querySelector('.phase-timeline img');
                    if (frame) frame.scrollIntoView({block:'center'});
                    return {images: imgs.length, failed: imgs.filter(i => !i.naturalWidth).map(i => i.getAttribute('src')),
                            timelineImageWidth: frame ? frame.naturalWidth : 0, url: location.href};
                })()"""
                result = call('Runtime.evaluate', {'expression': expression, 'awaitPromise': True, 'returnByValue': True})
                if 'exceptionDetails' in result:
                    raise RuntimeError(result['exceptionDetails'])
                data = result['result']['value']
                if not data['images'] or data['failed'] or not data['timelineImageWidth']:
                    raise ValueError(f'Browser image failure: {relative}: {data}')
                time.sleep(.2)
                screenshot = call('Page.captureScreenshot', {'format': 'png'})
                filename = name + '_' + relative.replace('/', '_').replace('.html', '.png')
                (root / 'browser_validation' / filename).write_bytes(base64.b64decode(screenshot['data']))
                results.append({'page': relative, **data, 'screenshot': filename})
            return results
        finally:
            if connection:
                try:
                    call('Browser.close')
                except Exception:
                    pass
            try:
                if connection:
                    connection.close()
            finally:
                close_owned_browser(profile, executable)
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Short case numbers; default first available case per benchmark')
    args = parser.parse_args()
    root = args.review_root.resolve()
    (root / 'browser_validation').mkdir(exist_ok=True)
    numbers = args.cases
    if not numbers:
        exported = json.loads((root / 'export_summary.json').read_text(encoding='utf-8'))
        representatives = {}
        for row in exported['cases']:
            if row['attempts']:
                representatives.setdefault(row['benchmark'], row['attempts'][-1]['review'].split('/')[1])
        numbers = list(representatives.values())
    if not numbers:
        raise ValueError('No generated cases to inspect')
    pages = [f'cases/{number}/{view}.html' for number in numbers for view in ('review', 'audit_review')]
    results = {}
    for name, executable in (
        ('chrome', Path('C:/Program Files/Google/Chrome/Application/chrome.exe')),
        ('edge', Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')),
    ):
        results[name] = inspect(executable, root, pages, name)
    (root / 'browser_validation/results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: [{'page': row['page'], 'images': row['images'], 'failed': row['failed']} for row in rows]
                      for name, rows in results.items()}))
