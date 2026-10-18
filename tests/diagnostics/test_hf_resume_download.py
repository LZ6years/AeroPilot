"""Real HTTP/process interruption test; no external requests or GPU use.

Requires the same huggingface_hub 1.33.0 used by the server downloader.
Can run directly with that interpreter, without pytest.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


class ResumeTest(unittest.TestCase):
    def test_resume_after_process_termination(self):
        try:
            version = importlib.metadata.version('huggingface_hub')
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('Requires downloader environment with huggingface_hub 1.33.0')
        if version != '1.33.0':
            self.skipTest('Requires downloader environment with huggingface_hub 1.33.0')
        payload = b'0123456789abcdef' * (2 * 1024 * 1024)  # 32 MiB
        prefix_size = 16 * 1024 * 1024
        ranges = []
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                header = self.headers.get('Range')
                ranges.append(header)
                start = int(header.split('=')[1].split('-')[0]) if header else 0
                self.send_response(206 if header else 200)
                self.send_header('Content-Length', str(len(payload) - start))
                if header:
                    self.send_header('Content-Range', f'bytes {start}-{len(payload)-1}/{len(payload)}')
                self.end_headers()
                try:
                    if len(ranges) == 1:
                        self.wfile.write(payload[:prefix_size])
                        self.wfile.flush()
                        release.wait(30)
                    else:
                        self.wfile.write(payload[start:])
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        root = Path(__file__).resolve().parents[2]
        code = ('import sys; from pathlib import Path; '
                'from tools.deployment.hf_resume_download import persistent_download; '
                'persistent_download(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],{},'
                'int(sys.argv[4]),"test.bin",False,sys.argv[5],None)')
        child = None
        try:
            with tempfile.TemporaryDirectory() as temp:
                partial, target = Path(temp) / 'test.incomplete', Path(temp) / 'test.bin'
                cmd = [sys.executable, '-c', code, str(partial), str(target),
                       f'http://127.0.0.1:{server.server_port}/test.bin', str(len(payload)),
                       hashlib.sha256(payload).hexdigest()]
                child = subprocess.Popen(cmd, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    if partial.exists() and partial.stat().st_size >= 1024 * 1024:
                        break
                    if child.poll() is not None:
                        self.fail('First downloader exited before writing a partial file')
                    time.sleep(0.1)
                child.terminate()
                child.wait(timeout=10)
                offset = partial.stat().st_size
                self.assertGreater(offset, 0)
                self.assertLess(offset, len(payload))
                release.set()
                result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'bytes={offset}-', ranges)
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), hashlib.sha256(payload).hexdigest())
                self.assertFalse(partial.exists())
                print(f'Verified new-process HTTP Range resume at {offset} bytes, SHA256 matched.')
        finally:
            release.set()
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
