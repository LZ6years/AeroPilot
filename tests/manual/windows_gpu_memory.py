"""Optional read-only WDDM observer in a separate hidden PowerShell process.

Counters are per adapter instance; never sum dedicated/shared/committed into a
single memory number. Shared usage alone does not prove paging. Raw status and
errors survive; unsupported values remain unknown.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


class WindowsGpuMemoryMonitor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.process = None

    def __enter__(self):
        shell = shutil.which('powershell')
        if os.name != 'nt' or not shell:
            raise RuntimeError('Windows GPU counter monitor requires PowerShell on Windows')
        self.error = (self.directory / 'windows_gpu_memory.stderr.log').open('w', encoding='utf-8')
        try:
            self.process = subprocess.Popen([shell, '-NoProfile', '-NonInteractive', '-File',
                str(Path(__file__).with_suffix('.ps1')), '-OutputPath', str(self.directory / 'windows_gpu_memory.jsonl')],
                stdout=subprocess.DEVNULL, stderr=self.error, creationflags=subprocess.CREATE_NO_WINDOW)
        except BaseException:
            self.error.close()
            raise
        self.clock_offset = time.time() - time.perf_counter()
        return self

    def __exit__(self, *args):
        stopped = self.process.poll() is None
        try:
            if stopped:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        finally:
            self.error.close()
            (self.directory / 'windows_gpu_monitor.json').write_text(json.dumps({
                'pid': self.process.pid, 'exit_code': self.process.returncode,
                'stopped_by_parent': stopped, 'unix_minus_perf_counter_sec': self.clock_offset}), encoding='utf-8')


def summarize_process_memory(directory, pid):
    path = Path(directory) / 'windows_gpu_memory.jsonl'
    result = {'pid': pid, 'samples': 0, 'invalid_counters': 0, 'invalid_rows': 0, 'instances': {}}
    if not path.exists():
        return result
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            result['invalid_rows'] += 1
            continue
        selected = [c for c in row['counters'] if re.match(rf'^pid_{pid}(?:_|$)', c['instance'])]
        if selected:
            result['samples'] += 1
        for c in selected:
            if c['status'] != 0 or c['bytes'] is None:
                result['invalid_counters'] += 1
                continue
            metric = c['path'].rsplit('\\', 1)[-1].lower()
            group = result['instances'].setdefault(c['instance'], {})
            values = group.setdefault(metric, {'min_bytes': c['bytes'], 'max_bytes': c['bytes']})
            values['min_bytes'] = min(values['min_bytes'], c['bytes'])
            values['max_bytes'] = max(values['max_bytes'], c['bytes'])
    return result
