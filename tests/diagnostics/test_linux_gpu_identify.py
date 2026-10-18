"""GPU-free checks for selection, isolation, monitoring failure and owned cleanup."""
import argparse
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.manual import linux_gpu_identify as probe


@pytest.mark.parametrize('value', ['', '0,', '-1', '0,0', '0,two'])
def test_reject_invalid_indices(value):
    with pytest.raises(argparse.ArgumentTypeError):
        probe.parse_gpus(value)


def test_uuid_selection_and_same_interpreter(monkeypatch):
    devices = {i: {'index': i, 'uuid': f'GPU-{i}', 'pci': str(i)} for i in range(4)}
    selected = probe.resolve_devices(probe.parse_gpus('0,2,3'), devices)
    assert [d['index'] for d in selected] == [0, 2, 3]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1')
    command, env = probe.worker_command(selected[-1], 123, 1.5)
    assert command[0] == sys.executable
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-3'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    with pytest.raises(ValueError, match='do not exist'):
        probe.resolve_devices([4], devices)


def snapshot(processes='<processes/>'):
    return f'''<nvidia_smi_log><gpu><uuid>GPU-test</uuid>
      <temperature><gpu_temp>82 C</gpu_temp></temperature>
      <utilization><gpu_util>100 %</gpu_util></utilization>
      <fb_memory_usage><used>1024 MiB</used></fb_memory_usage>
      <gpu_power_readings><power_draw>300 W</power_draw></gpu_power_readings>
      {processes}</gpu></nvidia_smi_log>'''


def test_telemetry_has_no_temperature_cutoff():
    result = probe.parse_snapshot(snapshot())
    assert result['GPU-test']['temperature'] == 82
    assert result['GPU-test']['power_w'] == 300
    probe.check_occupancy([{'index': 3, 'uuid': 'GPU-test'}], result)


@pytest.mark.parametrize('processes', ['', '<processes>N/A</processes>'])
def test_missing_process_monitoring_fails_closed(processes):
    with pytest.raises(RuntimeError, match='monitoring'):
        probe.parse_snapshot(snapshot(processes))


@pytest.mark.parametrize('kind,name,allowed', [('G', '/usr/lib/xorg/Xorg', True),
    ('G', '/usr/bin/Xwayland', True), ('C', '/usr/bin/python', False),
    ('C+G', 'Xorg', False), ('G', 'AirVLN-Linux-Shipping', False)])
def test_external_owners_and_desktop_exception(kind, name, allowed):
    device = {'index': 3, 'uuid': 'GPU-test'}
    state = {'GPU-test': {'owners': [{'pid': 42, 'type': kind, 'name': name}]}}
    if allowed:
        probe.check_occupancy([device], state)
    else:
        with pytest.raises(RuntimeError, match='occupied'):
            probe.check_occupancy([device], state)
    probe.check_occupancy([device], state, {42})


def test_only_owned_cpu_fixture_is_stopped():
    # Real processes, no torch imports or GPU work. Another process survives.
    command = [sys.executable, '-c', 'import time; time.sleep(60)']
    owned = subprocess.Popen(command)
    other = subprocess.Popen(command)
    try:
        probe.cleanup_workers([owned])
        assert owned.poll() is not None
        assert other.poll() is None
    finally:
        probe.cleanup_workers([owned, other])


def test_cleanup_escalates_after_timeout():
    calls = []
    def wait(timeout):
        calls.append(('wait', timeout))
        if timeout == 10:
            raise subprocess.TimeoutExpired('owned', timeout)
    child = SimpleNamespace(pid=12, poll=lambda: None, terminate=lambda: calls.append('term'),
                            kill=lambda: calls.append('kill'), wait=wait)
    probe.cleanup_workers([child])
    assert calls == ['term', ('wait', 10), 'kill', ('wait', 5)]


@pytest.mark.parametrize('failure', [RuntimeError('monitoring failed'), KeyboardInterrupt()])
def test_monitor_failure_or_interrupt_cleans_workers(monkeypatch, failure):
    monkeypatch.setattr(probe, 'inventory', lambda: {3: {'index': 3, 'uuid': 'GPU-test'}})
    count = 0
    def smi(_):
        nonlocal count
        count += 1
        if count > 1:
            raise failure
        return snapshot()
    monkeypatch.setattr(probe, 'smi', smi)
    child = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a, **kw: child)
    cleaned = []
    monkeypatch.setattr(probe, 'cleanup_workers', lambda workers: cleaned.extend(workers))
    with pytest.raises(type(failure)):
        probe.run([3])
    assert cleaned == [child]


def test_missing_cuda_never_enters_matrix_load(monkeypatch, capsys):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='test',
        version=SimpleNamespace(cuda=None), cuda=cuda))
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        probe.worker('GPU-test', os.getpid(), 0)
    assert '"cuda_available": false' in capsys.readouterr().out
