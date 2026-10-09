"""Pre-launch cleanup proof must refuse unknown ownership and live endpoints."""
import json
import subprocess
from types import SimpleNamespace

import psutil
import pytest

from src.sim_server.remote import recovery


def prepared(tmp_path, monkeypatch):
    scene = tmp_path / 'scene_unique'
    scene.mkdir()
    events = [{'stage': stage, 'status': 'ok'} for stage in
              ('start_requested', 'tunnel_closed', 'port_reservation_released')]
    events.append({'stage': 'startup_failed', 'status': 'error', 'cleanup_errors': []})
    (scene / 'lifecycle.jsonl').write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    (scene / 'settings.json').write_text(json.dumps({'ApiServerPort': 30001}), encoding='utf-8')
    processes = [{'pid': 1, 'created': 10, 'command': ['flyagent', 'child']},
                 {'pid': 2, 'created': 11, 'command': ['ssh', '-L', '127.0.0.1:35001:127.0.0.1:30001']}]
    remote = {'scene': scene.name, 'files': {'scene-registry.json': False, 'scene-spec.json': False,
                                          'settings.json': False}, 'matching_processes': [],
              'unreadable_processes': [], 'endpoint_closed': True, 'endpoint_connect_ex': 111}
    def absent(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(recovery.psutil, 'Process', absent)
    monkeypatch.setattr(recovery, '_closed_local_port', lambda *args: True)
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(remote)
    control = SimpleNamespace(runner=SimpleNamespace(run=run), target=lambda _: 'target',
                              _paths=lambda *args: SimpleNamespace(directory='.flyagent/scenes/' + scene.name))
    return scene, processes, remote, control, calls


def test_proof_requires_fresh_remote_process_and_endpoint_inspection(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    proof = recovery.verify_unlaunched_scene(None, scene, processes, control=control)
    assert proof['verified'] and proof['remote']['endpoint_closed']
    assert len(calls) == 1 and calls[0][1]['timeout_sec'] == 15
    assert all(not item['original_process_alive'] for item in proof['local_processes'])


@pytest.mark.parametrize('fault', ['spec', 'process_started', 'remote_spec', 'remote_process', 'unreadable', 'endpoint', 'tunnel', 'live_local'])
def test_ambiguous_or_live_scene_never_receives_cleanup_proof(tmp_path, monkeypatch, fault):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    if fault == 'spec':
        (scene / 'remote-scene-spec.json').write_text('{}', encoding='utf-8')
    elif fault == 'process_started':
        with (scene / 'lifecycle.jsonl').open('a', encoding='utf-8') as handle:
            handle.write('\n' + json.dumps({'stage': 'process_started', 'status': 'ok'}))
    elif fault == 'remote_spec':
        remote['files']['scene-spec.json'] = True
    elif fault == 'remote_process':
        remote['matching_processes'] = [123]
    elif fault == 'unreadable':
        remote['unreadable_processes'] = [123]
    elif fault == 'endpoint':
        remote['endpoint_closed'] = False
    elif fault == 'tunnel':
        monkeypatch.setattr(recovery, '_closed_local_port', lambda *args: False)
    elif fault == 'live_local':
        monkeypatch.setattr(recovery.psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 10,
                                                                  cmdline=lambda: ['flyagent', 'child']))
    assert not recovery.verify_unlaunched_scene(None, scene, processes, control=control)['verified']


def test_pid_reuse_is_distinguished_from_surviving_owned_process(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery.psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 999, cmdline=lambda: ['other']))
    assert recovery.verify_unlaunched_scene(None, scene, processes, control=control)['verified']


def test_ssh_probe_timeout_cannot_be_interpreted_as_cleanup(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ssh', 15)
    control.runner.run = timeout
    with pytest.raises(subprocess.TimeoutExpired):
        recovery.verify_unlaunched_scene(None, scene, processes, control=control)


def test_completed_parallel_attempt_can_prove_failure_before_tunnel_or_settings(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    events = [json.loads(line) for line in (scene / 'lifecycle.jsonl').read_text(encoding='utf-8').splitlines()]
    events = [event for event in events if event['stage'] != 'tunnel_closed']
    (scene / 'lifecycle.jsonl').write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    (scene / 'settings.json').unlink()
    package = SimpleNamespace(airsim={'port': 30002})
    proof = recovery.verify_unlaunched_scene(package, scene, [], control=control,
        completed_attempt=True, local_endpoint=('127.0.0.1', 35002))
    assert proof['verified'] and proof['local_tunnels'][0]['port'] == 35002
    assert not recovery.verify_unlaunched_scene(package, scene, [], control=control)['verified']


def test_completed_parallel_attempt_cannot_ignore_open_lane_port(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, '_closed_local_port', lambda *args: False)
    assert not recovery.verify_unlaunched_scene(None, scene, [], control=control,
        completed_attempt=True, local_endpoint=('127.0.0.1', 35002))['verified']


def failed_stop_before_launch(scene, *, error_type='TimeoutExpired', stage='owned_process_stopped'):
    """Model lost SSH rollback without inventing a launch/ownership receipt."""
    path = scene / 'lifecycle.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    events.insert(1, {'stage': 'backend_closed', 'status': 'ok'})
    events[-1]['cleanup_errors'] = [{'stage': stage, 'error_type': error_type, 'error': 'SSH unavailable'}]
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')


def test_recheck_replaces_old_stop_timeout_only_with_fresh_absence_evidence(tmp_path, monkeypatch):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    failed_stop_before_launch(scene)
    initial = (scene / 'lifecycle.jsonl').read_bytes()
    options = {'control': control, 'completed_attempt': True, 'local_endpoint': ('127.0.0.1', 35001)}
    assert not recovery.verify_unlaunched_scene(None, scene, processes, **options)['verified']
    remote['endpoint_closed'] = False
    assert not recovery.verify_unlaunched_scene(None, scene, processes, recheck_cleanup=True, **options)['verified']
    remote['endpoint_closed'] = True
    proof = recovery.verify_unlaunched_scene(None, scene, processes, recheck_cleanup=True, **options)
    assert proof['verified'] and proof['kind'] == 'startup_not_launched_rechecked'
    assert proof['cleanup_recheck_errors'][0]['error_type'] == 'TimeoutExpired'
    assert len(calls) == 2 and (scene / 'lifecycle.jsonl').read_bytes() == initial


@pytest.mark.parametrize('fault', ['unknown', 'auth', 'backend_error', 'missing_backend', 'missing_release',
                                  'not_completed', 'no_endpoint', 'spec', 'remote_spec', 'remote_process',
                                  'unreadable', 'local_process', 'local_port', 'mixed', 'malformed'])
def test_historical_cleanup_recheck_cannot_skip_missing_or_unsafe_evidence(tmp_path, monkeypatch, fault):
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    failed_stop_before_launch(scene, error_type={'unknown': 'ValueError', 'auth': 'AuthenticationError'}.get(fault, 'TimeoutExpired'),
                              stage='backend_closed' if fault == 'backend_error' else 'owned_process_stopped')
    path = scene / 'lifecycle.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if fault in {'missing_backend', 'missing_release'}:
        removed = 'backend_closed' if fault == 'missing_backend' else 'port_reservation_released'
        events = [event for event in events if event['stage'] != removed]
    elif fault == 'mixed':
        events[-1]['cleanup_errors'].append({'stage': 'owned_process_stopped', 'error_type': 'AuthenticationError'})
    elif fault == 'malformed':
        events[-1]['cleanup_errors'] = {'error_type': 'TimeoutExpired'}
    path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    if fault == 'spec': (scene / 'remote-scene-spec.json').write_text('{}', encoding='utf-8')
    if fault == 'remote_spec': remote['files']['scene-spec.json'] = True
    if fault == 'remote_process': remote['matching_processes'] = [123]
    if fault == 'unreadable': remote['unreadable_processes'] = [123]
    if fault == 'local_process': monkeypatch.setattr(recovery.psutil, 'Process', lambda _: SimpleNamespace(create_time=lambda: 10, cmdline=lambda: processes[0]['command']))
    if fault == 'local_port': monkeypatch.setattr(recovery, '_closed_local_port', lambda *args: False)
    proof = recovery.verify_unlaunched_scene(None, scene, processes, control=control,
        completed_attempt=fault != 'not_completed', local_endpoint=None if fault == 'no_endpoint' else ('127.0.0.1', 35001), recheck_cleanup=True)
    assert not proof['verified']


def test_windows_tcp_table_proves_closed_port_and_refuses_any_surviving_endpoint(monkeypatch):
    monkeypatch.setattr(recovery, 'os', SimpleNamespace(name='nt'))
    connections = []
    monkeypatch.setattr(recovery.psutil, 'net_connections', lambda **kwargs: list(connections))
    assert recovery._closed_local_port('127.0.0.1', 35001)
    connections.append(SimpleNamespace(laddr=SimpleNamespace(port=35001)))
    assert not recovery._closed_local_port('127.0.0.1', 35001)
    assert recovery._closed_local_port('127.0.0.1', 35002)
    assert not recovery._closed_local_port('example.com', 35002)


def test_windows_unreadable_tcp_table_is_not_cleanup_evidence(monkeypatch):
    monkeypatch.setattr(recovery, 'os', SimpleNamespace(name='nt'))
    def unreadable(**kwargs):
        raise psutil.AccessDenied()
    monkeypatch.setattr(recovery.psutil, 'net_connections', unreadable)
    assert not recovery._closed_local_port('127.0.0.1', 35001)


def test_connection_timeout_is_never_proof_of_closed_port(monkeypatch):
    monkeypatch.setattr(recovery, 'os', SimpleNamespace(name='posix'))
    class TimeoutSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, seconds): pass
        def connect_ex(self, endpoint): return 10035
    monkeypatch.setattr(recovery.socket, 'socket', TimeoutSocket)
    assert not recovery._closed_local_port('127.0.0.1', 35001)


def rolled_back_launch(tmp_path, monkeypatch):
    """A lost SSH start response leaves a registry but no ready SceneInstance."""
    scene, processes, remote, control, calls = prepared(tmp_path, monkeypatch)
    paths = SimpleNamespace(directory='.flyagent/scenes/' + scene.name,
                            settings='.flyagent/scenes/' + scene.name + '/settings.json',
                            registry='.flyagent/scenes/' + scene.name + '/scene-registry.json')
    control._paths = lambda *args: paths
    events = [{'stage': stage, 'status': 'ok'} for stage in (
        'start_requested', 'backend_closed', 'owned_process_stopped', 'tunnel_closed', 'port_reservation_released')]
    events.append({'stage': 'startup_failed', 'status': 'error', 'cleanup_errors': []})
    (scene / 'lifecycle.jsonl').write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    spec = {'scene_instance_id': scene.name, 'executable': '/sim/AirVLN.sh',
            'registry_path': paths.registry, 'settings_path': paths.settings}
    (scene / 'remote-scene-spec.json').write_text(json.dumps(spec), encoding='utf-8')
    registry = {'ok': True, 'scene_instance_id': scene.name, 'pid': 20, 'pgid': 20,
                'process_start_ticks': 999, 'executable': spec['executable'],
                'settings_path': '/home/user/' + paths.settings, 'status': 'stopped',
                'cleanup_verified': True, 'live_members': []}
    control.status = lambda *args: dict(registry)
    remote['files'] = {key: True for key in remote['files']}
    remote['documents'] = {'scene-spec.json': spec, 'scene-registry.json': registry}
    remote['resolved_paths'] = {'settings': registry['settings_path'], 'executable': registry['executable']}
    package = SimpleNamespace(launch={'linux_executable': spec['executable']})
    return scene, processes, remote, control, calls, registry, package


def test_lost_start_response_accepts_only_fresh_owned_group_and_endpoint_proof(tmp_path, monkeypatch):
    scene, processes, remote, control, calls, registry, package = rolled_back_launch(tmp_path, monkeypatch)
    proof = recovery.verify_failed_launch(package, scene, processes[1:], control=control,
                                         completed_attempt=True, local_endpoint=('127.0.0.1', 35001))
    assert proof['verified'] and proof['remote_status']['cleanup_verified']
    assert len(calls) == 1 and calls[0][1]['timeout_sec'] == 15
    assert not recovery.verify_unlaunched_scene(package, scene, processes, control=control)['verified']


@pytest.mark.parametrize('fault', ['rollback', 'rollback_error', 'ready', 'running', 'unknown_registry',
                                  'spec_mismatch', 'settings_mismatch', 'registry_mismatch', 'process',
                                  'unreadable', 'endpoint', 'local_port', 'local_process', 'not_completed'])
def test_launched_startup_cleanup_refuses_incomplete_or_mismatched_proof(tmp_path, monkeypatch, fault):
    scene, processes, remote, control, calls, registry, package = rolled_back_launch(tmp_path, monkeypatch)
    if fault in {'rollback', 'rollback_error', 'ready'}:
        path = scene / 'lifecycle.jsonl'
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if fault == 'rollback':
            events = [event for event in events if event['stage'] != 'owned_process_stopped']
        elif fault == 'rollback_error':
            events[-1]['cleanup_errors'] = ['stop failed']
        else:
            events.insert(1, {'stage': 'scene_ready', 'status': 'ok'})
        path.write_text('\n'.join(json.dumps(event) for event in events), encoding='utf-8')
    elif fault == 'running':
        registry['status'] = 'running'
    elif fault == 'unknown_registry':
        registry['pid'] = None
    elif fault == 'spec_mismatch':
        remote['documents']['scene-spec.json'] = {'scene_instance_id': 'other'}
    elif fault == 'settings_mismatch':
        remote['resolved_paths']['settings'] = '/other/settings.json'
    elif fault == 'registry_mismatch':
        remote['documents']['scene-registry.json'] = {**registry, 'process_start_ticks': 1000}
    elif fault == 'process':
        remote['matching_processes'] = [20]
    elif fault == 'unreadable':
        remote['unreadable_processes'] = [20]
    elif fault == 'endpoint':
        remote['endpoint_closed'] = False
    elif fault == 'local_port':
        monkeypatch.setattr(recovery, '_closed_local_port', lambda *args: False)
    elif fault == 'local_process':
        monkeypatch.setattr(recovery.psutil, 'Process', lambda _: SimpleNamespace(
            create_time=lambda: 11, cmdline=lambda: processes[1]['command']))
    assert not recovery.verify_failed_launch(package, scene, processes[1:], control=control,
        completed_attempt=fault != 'not_completed', local_endpoint=('127.0.0.1', 35001))['verified']
