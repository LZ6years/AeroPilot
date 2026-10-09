"""Diagnose the manual pilot/report guard without a real provider or flight."""
import json
from types import SimpleNamespace

import pytest

from tests.manual.aerial_vln_pilot_handoff import final_child_finished, guard, monitor_split, recover_startup, shared_simulator_busy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rows():
    return [{"task": {"split": "val_seen", "episode_id": str(index), "scene_id": "2"},
             "status": "scored" if index < 14 else "running", "attempts": [{"number": 1}],
             "result": {"metrics": {"success": False}}} for index in range(15)]


def test_handoff_arms_only_after_last_valid_child_has_exited(tmp_path):
    cases = rows()
    base = tmp_path / "val_seen/attempts/14/attempt_01"
    write(base / "result.json", {"metrics": {"success": False}})
    write(base / "process.json", {"status": "running"})
    assert not final_child_finished(tmp_path, cases)
    write(base / "process.json", {"exit_code": 0})
    assert final_child_finished(tmp_path, cases)
    write(base / "result.json", {"metrics": {"trajectory_ned": []}, "infrastructure_error": {"stage": "provider"}})
    assert not final_child_finished(tmp_path, cases)


def test_complete_pilot_produces_report_and_keeps_full_seen_pending(tmp_path):
    cases = rows()
    cases[-1]["status"] = "scored"
    pilot = [row["task"] for row in cases]
    task_state = {f'val_seen/{row["task"]["episode_id"]}': row for row in cases}
    task_state['val_seen/remaining'] = {"task": {"split": "val_seen", "episode_id": "remaining"},
                                       "status": "pending", "attempts": [], "result": None}
    write(tmp_path / "manifest.json", {"pilot": pilot})
    write(tmp_path / "progress.json", {"status": "paused", "reason": "manual_interrupt", "tasks": task_state})
    guard(tmp_path)
    assert (tmp_path / "PAUSE").is_file()
    ready = json.loads((tmp_path / "pilot_report_ready.json").read_text())
    assert ready["scored"] == 15 and not ready["full_seen_started"]
    assert (tmp_path / "pilot_report.html").is_file()
    assert not (tmp_path / "val_seen/attempts").exists()


def startup_campaign(tmp_path, monkeypatch):
    from tests.manual import aerial_vln_pilot_handoff as module
    task = {'split': 'val_unseen', 'episode_id': 'pilot', 'scene_id': '7'}
    directory = tmp_path / 'val_unseen/attempts/pilot/attempt_01'
    (directory / 'runs/run/scenes/scene_test').mkdir(parents=True)
    attempt = {'number': 1, 'status': 'running', 'cleanup_verified': False,
               'directory': str(directory), 'process': {'pid': 99, 'created': 10, 'command': ['child'], 'descendants': []},
               'result': {'infrastructure_error': {'stage': 'scene_setup', 'error': {
                   'error_type': 'TimeoutExpired', 'error': "Command '['ssh', 'mkdir']' timed out"}}}}
    row = {'task': task, 'status': 'retry_pending', 'attempts': [attempt], 'result': None}
    state = {'fingerprint': 'same-flight-source', 'status': 'paused', 'reason': 'cleanup_unverified',
             'tasks': {'val_unseen/pilot': row}}
    write(tmp_path / 'progress.json', state)
    manifest = {'pilot': [task], 'configs': {'val_unseen': {'env_path': 'env', 'map_config_path': 'maps',
                                                        'scene_map': {'7': 'map'}}}}
    monkeypatch.setattr(module, 'read_env_file', lambda _: {})
    monkeypatch.setattr(module, 'load_map_config', lambda _: ([SimpleNamespace(map_id='map')], None))
    launches = []
    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return SimpleNamespace(pid=123)
    return manifest, state, directory, launches, launch


def test_verified_startup_resume_preserves_failure_and_scores(tmp_path, monkeypatch):
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    proof = {'verified': True, 'scene': 'scene_test', 'kind': 'startup_not_launched_verified'}
    assert recover_startup(tmp_path, manifest, verify=lambda *args: proof, launch=launch)
    updated = json.loads((tmp_path / 'progress.json').read_text(encoding='utf-8'))
    row = updated['tasks']['val_unseen/pilot']
    assert updated['status'] == 'prepared' and updated['fingerprint'] == state['fingerprint']
    assert len(row['attempts']) == 1 and row['result'] is None and row['status'] == 'retry_pending'
    assert row['attempts'][0]['cleanup_verified']
    assert row['attempts'][0]['result'] == state['tasks']['val_unseen/pilot']['attempts'][0]['result']
    assert len(launches) == 1 and '--stage' in launches[0][0][0]
    assert launches[0][1]['cwd'] == str(tmp_path / 'source')
    assert (directory / 'cleanup_recovery.json').exists()


@pytest.mark.parametrize('stop', ['manual', 'unknown_cleanup', 'wrong_error'])
def test_monitor_keeps_unsafe_or_requested_pause(tmp_path, monkeypatch, stop):
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    if stop == 'manual':
        (tmp_path / 'PAUSE').write_text('manual stop', encoding='utf-8')
    elif stop == 'wrong_error':
        state['tasks']['val_unseen/pilot']['attempts'][0]['result']['infrastructure_error']['stage'] = 'provider'
        write(tmp_path / 'progress.json', state)
    assert not recover_startup(tmp_path, manifest, verify=lambda *args: {'verified': False}, launch=launch)
    assert not launches
    assert json.loads((tmp_path / 'progress.json').read_text(encoding='utf-8')) == state


def test_third_startup_attempt_never_creates_fourth_attempt(tmp_path, monkeypatch):
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    row = state['tasks']['val_unseen/pilot']
    row['attempts'] = [dict(row['attempts'][0], number=index) for index in (1, 2, 3)]
    write(tmp_path / 'progress.json', state)
    assert recover_startup(tmp_path, manifest, verify=lambda *args: {'verified': True, 'scene': 'scene_test'}, launch=launch)
    updated = json.loads((tmp_path / 'progress.json').read_text(encoding='utf-8'))['tasks']['val_unseen/pilot']
    assert updated['status'] == 'retry_exhausted' and len(updated['attempts']) == 3


def test_observer_rechecks_temporary_ssh_failure_then_stops_at_report(tmp_path, monkeypatch):
    import subprocess
    from tests.manual import aerial_vln_pilot_handoff as module
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    write(tmp_path / 'manifest.json', manifest)
    clock = [0]
    checks = []
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda duration: clock.__setitem__(0, clock[0] + duration))
    def recover(*args, **kwargs):
        checks.append(clock[0])
        if len(checks) == 1:
            raise subprocess.TimeoutExpired('ssh', 15)
        state['status'] = 'paused'
        state['reason'] = 'manual_interrupt'
        row = state['tasks']['val_unseen/pilot']
        row.update(status='scored', result={'metrics': {'success': False}})
        write(tmp_path / 'progress.json', state)
        return True
    monkeypatch.setattr(module, 'recover_startup', recover)
    guard(tmp_path, recover=True)
    assert len(checks) == 2 and checks[1] - checks[0] == 30
    assert (tmp_path / 'PAUSE').exists() and (tmp_path / 'pilot_report_ready.json').exists()


def test_shared_simulator_wait_tracks_dispatcher_across_closed_port_gap(tmp_path, monkeypatch):
    import psutil
    from tests.manual import aerial_vln_pilot_handoff as module
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    write(directory / 'startup_recovery_check.json', {'verified': False,
          'reason': 'local_tunnel_not_verified_closed', 'local_tunnels': [{'port': 35001, 'closed': False}]})
    parent = SimpleNamespace(pid=41, create_time=lambda: 10, cmdline=lambda: ['python.exe', 'other_campaign.py', 'run'])
    listener = SimpleNamespace(pid=42, create_time=lambda: 11, cmdline=lambda: ['ssh', '-L', '127.0.0.1:35001:127.0.0.1:30001'],
                               parents=lambda: [parent])
    processes = {41: parent, 42: listener}
    def lookup(pid):
        if pid not in processes:
            raise psutil.NoSuchProcess(pid)
        return processes[pid]
    monkeypatch.setattr(module.psutil, 'Process', lookup)
    monkeypatch.setattr(module.psutil, 'net_connections', lambda **kwargs: [SimpleNamespace(
        pid=42, status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=35001))])
    assert shared_simulator_busy(tmp_path, list(state['tasks'].values()))
    assert {p['pid'] for p in json.loads((tmp_path / 'shared_simulator_wait.json').read_text())['owners']} == {41, 42}
    processes.pop(42)
    monkeypatch.setattr(module.psutil, 'net_connections', lambda **kwargs: [])
    assert shared_simulator_busy(tmp_path, list(state['tasks'].values()))
    processes.pop(41)
    assert not shared_simulator_busy(tmp_path, list(state['tasks'].values()))
    assert json.loads((tmp_path / 'shared_simulator_wait.json').read_text())['status'] == 'released'


def test_full_split_observer_requires_report_delivery(tmp_path):
    with pytest.raises(ValueError, match='Deliver the pilot report'):
        monitor_split(tmp_path, 'val_seen', recover=True)
    assert not (tmp_path / 'PAUSE').exists()


def test_full_split_completion_does_not_rearm_pilot_barrier(tmp_path):
    task = {'split': 'val_seen', 'episode_id': 'done'}
    other = {'split': 'val_unseen', 'episode_id': 'later'}
    write(tmp_path / 'pilot_report_delivered.json', {'report': 'pilot_report.html'})
    write(tmp_path / 'manifest.json', {'tasks': [task, other]})
    write(tmp_path / 'progress.json', {'status': 'seen_complete', 'reason': None, 'tasks': {
        'val_seen/done': {'task': task, 'status': 'scored', 'attempts': [], 'result': {'metrics': {'success': False}}},
        'val_unseen/later': {'task': other, 'status': 'pending', 'attempts': [], 'result': None}}})
    monitor_split(tmp_path, 'val_seen', recover=True)
    assert not (tmp_path / 'PAUSE').exists()
    assert json.loads((tmp_path / 'split_observer_status.json').read_text())['counts'] == {'scored': 1}
    assert not (tmp_path / 'val_unseen').exists()


def test_full_split_manual_pause_is_preserved(tmp_path, monkeypatch):
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    manifest['tasks'] = manifest['pilot']
    state['reason'] = 'manual_interrupt'
    write(tmp_path / 'progress.json', state)
    write(tmp_path / 'manifest.json', manifest)
    write(tmp_path / 'pilot_report_delivered.json', {'report': 'pilot_report.html'})
    (tmp_path / 'PAUSE').write_text('manual stop', encoding='utf-8')
    monitor_split(tmp_path, 'val_unseen', recover=True)
    assert not launches and (tmp_path / 'PAUSE').exists()


def test_full_split_startup_recovery_uses_all_selected_tasks_and_same_stage(tmp_path, monkeypatch):
    manifest, state, directory, launches, launch = startup_campaign(tmp_path, monkeypatch)
    tasks = manifest['pilot']
    # The interrupted episode is outside the earlier pilot. A split observer
    # must use the selected full queue, rather than filtering by pilot IDs.
    manifest['pilot'] = []
    assert recover_startup(tmp_path, manifest, tasks=tasks, stage='unseen',
                           verify=lambda *args: {'verified': True, 'scene': 'scene_test'}, launch=launch)
    command = launches[0][0][0]
    assert command[command.index('--stage') + 1] == 'unseen'
    assert json.loads((tmp_path / 'progress.json').read_text())['tasks']['val_unseen/pilot']['result'] is None
