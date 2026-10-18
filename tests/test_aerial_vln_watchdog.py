"""Offline supervisor policy, ownership, persisted backoff and queue isolation."""
from contextlib import contextmanager
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.aerial_vln import parallel_campaign as campaign
from evaluation.benchmarks.aerial_vln import parallel_watchdog as watchdog


def campaign_state(reason='cleanup_unverified'):
    return {'fingerprint': 'frozen', 'status': 'paused', 'reason': reason, 'workers': 2,
            'tasks': {'val_seen/one': {'task': {'split': 'val_seen', 'episode_id': 'one'},
                                     'status': 'retry_pending', 'attempts': [{'cleanup_verified': False}]}}}


@pytest.mark.parametrize('reason', sorted(watchdog.RECOVERABLE_REASONS - {'provider_unavailable'}))
def test_known_interruptions_enter_controller_recovery(reason):
    assert watchdog.resume_policy(campaign_state(reason)) == 'recover'


@pytest.mark.parametrize('reason', ['manual_interrupt', 'non_retryable_provider_error', 'unauthorized_remote_gpu',
                                  'parallel_technical_check', 'episode_attempts_exhausted', 'infrastructure_failure'])
def test_operator_and_unsafe_pauses_never_auto_dispatch(reason):
    assert watchdog.resume_policy(campaign_state(reason)) in {'manual_pause', 'needs_attention'}


def test_manual_pause_requires_explicit_resume_and_seen_completion_stops():
    state = campaign_state('manual_interrupt')
    assert watchdog.resume_policy(state, explicit_resume=True) == 'recover'
    assert watchdog.resume_policy(state, explicit_resume=True, manual_pause=True) == 'manual_pause'
    state.update(status='seen_complete', reason=None)
    assert watchdog.resume_policy(state) == 'seen_complete'


def test_exception_and_storage_classification_does_not_restart_unknown_faults():
    state = campaign_state('controller_exception')
    assert watchdog.resume_policy(state, controller_error={'error_type': 'PermissionError'}) == 'recover'
    assert watchdog.resume_policy(state, controller_error={'error_type': 'ValueError'}) == 'needs_attention'
    assert watchdog.resume_policy(state, controller_error={'error_type': 'UnicodeDecodeError',
        'traceback': 'read_text(session / "stderr.log")'}) == 'recover'
    assert watchdog.resume_policy(state, controller_error={'error_type': 'UnicodeDecodeError',
        'traceback': 'invalid progress.json'}) == 'needs_attention'
    state['reason'] = 'storage_or_coverage_check'
    assert watchdog.resume_policy(state, storage={'passed': False}) == 'recover'
    assert watchdog.resume_policy(state, storage={'passed': True}) == 'needs_attention'


def test_repaired_controller_fault_requires_a_one_use_explicit_resume(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, 'controller_exception')
    campaign.write_json(tmp_path / 'parallel/controller_error.json', {'error_type': 'AttributeError'})
    watcher.tick()
    assert not starts
    (watcher.directory / 'RESUME').write_text('Verified controller repair', encoding='utf-8')
    watcher.tick()
    assert len(starts) == 1 and not (watcher.directory / 'RESUME').exists()
    alive.clear()
    watcher.tick()
    assert len(starts) == 1
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'needs_attention'


def fixture_watchdog(tmp_path, monkeypatch, reason='cleanup_unverified'):
    directory = tmp_path / 'parallel/watchdog'
    directory.mkdir(parents=True)
    script = tmp_path / 'parallel/controller_source/revision/parallel_campaign.py'
    campaign.write_json(tmp_path / 'manifest.json', {'fingerprint': 'frozen', 'configs': {'val_seen': {'env_path': 'frozen.env'}}})
    campaign.write_json(tmp_path / 'parallel/execution_overlay.json', {'controller_script': str(script)})
    campaign.write_json(tmp_path / 'progress.json', campaign_state(reason))
    clock, alive, starts, kills, probes = [1000], set(), [], [], []
    control = SimpleNamespace(
        ParallelRecoveryQueue=campaign.ParallelRecoveryQueue,
        read_json=campaign.read_json, write_json=campaign.write_json, atomic_write_text=campaign.atomic_write_text,
        retry_file_access=campaign.retry_file_access, verify_frozen=lambda *args: None, verify_overlay=lambda *args: None,
        owner_of=lambda process: {'pid': process.pid, 'created': 500, 'command': [str(script), 'run', '--output', str(tmp_path)]},
        matching_process=lambda owner: object() if owner.get('pid') in alive else None,
        kill_owned=lambda owner: (kills.append(owner), alive.discard(owner['pid'])),
        provider_probe=lambda *a, **k: probes.append(True) or False)
    def launch(command, **kwargs):
        starts.append((command, kwargs))
        alive.add(os.getpid())
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(watchdog.subprocess, 'Popen', launch)
    watcher = watchdog.SeenWatchdog(tmp_path, 2, control, clock=lambda: clock[0])
    return watcher, clock, alive, starts, kills, probes


def test_dead_controller_retries_without_modifying_tasks_or_double_launch(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    previous = {'pid': 42, 'created': 10, 'command': ['old-controller']}
    campaign.write_json(tmp_path / 'dispatcher.json', previous)
    before = (tmp_path / 'progress.json').read_bytes()
    assert not watcher.tick() and not starts
    assert watcher.state['next_retry_at'] == 1030
    clock[0] = 1030
    watcher.tick()
    watcher.tick()
    assert len(starts) == 1 and not kills
    assert starts[0][0][0] == watchdog.sys.executable and starts[0][0][-1] == '2'
    assert (tmp_path / 'progress.json').read_bytes() == before
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'monitoring'


def test_live_dispatcher_is_adopted_and_pid_reuse_never_authorizes_kill(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    campaign.write_json(tmp_path / 'dispatcher.json', {'pid': 42, 'created': 999, 'command': ['unrelated']})
    alive.add(42)
    clock[0] = 10000
    for name in ('progress.json', 'parallel/status.json'):
        path = tmp_path / name
        if not path.exists():
            campaign.write_json(path, {})
        os.utime(path, (0, 0))
    watcher.tick()
    assert not starts and not kills
    assert not watchdog.controller_owner_allowed({'command': ['python', 'unrelated.py', 'run']}, tmp_path)


def test_stalled_owned_controller_is_recovered_after_persisted_backoff(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    owner = watcher.control.owner_of(SimpleNamespace(pid=42))
    campaign.write_json(tmp_path / 'dispatcher.json', owner)
    alive.add(42)
    clock[0] = 3000
    os.utime(tmp_path / 'progress.json', (0, 0))
    watcher.tick()
    assert kills == [owner] and not starts
    assert watcher.state['next_retry_at'] == 3030
    # Restarting the watchdog itself must retain this cooldown.
    restored = watchdog.SeenWatchdog(tmp_path, 2, watcher.control, clock=lambda: clock[0])
    restored.tick()
    assert not starts and restored.state['next_retry_at'] == 3030
    clock[0] = 3030
    restored.tick()
    assert len(starts) == 1


def test_live_budget_handoff_is_guarded_without_stall_kill_or_extra_dispatch(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    controller = watcher.control.owner_of(SimpleNamespace(pid=42))
    campaign.write_json(tmp_path / 'dispatcher.json', controller)
    campaign.write_json(tmp_path / 'parallel/scheduling_handoff/status.json', {'status': 'draining', 'controller': controller})
    campaign.write_json(tmp_path / 'parallel/scheduling_handoff/process.json', {
        'pid': 43, 'created': 900, 'command': ['python', 'image_budget_order_handoff.py', '--output', str(tmp_path)]})
    alive.update({42, 43})
    clock[0] = 10000
    os.utime(tmp_path / 'progress.json', (0, 0))
    before = (tmp_path / 'progress.json').read_bytes()
    watcher.tick()
    assert not starts and not kills and (tmp_path / 'progress.json').read_bytes() == before
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'monitoring_handoff'


def test_dead_handoff_is_reported_and_paused_without_starting_competing_controller(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    campaign.write_json(tmp_path / 'parallel/scheduling_handoff/status.json', {'status': 'installing'})
    campaign.write_json(tmp_path / 'parallel/scheduling_handoff/process.json', {
        'pid': 43, 'created': 900, 'command': ['python', 'image_budget_order_handoff.py', '--output', str(tmp_path)]})
    watcher.tick()
    assert not starts and not kills and (tmp_path / 'PAUSE').exists()
    assert campaign.read_json(watcher.directory / 'status.json')['reason'] == 'budget_order_handoff_exited'


def test_provider_outage_probes_before_any_new_task_process(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, 'provider_unavailable')
    before = (tmp_path / 'progress.json').read_bytes()
    watcher.tick()
    assert len(probes) == 1 and not starts and watcher.state['next_retry_at'] == 1030
    watcher.control.provider_probe = lambda *a, **k: True
    clock[0] = 1030
    watcher.tick()
    assert len(starts) == 1 and (tmp_path / 'progress.json').read_bytes() == before


@pytest.mark.parametrize('reason', ['pilot_storage_budget', 'storage_budget', 'storage_or_coverage_check'])
def test_storage_pause_waits_without_relaunch_and_recovers_only_when_capacity_is_sufficient(tmp_path, monkeypatch, reason):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, reason)
    campaign.write_json(tmp_path / 'storage_check.json', {'passed': False, 'required_bytes': 200})
    free = [100]
    monkeypatch.setattr(watchdog.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free[0]))
    before = (tmp_path / 'progress.json').read_bytes()
    for _ in range(3):
        watcher.tick()
    assert not starts and not kills and watcher.state['failures'] == 0
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'waiting_for_storage'
    assert campaign.read_json(watcher.directory / 'storage_wait.json')['required_bytes'] == 200
    free[0] = 200
    watcher.tick()
    assert len(starts) == 1 and starts[0][0][-1] == '2'
    assert (tmp_path / 'progress.json').read_bytes() == before


def test_watchdog_completion_and_manual_pause_do_not_start_a_controller(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    (tmp_path / 'PAUSE').write_text('manual', encoding='utf-8')
    watcher.tick()
    assert not starts and campaign.read_json(watcher.directory / 'status.json')['status'] == 'manual_pause'
    (tmp_path / 'PAUSE').unlink()
    state = campaign_state(None)
    state['status'] = 'seen_complete'
    campaign.write_json(tmp_path / 'progress.json', state)
    assert watcher.tick() and not starts


def test_valid_progress_resets_backoff_and_single_worker_fallback_is_retained(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch)
    watcher.state.update(failures=5, next_retry_at=5000)
    state = campaign_state(None)
    state['workers'] = 1
    state['tasks']['val_seen/one']['status'] = 'scored'
    campaign.write_json(tmp_path / 'progress.json', state)
    watcher.tick()
    assert watcher.state['failures'] == 0 and starts[0][0][-1] == '1'


def paused_transport_receipt():
    state = campaign_state('infrastructure_failure')
    item = state['tasks']['val_seen/one']
    item['attempts'] = [{'number': 1, 'status': 'infrastructure_failed', 'cleanup_verified': True,
                         'valid_score': False, 'result': {'episode_id': 'one', 'metrics': {'success': False},
                             'infrastructure_error': {'stage': 'cleanup', 'errors': [
                                 {'stage': 'final_observation', 'error_type': 'TimeoutError'}]}}}]
    state['pause_cause'] = {'task_key': 'val_seen/one', 'attempt_number': 1, 'reason': 'infrastructure_failure'}
    return state


def test_guard_uses_queue_classifier_to_resume_verified_cleanup_timeout_without_changing_progress(tmp_path, monkeypatch):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, 'infrastructure_failure')
    state = paused_transport_receipt()
    campaign.write_json(tmp_path / 'progress.json', state)
    before = (tmp_path / 'progress.json').read_bytes()
    watcher.tick()
    assert len(starts) == 1 and not kills
    assert (tmp_path / 'progress.json').read_bytes() == before
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'monitoring'


@pytest.mark.parametrize('action', ['move_body', 'turn_relative'])
@pytest.mark.parametrize('stop', [None, 'unverified_cleanup', 'manual_pause'])
def test_guard_resumes_cleaned_motion_discontinuity_once_and_preserves_operator_pause(tmp_path, monkeypatch, action, stop):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, 'infrastructure_failure')
    state = paused_transport_receipt()
    attempt = state['tasks']['val_seen/one']['attempts'][0]
    attempt['result']['infrastructure_error'] = {
        'error_type': 'SceneRecoveryRequired',
        'error': f'AirSim paused Multirotor {action} failed: motion_discontinuity'}
    if stop == 'unverified_cleanup':
        attempt['cleanup_verified'] = False
    if stop == 'manual_pause':
        (tmp_path / 'PAUSE').write_text('operator pause', encoding='utf-8')
    campaign.write_json(tmp_path / 'progress.json', state)
    before = (tmp_path / 'progress.json').read_bytes()
    watcher.tick()
    watcher.tick()
    assert len(starts) == (1 if stop is None else 0) and not kills
    assert (tmp_path / 'progress.json').read_bytes() == before
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == (
        'monitoring' if stop is None else 'manual_pause' if stop == 'manual_pause' else 'needs_attention')


@pytest.mark.parametrize('fault', ['legacy', 'unknown', 'authentication', 'mixed', 'unverified',
                                  'stale_attempt', 'foreign_episode', 'malformed_cause', 'scored'])
def test_guard_never_resumes_unknown_or_mismatched_infrastructure_pause(tmp_path, monkeypatch, fault):
    watcher, clock, alive, starts, kills, probes = fixture_watchdog(tmp_path, monkeypatch, 'infrastructure_failure')
    state = paused_transport_receipt()
    attempt = state['tasks']['val_seen/one']['attempts'][0]
    if fault == 'legacy':
        state.pop('pause_cause')
    elif fault in {'unknown', 'authentication', 'mixed'}:
        error = {'error_type': 'AuthenticationError' if fault == 'authentication' else 'ValueError'}
        if fault == 'mixed':
            attempt['result']['infrastructure_error']['error'] = error
        else:
            attempt['result']['infrastructure_error']['errors'].append(error)
    elif fault == 'unverified':
        attempt['cleanup_verified'] = False
    elif fault == 'stale_attempt':
        state['pause_cause']['attempt_number'] = 2
    elif fault == 'foreign_episode':
        state['pause_cause']['task_key'] = 'val_seen/other'
    elif fault == 'malformed_cause':
        state['pause_cause']['task_key'] = []
    else:
        state['tasks']['val_seen/one']['status'] = 'scored'
    campaign.write_json(tmp_path / 'progress.json', state)
    watcher.tick()
    assert not starts and not kills
    assert campaign.read_json(watcher.directory / 'status.json')['status'] == 'needs_attention'


def test_duplicate_os_watchdog_launch_exits_before_state_or_progress_mutation(tmp_path):
    @contextmanager
    def occupied(_):
        raise BlockingIOError(errno.EACCES, 'another supervisor owns lock')
        yield
    control = SimpleNamespace(campaign_lock=occupied)
    watchdog.run_watchdog(SimpleNamespace(output=tmp_path, workers=2), control)
    assert not list((tmp_path / 'parallel/watchdog').glob('*.json'))
