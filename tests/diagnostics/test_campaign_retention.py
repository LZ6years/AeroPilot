"""Retention cohorts preserve samples, bounded failures and calibrated disk costs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.core.campaign import retention
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue


def setup_queue(tmp_path):
    tasks = [dict(split='val_unseen', episode_id=str(i), scene_id='7') for i in range(3)]
    queue = ParallelRecoveryQueue(tmp_path / 'progress.json', tasks, 'frozen')
    policy = dict(full_task_keys=[queue.key(t) for t in tasks[:2]], full_seed_bytes=6 * 1024**3)
    return tasks, queue, policy


def test_full_cohort_includes_retries_before_compact_pending(tmp_path):
    tasks, queue, policy = setup_queue(tmp_path)
    queue.state['tasks'][queue.key(tasks[0])]['status'] = 'scored'
    queue.state['tasks'][queue.key(tasks[1])]['status'] = 'retry_pending'
    assert retention.candidates(tasks, queue, policy) == tasks[1:2]
    queue.state['tasks'][queue.key(tasks[1])]['status'] = 'retry_exhausted'
    assert retention.candidates(tasks, queue, policy) == tasks[1:2]
    queue.state['tasks'][queue.key(tasks[1])]['status'] = 'scored'
    assert retention.candidates(tasks, queue, policy) == tasks[2:]


def test_cleaned_exhausted_full_task_releases_compact_work_without_changing_scores(tmp_path, monkeypatch):
    tasks, queue, policy = setup_queue(tmp_path)
    queue.state['tasks'][queue.key(tasks[0])]['status'] = 'scored'
    failed = tasks[1]
    for number in range(1, 4):
        _, attempt = queue.claim([failed], 'A')
        assert attempt['number'] == number
        assert queue.accept(failed, number, dict(cleanup_verified=True, valid_score=False,
            logging_profile='full', retained_bytes=20 * 1024**3, result=dict(
                episode_id=failed['episode_id'], infrastructure_error=dict(
                    stage='scene_control', error=dict(error_type='TimeoutError'))))) == 'deferred'
    before = deepcopy(queue.state)
    assert retention.candidates(tasks, queue, policy) == tasks[2:]
    monkeypatch.setattr(retention.shutil, 'disk_usage', lambda _: SimpleNamespace(free=24 * 1024**3))
    budget = retention.budget(tmp_path, tasks, queue, policy, 3)
    assert budget['logging_profile'] == 'paper_process_metrics_v1'
    assert budget['remaining_cases'] == budget['total_remaining_cases'] == 1
    assert budget['required_bytes'] == 10 * 1024**3 and budget['calibration_pending']
    assert queue.state == before
    item = queue.state['tasks'][queue.key(failed)]
    assert item['status'] == 'retry_exhausted' and item['result'] is None
    assert not queue.terminal(item) and len(item['attempts']) == 3
    assert queue.claim(retention.candidates(tasks, queue, policy), 'B')[0] == tasks[2]


@pytest.mark.parametrize('fault', ['cleanup', 'missing_receipt', 'active_attempt',
                                  'retry_budget_available', 'valid_score', 'retained_result'])
def test_unsettled_full_failure_still_blocks_next_retention_cohort(tmp_path, fault):
    tasks, queue, policy = setup_queue(tmp_path)
    queue.state['tasks'][queue.key(tasks[0])]['status'] = 'scored'
    item = queue.state['tasks'][queue.key(tasks[1])]
    item.update(status='retry_exhausted', attempts=[dict(number=n, status='infrastructure_failed',
                cleanup_verified=True, valid_score=False) for n in range(1, 4)])
    if fault == 'cleanup':
        item['attempts'][0]['cleanup_verified'] = False
    elif fault == 'missing_receipt':
        item['attempts'] = []
    elif fault == 'active_attempt':
        item['attempts'][-1]['status'] = 'running'
    elif fault == 'retry_budget_available':
        item['attempt_limit'] = 4
    elif fault == 'valid_score':
        item['attempts'][-1]['valid_score'] = True
    else:
        item['result'] = dict(metrics=dict(success=False))
    assert retention.candidates(tasks, queue, policy) == tasks[1:2]


def test_capacity_uses_only_current_profile_and_keeps_floor(tmp_path, monkeypatch):
    tasks, queue, policy = setup_queue(tmp_path)
    gib = 1024**3
    monkeypatch.setattr(retention.shutil, 'disk_usage', lambda _: SimpleNamespace(free=24*gib))
    assert retention.budget(tmp_path, tasks, queue, policy, 1)['passed']
    monkeypatch.setattr(retention.shutil, 'disk_usage', lambda _: SimpleNamespace(free=24*gib-1))
    assert not retention.budget(tmp_path, tasks, queue, policy, 1)['passed']
    for task in tasks[:2]:
        queue.state['tasks'][queue.key(task)].update(status='scored', attempts=[dict(retained_bytes=20*gib)])
    result = retention.budget(tmp_path, tasks, queue, policy, 1)
    assert result['required_bytes'] == 10*gib and result['calibration_pending']
    queue.state['tasks'][queue.key(tasks[2])]['attempts'] = [dict(logging_profile='paper_process_metrics_v1', retained_bytes=7*gib)]
    result = retention.budget(tmp_path, tasks, queue, policy, 1)
    assert result['required_bytes'] == 14*gib and not result['calibration_pending']


@pytest.mark.parametrize('change', [dict(full_task_keys=[]), dict(full_task_keys=['unknown']), dict(full_seed_bytes=0)])
def test_invalid_cohort_rejected(tmp_path, change):
    tasks, queue, policy = setup_queue(tmp_path)
    with pytest.raises(ValueError):
        retention.validate({**policy, **change}, tasks)


def test_logger_selection_preserves_full_and_compact_bindings(tmp_path, monkeypatch):
    from evaluation.experiments import compact_logging as compact
    from evaluation.benchmarks.aerial_vln import runner as aerial
    from evaluation.benchmarks.travel_uav import runner as travel
    from evaluation.benchmarks.core import audit
    from src.runtime.task.journal import RuntimeJournal
    # Restore process-wide bindings after checking the private-host adapter.
    for obj, name in ((aerial, 'AsyncRunLogger'), (travel, 'AsyncRunLogger'),
                      (audit, '_copy_recorded_path'), (RuntimeJournal, 'capture_provider_exchange')):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    compact.install(full_episode_ids={'full-episode'})
    full = aerial.AsyncRunLogger(tmp_path / 'full-episode' / 'run', capture_provider_debug=True)
    light = aerial.AsyncRunLogger(tmp_path / 'other-episode' / 'run', capture_provider_debug=True)
    try:
        assert not isinstance(full, compact.MetricsRunLogger)
        assert isinstance(light, compact.MetricsRunLogger)
    finally:
        full.close()
        light.close()
