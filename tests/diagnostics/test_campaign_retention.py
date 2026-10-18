"""Retention cohorts preserve full samples and calibrate compact disk costs."""
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
