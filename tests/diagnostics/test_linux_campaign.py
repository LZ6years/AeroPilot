"""Explicit GPU policies and recovered Linux groups, without a simulator/API."""
import json
import os
import platform
import signal
import sys

import pytest

from evaluation.benchmarks.core.campaign.parallel import execution_settings, gpu_violation, lane_catalog
from evaluation.benchmarks.core.campaign.local_execution import cleanup_local, owned_members
from src.sim_server.process_manager import ProcessManager


def test_selection_is_distinct_and_deterministic():
    from evaluation.benchmarks.aerial_vln.scene_stability import select_scene_episodes
    rows = [{'scene_id': s, 'episode_id': e} for s, e in [(10, 'c'), (2, 'b'), (10, 'a'), (2, 'a'), (10, 'b')]]
    assert select_scene_episodes(rows) == {'2': ['a', 'b'], '10': ['a', 'b']}
    with pytest.raises(ValueError, match='two distinct'):
        select_scene_episodes([{'scene_id': 1, 'episode_id': 'a'}] * 2)


def test_supplemental_report_keeps_missing_coverage_explicit(tmp_path):
    from evaluation.benchmarks.core.campaign.reporting import write_report
    tasks = {}
    for identity, status, distance in [('near', 'scored', 22), ('missing', 'pending', None)]:
        tasks[identity] = {'task': {'dataset': 'aerial_vln', 'split': 'val_seen', 'episode_id': identity, 'scene_id': '2'},
            'status': status, 'attempts': [{'cleanup_verified': True, 'status': 'scored'}] if status == 'scored' else [],
            'result': {'status': 'ok', 'metrics': {'success': False, 'success_25m': True, 'ndtw': .5,
                'distance_to_goal': distance, 'marginal_20_25m': True}} if status == 'scored' else None}
    report = write_report(tmp_path, {'version': 2, 'dataset': 'aerial_vln', 'fingerprint': 'fixed'},
                          {'tasks': tasks, 'status': 'running'})
    assert report['groups'][0]['provisional']['success_25m'] == 100
    assert report['groups'][0]['paper']['success_25m'] is None
    assert report['marginal_cases'][0]['episode_id'] == 'near'
    assert not report['stability_passed']
    assert 'success_25m' in (tmp_path / 'per_case.csv').read_text(encoding='utf-8-sig')
    assert 'near' in (tmp_path / 'report.html').read_text(encoding='utf-8')


def test_explicit_local_gpu_does_not_authorize_legacy_cards():
    from tests.test_aerial_vln_parallel_campaign import gpu_binding, catalog
    policy = {'mode': 'local', 'gpu_indices': [3], 'require_cuda': False}
    binding = gpu_binding(3)
    data = catalog()
    data['port_policy'] = {'control_port_start': 39131}
    for item in data['maps']:
        item['connection']['mode'] = 'local'
    configured = lane_catalog(data, 'A', binding, policy)
    assert configured['maps'][0]['airsim']['port'] == 39131
    assert configured['maps'][0]['launch']['env']['CUDA_VISIBLE_DEVICES'] == binding['uuid']
    assert gpu_violation({'owned': [{'observed_gpus': [0, 3]}]}, [3])
    assert not gpu_violation({'owned': [{'observed_gpus': [3]}]}, [3])
    with pytest.raises(ValueError):
        execution_settings({'execution': {**policy, 'gpu_indices': [0, 3]}})


@pytest.mark.skipif(platform.system() != 'Linux', reason='Requires real Linux process groups')
def test_recovered_owner_cleanup_and_foreign_group_rejection(tmp_path):
    from tests.diagnostics.test_linux_process_lifecycle import wait_for
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'ApiServerPort': 39991}))
    child_pid = tmp_path / 'child.pid'
    script = tmp_path / 'parent.py'
    child = f"import os,time,pathlib;pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()));time.sleep(60)"
    script.write_text('import subprocess,sys\n' + f'subprocess.Popen([sys.executable,"-c",{child!r}])\n')
    manager = ProcessManager()
    groups = []
    try:
        group = manager.start('test', sys.executable, [str(script)], settings_path=str(settings),
                              log_path=str(tmp_path / 'scene.log'))
        groups.append(group)
        other = manager.start('other', sys.executable, ['-c', 'import time;time.sleep(60)'])
        groups.append(other)
        wait_for(child_pid.exists)
        manager._processes['test'].wait(timeout=5)
        with pytest.raises(RuntimeError, match='token mismatch'):
            owned_members(other, 'wrong-token')
        receipt = cleanup_local(tmp_path)
        assert receipt[0]['remote_stop']['cleanup_verified']
        assert not manager._group_alive(group)
        assert manager._group_alive(other)
        assert cleanup_local(tmp_path)[0]['local_endpoint_closed']
    finally:
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in manager._processes.values():
            proc.wait(timeout=5)
