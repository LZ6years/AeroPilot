"""Exercise configurable GPU lanes, trial waves and audit selection offline."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.core.campaign import parallel, retention
from evaluation.benchmarks.core.campaign.recovery import ParallelRecoveryQueue
from tests.test_aerial_vln_parallel_campaign import catalog, gpu_binding


@pytest.mark.parametrize('indices', [[2], [0, 2], [1, 2, 3]])
def test_local_lanes_pin_distinct_gpus_and_ports_without_mutating_catalog(indices):
    policy = dict(mode='local', gpu_indices=indices)
    base = catalog()
    base['port_policy']['control_port_start'] = 39141
    for item in base['maps']:
        item['connection']['mode'] = 'local'
    original = deepcopy(base)
    configured = [parallel.lane_catalog(base, lane, gpu_binding(index), policy)
                  for lane, index in zip(parallel.execution_lanes(policy), indices)]
    assert [c['maps'][0]['airsim']['port'] for c in configured] == list(range(39141, 39141 + len(indices)))
    assert [c['maps'][0]['launch']['env']['CUDA_VISIBLE_DEVICES'] for c in configured] == [gpu_binding(i)['uuid'] for i in indices]
    assert base == original
    with pytest.raises(ValueError, match='Worker count'):
        parallel.execution_lanes(policy, len(indices) + 1)


def test_busy_shared_card_does_not_block_other_pinned_cards(monkeypatch):
    sample = dict(gpus=[dict(index=i, uuid=gpu_binding(i)['uuid'], memory_free=free)
                        for i, free in [(1, '8000 MiB'), (2, '4095 MiB'), (3, '9000 MiB')]])
    monkeypatch.setattr(parallel.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=json.dumps(sample)))
    policy = dict(mode='local', gpu_indices=[1, 2, 3], allow_shared_gpu=True,
                  gpu_uuids={str(i): gpu_binding(i)['uuid'] for i in (1, 2, 3)})
    assert [parallel.local_gpu_idle(policy, lane) for lane in parallel.execution_lanes(policy)] == [True, False, True]


def test_discovery_and_capture_require_all_three_lanes(monkeypatch, tmp_path):
    policy = dict(mode='local', gpu_indices=[1, 2, 3],
                  gpu_uuids={str(i): gpu_binding(i)['uuid'] for i in (1, 2, 3)})
    monkeypatch.setattr(parallel.sys, 'platform', 'linux')
    monkeypatch.setattr(parallel, 'load_map_config', lambda _: ([SimpleNamespace(connection={'mode': 'local'})], {}))
    monkeypatch.setattr(parallel.subprocess, 'run', lambda args, **kw: SimpleNamespace(stdout=json.dumps(gpu_binding(int(args[-1])))))
    assert list(parallel.discover_gpu_bindings('maps', policy)) == ['A', 'B', 'C']
    maps = tmp_path / 'maps.json'
    maps.write_text(json.dumps({'maps': [{'scene_id': 'one'}, {'scene_id': 'two'}]}))
    overlay = dict(lanes={lane: dict(map_config_path=str(maps), sha256=lane) for lane in 'ABC'})
    captures = {lane: dict(passed=True, map_sha256=lane, scenes=[{'scene_id': name}])
                for lane, name in [('A', 'one'), ('B', 'two'), ('C', 'one')]}
    record = dict(passed=True, lane_captures=captures)
    parallel.validate_local_capture(record, overlay, policy)
    del captures['C']
    with pytest.raises(RuntimeError, match='every execution lane'):
        parallel.validate_local_capture(record, overlay, policy)


def test_three_case_waves_and_recovery_never_duplicate_a_scored_task(tmp_path):
    tasks = [dict(dataset='travel_uav', split='test_seen', episode_id=str(i), scene_id='one', difficulty='easy')
             for i in range(7)]
    q = ParallelRecoveryQueue(tmp_path / 'progress.json', tasks, 'frozen')
    assert parallel.stage_candidates(tasks, tasks[:6], False, q, 3) == tasks[:3]
    claimed = [q.claim(tasks[:3], lane) for lane in 'ABC']
    assert len({t['episode_id'] for t, _ in claimed}) == 3
    for task, attempt in claimed:
        q.accept(task, attempt['number'], dict(cleanup_verified=True, valid_score=True, process={},
                  result=dict(metrics={'success': False})))
    restored = ParallelRecoveryQueue(tmp_path / 'progress.json', tasks, 'frozen')
    assert parallel.stage_candidates(tasks, tasks[:6], False, restored, 3) == tasks[3:6]
    assert restored.claim(tasks[:3], 'C') is None


def test_audit_strata_and_hashes_ignore_input_order_and_outcomes():
    tasks = []
    for scene, difficulty, size in [('a', 'easy', 70), ('a', 'hard', 30), ('b', 'easy', 40), ('b', 'hard', 60)]:
        tasks.extend(dict(dataset='travel_uav', split='test_seen', scene_id=scene, difficulty=difficulty,
                          episode_id=f'{scene}-{difficulty}-{i}') for i in range(size))
    chosen = retention.select_full_audit(tasks, 10)
    assert Counter((t['scene_id'], t['difficulty']) for t in chosen) == {('a', 'easy'): 4, ('a', 'hard'): 1, ('b', 'easy'): 2, ('b', 'hard'): 3}
    changed = [dict(t, success=True) for t in reversed(tasks)]
    assert [t['episode_id'] for t in retention.select_full_audit(changed, 10)] == [t['episode_id'] for t in chosen]
    with pytest.raises(ValueError, match='duplicate'):
        retention.select_full_audit(tasks + tasks[:1], 10)


def test_source_audit_counts_episodes_not_frame_rows(monkeypatch, tmp_path):
    from evaluation.benchmarks.travel_uav import full_seen
    source = tmp_path / 'seen.json'
    source.write_text(json.dumps([{'json': 'dataset/scene/first/merged_data.json'}] * 5))
    provenance = SimpleNamespace(index_sha256='index', source_split_sha256='old')
    monkeypatch.setattr(full_seen, 'load_travel_uav_index', lambda *a, **kw: ([SimpleNamespace(episode_id='scene__first')], provenance))
    _, record = full_seen.audit_source('index', source)
    assert record['frame_rows'] == 5 and record['released_unique_episodes'] == 1
    source.write_text(json.dumps([{'json': 'dataset/scene/other/merged_data.json'}]))
    with pytest.raises(ValueError, match='differs'):
        full_seen.audit_source('index', source)
