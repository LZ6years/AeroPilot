"""Serial lane isolation and exact distance-review boundary regressions."""
import copy
import json

import pytest

from evaluation.benchmarks.aerial_vln import parallel_campaign as controller
from evaluation.benchmarks.aerial_vln.parallel_recovery import ParallelRecoveryQueue
from tests.manual import aerial_vln_plan_context_probe as probe
from tests.manual.aerial_vln_hard10_serial_retest import serial_manifest


@pytest.mark.parametrize('distance,success,near,gap',[
    (20,False,False,0),(20.001,False,True,.001),(25,False,True,5),
    (25.001,False,False,5.001),(17,True,False,-3),(24.78,None,None,4.78),
    (None,False,None,None),(float('nan'),False,None,None)])
def test_distance_review_preserves_scores(distance,success,near,gap):
    result={'metrics':{'distance_to_goal':distance,'success':success},'stopped':True}
    review=probe.arrival_distance_review(result)
    assert review['near_boundary_failure'] is near
    assert review['gap_to_20m_m'] == (pytest.approx(gap) if gap is not None else None)
    assert result['metrics']['success'] is success


def test_serial_retains_case_order_and_gpu_without_overlap(tmp_path):
    tasks=[{'split':'val_seen','episode_id':str(i)} for i in range(3)]
    queue=ParallelRecoveryQueue(tmp_path/'progress.json',tasks,'test')
    manifest={'tasks':tasks,'dispatch_concurrency':1,'task_lane_assignments':{'0':'B','1':'A','2':'B'}}
    assert controller.dispatch_candidates(manifest,tasks,'A',{},2,queue)==[]
    assert controller.dispatch_candidates(manifest,tasks,'B',{},2,queue)==[tasks[0]]
    assert controller.dispatch_candidates(manifest,tasks,'A',{'B':{}},2,queue)==[]
    queue.state['tasks'][queue.key(tasks[0])]['status']='scored'
    assert controller.dispatch_candidates(manifest,tasks,'A',{},2,queue)==[tasks[1]]
    assert controller.dispatch_candidates(manifest,tasks,'B',{},2,queue)==[]
    assert controller.dispatch_candidates({'tasks':tasks},tasks,'A',{},2,queue)==tasks[1:]
    manifest['task_lane_assignments']['1']='C'
    with pytest.raises(ValueError):
        controller.dispatch_candidates(manifest,tasks,'A',{},2,queue)


def test_serial_manifest_preserves_frozen_source_and_config(tmp_path):
    previous=tmp_path/'previous'; previous.mkdir()
    for filename in ('progress.json','manifest.json'):
        (previous/filename).write_text('{}',encoding='utf-8')
    tasks=[{'split':'val_seen','episode_id':str(i)} for i in range(10)]
    original={'tasks':tasks,'source_hashes':{'src/agent/context/projection.py':'old'},'model':{'name':'same'},
        'config':{'runtime_loop_guard_windows':300},'previous_root':'older',
        'cohorts':[{'name':'hard10'}],'fingerprint':'old'}
    state={'status':'seen_complete','tasks':{ParallelRecoveryQueue.key(t):{
        'status':'scored','attempts':[{'cleanup_verified':True,'lane':'A' if i%2 else 'B'}]}
        for i,t in enumerate(tasks)}}
    before=copy.deepcopy(original)
    new=serial_manifest(original,previous,tmp_path/'new',state)
    assert original==before
    assert new['source_hashes']==original['source_hashes'] and new['model']==original['model']
    assert new['config']['runtime_loop_guard_windows']==300
    assert new['task_lane_assignments']['0']=='B' and new['dispatch_concurrency']==1
    assert new['retest']['post_navigation_photo_repair_included'] is False
