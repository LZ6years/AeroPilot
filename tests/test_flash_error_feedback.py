"""Reject malformed actions and bad provenance without executing or silently repairing them."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from src.agent.planning.model_action_planner import _parse_assistant_turn, AgentProtocolError
from src.runtime.control.actions.terminal import apply_terminal
from src.tools.registry import default_registry
from tests.support.planner import response_with_calls, schema_contract
from tests.support.process import cleanup_owned, supervise, write_review
from tests.support.runtime import done, harness, move, run_loop


@pytest.mark.parametrize('number',['00.3','00.5','-00.3'])
def test_invalid_number_is_rejected_with_location_and_original_intact(number):
    response=response_with_calls(('bad','uav_move_body',{}))
    message=response['choices'][0]['message']
    message['tool_calls'][0]['function']['arguments']='{"vx": 1, "vy": 0, "vz": '+number+', "duration": 2}'
    before=copy.deepcopy(message)
    with pytest.raises(AgentProtocolError) as error:
        _parse_assistant_turn(message,default_registry())
    assert 'column' in str(error.value) and 'leading zeros' in str(error.value)
    assert number in str(error.value) and message==before


@pytest.mark.parametrize('value',[0.3,-0.3,0,1e-3])
def test_legal_numbers_are_not_changed(value):
    message=response_with_calls(('ok','uav_move_body',{'vx':1,'vy':0,'vz':value,'duration':2}))['choices'][0]['message']
    assert _parse_assistant_turn(message,default_registry()).tool_calls[0].args['vz']==value


def test_historical_schema_guard_preserves_description_named_properties():
    schema={'type':'object','properties':{'description':{'type':'string','description':'old prose'}}}
    changed=copy.deepcopy(schema)
    changed['properties']['description']['description']='new prose'
    assert schema_contract(schema)==schema_contract(changed)
    changed['properties']['description']['type']='number'
    assert schema_contract(schema)!=schema_contract(changed)


def test_rejected_turn_executes_no_action_before_retry(tmp_path):
    runtime,state,scene,backend=harness(tmp_path,[[move('real')],[done()]])
    original=runtime.planner.plan_turn
    first=True
    def reject_then_continue(*args,**kwargs):
        nonlocal first
        if first:
            first=False
            assert not backend.actions
            raise AgentProtocolError('invalid JSON; no actions executed')
        return original(*args,**kwargs)
    runtime.planner.plan_turn=reject_then_continue
    run_loop(runtime, state, scene)
    assert state.completed and state.physical_window_count==1
    assert sum(action=='move_body' for action,_ in backend.actions)==1


@pytest.mark.parametrize('refs',[['obs1','frame1'],['obs1','/image.png'],['previous_task'],['knowledge'],['plan'],['obs1','obs1']])
def test_bad_evidence_feedback_preserves_state(refs):
    state=SimpleNamespace(completed=False,stopped=False)
    before=vars(state).copy()
    with pytest.raises(ValueError) as error:
        apply_terminal(state,{'reason':'Done','evidence_refs':refs},completed=True,available_evidence=frozenset({'obs1','call_env'}))
    assert 'accepted examples' in str(error.value) and 'obs1' in str(error.value)
    assert vars(state)==before


def test_supervisor_records_abrupt_exit(tmp_path):
    out=tmp_path/'child'
    result=supervise([sys.executable,'-u','-c','import os; print("stage-before-exit",flush=True); os._exit(7)'],out,10)
    assert result['exit_code']==7 and not result['timed_out']
    assert 'stage-before-exit' in (out/'stdout.log').read_text(encoding='utf-8')
    assert (out/'resources.jsonl').exists()


def test_supervisor_timeout_is_distinct_from_crash(tmp_path):
    result=supervise([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'timeout',0.1)
    assert result['timed_out'] and result['exit_code'] != 0


def test_interrupted_review_does_not_claim_benchmark_success(tmp_path):
    write_review(tmp_path, {'exit_code': 3221225477, 'timed_out': False})
    page = (tmp_path/'review.html').read_text(encoding='utf-8')
    assert 'interrupted run has no final benchmark score' in page
    assert '3221225477' in page and 'stderr.log' in page


def test_supervisor_cleanup_requires_owned_identity(tmp_path):
    run=tmp_path/'runs'/'case'; (run/'state').mkdir(parents=True); (run/'scenes'/'scene_owned').mkdir(parents=True)
    scene={'scene_instance_id':'scene_owned','map_id':'airvln_remote_env_12','process_ownership':'remote_managed',
           'remote_process_id':123,'control_port':35001}
    (run/'state/scene_instance.json').write_text(json.dumps(scene),encoding='utf-8')
    (run/'scenes/scene_owned/settings.json').write_text('{"ApiServerPort":30001}',encoding='utf-8')
    class Control:
        pid=999
        stopped=False
        def status(self,*args): return {'pid':self.pid,'scene_instance_id':'scene_owned'}
        def stop(self,*args): self.stopped=True; return {'cleanup_verified':True,'live_members':[]}
        def probe_prepared_port(self,*args): return {'status':'unreachable'}
    control=Control()
    config='configs/maps/aerial_vln_s_remote.example.json'
    with pytest.raises(ValueError,match='ownership mismatch'):
        cleanup_owned(tmp_path/'runs',config,[],control=control)
    assert not control.stopped
    control.pid=123
    result=cleanup_owned(tmp_path/'runs',config,[],control=control)
    assert control.stopped and result[0]['endpoint']['status']=='unreachable'
