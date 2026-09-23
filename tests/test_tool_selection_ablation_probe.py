"""Ablation alters request copies only; no tool execution or simulator access."""
import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.llm.contracts import ModelResponse
from tests.manual import tool_selection_ablation_probe as probe


def fixture_case(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    raw = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
    (run/'image.png').write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    image = {'type':'image_url','image_url':{'url':f'[provider_media sha256={sha}]'}}
    messages = [{'role':'system','content':prefix+'original'} for prefix in probe.PREFIXES]
    messages += [{'role':'system','content':"You are FlyAgent's mission tool-loop agent. original"},
                 {'role':'user','content':[{'type':'text','text':'CURRENT obs_1'},image]}]
    tool = {'type':'function','function':{'name':'uav_hover','description':'original',
            'parameters':{'type':'object','additionalProperties':False,'properties':{}}}}
    payload = {'model':'stub','messages':messages,'tools':[tool], 'tool_choice':'auto','temperature':0}
    media = [{'request_path':['messages',3,'content',1,'image_url','url'], 'sha256':sha,
              'mime_type':'image/png','asset_ref':'image.png','timeline_role':'CURRENT'}]
    rounds = []
    for i in range(3):
        request = copy.deepcopy(payload)
        if i:
            request['messages'] += [
                {'role':'assistant','content':'old interpretation', 'tool_calls':[{'id':'ground',
                  'type':'function','function':{'name':'grounding','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'ground','content':'Candidate, not confirmed'}]
        if i==2:
            request['messages'].append({'role':'user','content':'another unchanged checkpoint'})
        rounds.append({'request':request,'request_media':media,
                       'canonical_turn':{'tool_calls':[{'tool_name':'runtime.report_blocked'}] if i==2 else []}})
    audit = {'run_dir':str(run), 'planner_debug':[{'model_rounds':rounds}],
             'agent_transcript':[{'kind':'tool_result','payload':{
                 'tool_name':'skill.visual_grounding.locate','tool_call_id':'ground','status':'ok'}}]}
    case = tmp_path/'case'
    case.mkdir()
    probe.write_json(case/'audit.json',audit)
    return case, audit


@pytest.mark.parametrize('group,removed', [('A',()),('B',(0,)),('C',(1,)),('D',(0,1))])
def test_ablation_removes_only_system_blocks_and_restores_image_indices(tmp_path,group,removed):
    case, audit = fixture_case(tmp_path)
    source = copy.deepcopy(audit)
    sample = probe.select_samples(case)[1]
    payload = sample['payload']
    actual = probe.variant(payload,group)
    assert actual['messages'] == [m for i,m in enumerate(payload['messages']) if i not in removed]
    assert actual['tools'] == payload['tools']
    assert actual['temperature'] == 0 and actual['tool_choice']=='auto'
    snapshot, media = probe.snapshot_request(actual)
    assert media[0]['request_path'][1] == 3-len(removed)
    assert media[0]['sha256'] == sample['media'][0]['sha256']
    assert probe.read_json(case/'audit.json') == source
    assert len(payload['messages']) == 6


@pytest.mark.parametrize('fault', ['missing','sha','outside','duplicate','unmapped'])
def test_media_corruption_fails_closed(tmp_path,fault):
    case, audit = fixture_case(tmp_path)
    r = audit['planner_debug'][0]['model_rounds'][0]
    if fault=='missing':
        (tmp_path/'run/image.png').unlink()
    elif fault=='sha':
        (tmp_path/'run/image.png').write_bytes(b'tampered')
    elif fault=='outside':
        (tmp_path/'outside.png').write_bytes((tmp_path/'run/image.png').read_bytes())
        r['request_media'][0]['asset_ref']='../outside.png'
    elif fault=='duplicate':
        r['request_media'] *= 2
    else:
        r['request_media']=[]
    with pytest.raises((ValueError,FileNotFoundError)):
        probe.restore_request(r['request'],r['request_media'],audit['run_dir'])


def test_candidate_image_and_same_pixel_observation_are_not_dropped(tmp_path):
    case, audit = fixture_case(tmp_path)
    r = copy.deepcopy(audit['planner_debug'][0]['model_rounds'][0])
    r['request']['messages'].append({'role':'user','content':[copy.deepcopy(r['request']['messages'][3]['content'][1])]})
    second = copy.deepcopy(r['request_media'][0])
    second.update(request_path=['messages',4,'content',0,'image_url','url'],timeline_role='TOOL_EVIDENCE')
    r['request_media'].append(second)
    restored = probe.restore_request(r['request'],r['request_media'],audit['run_dir'])
    _, media = probe.snapshot_request(probe.variant(restored,'D'))
    assert len(media)==2 and media[0]['sha256']==media[1]['sha256']


def test_validate_only_does_not_create_client_or_read_credentials(tmp_path,monkeypatch):
    case, _ = fixture_case(tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError('Provider/config must not run during validation')
    monkeypatch.setattr(probe,'load_model_config',forbidden)
    monkeypatch.setattr(probe,'DefaultModelClient',forbidden)
    result = probe.run([case],tmp_path/'out',validate_only=True)
    assert result == {'status':'validated_only','planned_requests':36}
    assert probe.read_json(tmp_path/'out/results.json')['results']==[]


def test_snapshots_and_order_are_explicit_and_repeatable(tmp_path):
    case, _ = fixture_case(tmp_path)
    samples=probe.select_samples(case)
    assert [s['kind'] for s in samples]==['initial','post_grounding','pre_blocked']
    order=list(probe.schedule(6,3))
    assert len(order)==72 and len(set(order))==72
    assert order[:4]==[(0,1,g) for g in 'ABCD']
    assert order[4:8]==[(0,2,g) for g in 'BCDA']
    assert order==list(probe.schedule(6,3))


class Client:
    def __init__(self,fail=False,malformed=False):
        self.calls=[]
        self.fail=fail
        self.malformed=malformed

    def request(self,request,**kwargs):
        self.calls.append(request)
        if self.fail:
            raise TimeoutError('do not persist api_key=secret')
        return ModelResponse({'choices':[{'finish_reason':'tool_calls','message':{
            'content':'<script>public</script>', 'reasoning_content':'PRIVATE_THOUGHT',
            'tool_calls':[{'id':'new','type':'function','function':{'name':'uav_hover',
                          'arguments':'invalid' if self.malformed else '{}'}}]}}],
            'api_key':'SECRET'}, {'attempt_count':1,'retry_count':0})


def test_public_output_is_escaped_private_fields_removed_no_continuations(tmp_path):
    case,_=fixture_case(tmp_path)
    client=Client(malformed=True)
    result=probe.run([case],tmp_path/'out',repeats=1,client_factory=lambda:client)
    assert result['status']=='completed' and len(client.calls)==12
    output=(tmp_path/'out/results.json').read_text(encoding='utf-8')
    assert 'PRIVATE_THOUGHT' not in output and 'SECRET' not in output and 'data:image/' not in output
    rows=json.loads(output)['results']
    assert all(r['wire_check']['issues'] and not r['tools_executed'] for r in rows)
    assert all('new' not in {c['id'] for m in q.messages for c in m.get('tool_calls',[])} for q in client.calls)
    page=(tmp_path/'out/review.html').read_text(encoding='utf-8')
    assert '<script>public</script>' not in page and '&lt;script&gt;' in page


def test_provider_failure_stops_without_retry_or_leaking_exception(tmp_path):
    case,_=fixture_case(tmp_path)
    client=Client(fail=True)
    result=probe.run([case],tmp_path/'out',client_factory=lambda:client)
    assert result['status']=='failed' and len(client.calls)==1
    assert 'secret' not in (tmp_path/'out/results.json').read_text(encoding='utf-8')


def test_probe_has_no_execution_imports():
    import ast
    tree=ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
    imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert not any(name and any(s in name for s in ('runners','sim_server','dispatcher','env_client')) for name in imports)


def test_duplicate_or_missing_system_blocks_rejected(tmp_path):
    case,_=fixture_case(tmp_path)
    payload=probe.select_samples(case)[0]['payload']
    payload['messages'].append(copy.deepcopy(payload['messages'][0]))
    with pytest.raises(ValueError,match='exactly one'):
        probe.variant(payload,'A')
