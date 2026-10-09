"""API lifecycle, idempotency, replay identity and disconnect recovery contracts."""

import json
import time
from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi", reason="Operator API requires requirements-ui.txt")
from fastapi.testclient import TestClient

from src.operator_api.app import create_app
from src.operator_api.config import OperatorConfig
from src.operator_api.services.sessions import SessionManager
from src.operator_api.services.event_stream import EventStream
from src.operator_api.projections.replay import RunRepository, read_jsonl
from src.sim_server.preview import PreviewConfig, PreviewFrame, PreviewService
from src.sim_server.preview.base import add_preview_camera
from tests.test_operator_control import until


@pytest.fixture
def service(tmp_path):
    maps=tmp_path/'maps.json'
    original=json.loads((OperatorConfig().root/'configs/maps/airvln_maps.json').read_text(encoding='utf-8-sig'))
    maps.write_text(json.dumps(original),encoding='utf-8')
    config=OperatorConfig(root=tmp_path,maps_config='maps.json',disconnect_grace_sec=.15)
    owners=[]
    class FakeSession:
        closed=False
        def start_scene(self):
            owners.append(__import__('threading').get_ident())
            return {'scene':{'scene_instance_id':'test-scene'}}
        def submit_instruction(self,instruction,*,control):
            owners.append(__import__('threading').get_ident())
            while control.checkpoint('after_turn',lambda p:sink({'kind':'operator_control','payload':p})):
                time.sleep(.01)
            return {'stopped':True,'completed':False,'instruction':instruction}
        def shutdown(self):
            owners.append(__import__('threading').get_ident())
            self.closed=True
    def compose(config,request,run,s):
        nonlocal sink
        sink=s
        return FakeSession(), SimpleNamespace(open_preview=lambda _: (_ for _ in ()).throw(ValueError('test preview disabled')))
    sink=None
    manager=SessionManager(config,compose=compose)
    with TestClient(create_app(config,manager=manager)) as client:
        yield client,manager,owners


def test_lifecycle_idempotency_disconnect_and_explicit_recovery(service):
    client,manager,owners=service
    result=client.post('/api/sessions',json={'map_id':'cityenviron_win64'})
    assert result.status_code==202
    identity=result.json()['session_id']; target=manager.get(identity)
    until(lambda:target.phase=='ready')
    assert client.post('/api/sessions',json={'map_id':'cityenviron_win64'}).status_code==409
    url=f'/api/sessions/{identity}'
    with client.websocket_connect(url+'/ws') as socket:
        socket.send_json({'type':'ping'})
        request={'instruction':'Hold','request_id':'same'}
        assert client.post(url+'/instructions',json=request).status_code==202
        assert client.post(url+'/instructions',json=request).json()['accepted']
        assert client.post(url+'/instructions',json={**request,'instruction':'Different'}).status_code==409
    until(lambda:target.control.snapshot()['status']=='paused')
    with client.websocket_connect(url+'/ws') as socket:
        socket.send_json({'type':'ping'})
        assert target.control.snapshot()['status']=='paused'
        assert client.post(url+'/control',json={'action':'resume'}).status_code==202
        until(lambda:target.control.snapshot()['status']=='running')
        assert client.post(url+'/control',json={'action':'stop'}).status_code==202
        until(lambda:not target.busy)
    assert client.delete(url).status_code==202
    until(lambda:target.phase=='closed')
    assert len(set(owners))==1
    events=client.get(url+'/events').json()
    assert events and all(e['seq']==i+1 for i,e in enumerate(events))
    assert client.get(url+'/events?after=2').json()[0]['seq']==3


def test_replay_repeated_decisions_are_instruction_scoped_and_inputs_stay_historical(tmp_path):
    repo=RunRepository(tmp_path); run=tmp_path/'saved'; run.mkdir(); identity=repo.register(run)
    (run/'contexts').mkdir();(run/'memory').mkdir();(run/'operator').mkdir()
    (run/'contexts/instruction_envelope.json').write_text(json.dumps({'schema_version':'instruction_envelope_v3'}))
    (run/'memory/agent_transcript.jsonl').write_text('')
    events=[{'seq':i+1,'kind':'model_turn_recorded','instruction_id':f'instruction_{i}','decision_id':'D001','payload':{'canonical_turn':{'public_intent':f'intent_{i}','tool_calls':[]},'observation_id':f'obs_{i}'}} for i in range(2)]
    (run/'operator/events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert repo.decision(identity,'D001','instruction_0')['public_intent']=='intent_0'
    with pytest.raises(ValueError,match='repeats'):repo.decision(identity,'D001')
    with pytest.raises(ValueError):repo.child(run,'../secrets')
    assert repo.asset_url(identity,'../secrets') is None
    assert repo.asset_url(identity,'operator/events.jsonl') is None
    assert repo.decision(identity,'D001','instruction_0')['task_state']=={}


def test_event_snapshots_are_frozen_and_partial_lines_wait(tmp_path):
    stream=EventStream(tmp_path,'s'); payload={'value':[1],'api_key':'private','reasoning_content':'private'}
    stream.publish('one',payload);payload['value'].append(2);stream.close()
    assert stream.since(0)[0]['payload']=={'value':[1]}
    with stream.path.open('a') as f:f.write('{"fragment":')
    assert len(read_jsonl(stream.path))==1


def test_run_list_keeps_recent_numeric_names_ahead_of_old_named_runs(tmp_path):
    import os
    repo=RunRepository(tmp_path)
    for i in range(101):
        run=tmp_path/f'z_old_{i}';(run/'state').mkdir(parents=True)
        state=run/'state/latest_state.json';state.write_text('{"user_instruction":"old"}')
        os.utime(state,(1,1));repo.register(run)
    fresh=tmp_path/'20261008_workbench';(fresh/'state').mkdir(parents=True)
    (fresh/'state/latest_state.json').write_text('{"user_instruction":"new"}')
    identity=repo.register(fresh)
    assert len(repo.list())==100 and repo.list()[0]['run_id']==identity


def test_preview_uses_new_camera_and_closes_its_source():
    import copy
    settings={'Vehicles':{'Drone1':{'Cameras':{'front_center':{'CaptureSettings':[{'ImageType':0,'Width':224,'Height':224}]}}}}}
    original=copy.deepcopy(settings)
    config=PreviewConfig(fps=60)
    add_preview_camera(settings,'Drone1','front_center',config)
    assert settings['Vehicles']['Drone1']['Cameras']['front_center']==original['Vehicles']['Drone1']['Cameras']['front_center']
    assert settings['Vehicles']['Drone1']['Cameras']['ui_preview']['CaptureSettings'][0]['Width']==960
    assert all(isinstance(settings['Vehicles']['Drone1']['Cameras']['ui_preview'][key],float) for key in ('X','Y','Z','Pitch','Roll','Yaw'))
    class Source:
        closed=False
        count=0
        def get_frame(self):
            self.count+=1
            return PreviewFrame(self.count,'now',time.monotonic(),1,960,540,b'jpeg')
        def close(self):self.closed=True
    source=Source();service=PreviewService(lambda:source,config);service.start()
    until(lambda:source.count>=3);service.close()
    assert source.closed and service.latest().frame_id>=3
