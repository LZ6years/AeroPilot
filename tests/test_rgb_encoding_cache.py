"""Content integrity, lifecycle, bounds and offline request equivalence."""
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from types import SimpleNamespace

import pytest
from PIL import Image
from src.perception.frame_payload import RgbEncodingCache, rgb_data_url
from src.memory.context.modules.image import DefaultImageContextModule, latest_rgb_model_inputs_from_frame


def pixels(color, fmt='PNG'):
    stream=BytesIO()
    Image.new('RGB',(20,30),color).save(stream,format=fmt)
    return stream.getvalue()


def frame(color):
    return dict(base64=base64.b64encode(pixels(color)).decode(),width=20,height=30)


@pytest.mark.parametrize('size',[-1,True,1.5,'32'])
def test_invalid_capacity(size):
    with pytest.raises(ValueError): RgbEncodingCache(size)


def test_same_path_same_size_timestamp_replacement_and_source_precedence(tmp_path):
    path=tmp_path/'frame.bmp'
    path.write_bytes(pixels('red','BMP'))
    stamp=path.stat()
    data={**frame('green'),'path':str(path)}
    cache=RgbEncodingCache(100000)
    red=cache.encode(data)
    path.write_bytes(pixels('blue','BMP'))
    os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    assert path.stat().st_size==stamp.st_size
    assert cache.encode(data)!=red
    assert cache.encode(data)==rgb_data_url(data)
    path.unlink()
    assert cache.encode(data)==rgb_data_url(frame('green'))
    with pytest.raises(ValueError): cache.encode({'path':str(path)})
    path.write_bytes(b'broken')
    with pytest.raises(OSError): cache.encode(data)


def test_hits_validate_dimensions_and_rebuild_labels():
    cache=RgbEncodingCache(100000)
    source=frame('red')
    before=latest_rgb_model_inputs_from_frame({**source,'id':'one'},cache=cache)
    after=latest_rgb_model_inputs_from_frame({**source,'id':'two','observation_id':'obs2'},cache=cache)
    assert before[0]['data']==after[0]['data']
    assert after[0]['frame_id']=='two' and after[0]['observation_id']=='obs2'
    with pytest.raises(ValueError): cache.encode({**source,'width':99})
    assert cache.stats()['encodes']==1


def test_lru_size_oversize_and_namespace():
    images=[frame(c) for c in ('red','blue','green','white','black')]
    cache=RgbEncodingCache(100000)
    for f in images: assert cache.encode(f)==rgb_data_url(f)
    assert cache.stats()['entries']==4 and cache.stats()['evictions']==1
    cache.encode(images[1])  # Recently used must survive the next insertion.
    cache.encode(images[0])
    before=cache.stats()['hits']
    cache.encode(images[1])
    assert cache.stats()['hits']==before+1
    cache.encode(images[1],namespace=('new_task',))
    assert cache.stats()['entries']==1
    assert cache.stats()['payload_bytes']==len(rgb_data_url(images[1])[0])
    tiny=RgbEncodingCache(1)
    assert tiny.encode(images[0])==rgb_data_url(images[0])
    assert tiny.stats()['entries']==0
    budget=len(rgb_data_url(images[0])[0])
    bounded=RgbEncodingCache(budget)
    for f in images: bounded.encode(f)
    assert bounded.stats()['payload_bytes']<=budget


def test_concurrent_same_image_encodes_once_and_error_does_not_poison():
    cache=RgbEncodingCache(100000)
    f=frame('red')
    with ThreadPoolExecutor(4) as pool:
        results=list(pool.map(lambda _:cache.encode(f),range(20)))
    assert all(r==rgb_data_url(f) for r in results)
    assert cache.stats()['encodes']==1
    with pytest.raises(ValueError): cache.encode({'base64':'invalid'})
    assert cache.encode(f)==results[0]


def test_default_disabled_reconfigure_and_task_reset(tmp_path):
    from tests.test_context_builder import make_state, make_memory
    from src.perception.context_builder import ObservationContextBuilder
    state=make_state()
    memory=make_memory(tmp_path,state,observations=[{'frames':{'rgb':frame('red')}}])
    builder=ObservationContextBuilder()
    module=builder.composer.registry.modules['image']
    assert module._rgb_encoding_cache.max_bytes==0
    module.configure({'rgb_encoding_cache_max_bytes':100000})
    builder.build(state,memory); builder.build(state,memory)
    assert module._rgb_encoding_cache.stats()['hits']==1
    for field,value in [('active_instruction_id','new'),('reset_epoch',1),('run_id','newrun'),('scene_instance_id','newscene')]:
        setattr(memory,field,value)
        before=module._rgb_encoding_cache.stats()['encodes']
        builder.build(state,memory)
        assert module._rgb_encoding_cache.stats()['encodes']==before+1
        assert module._rgb_encoding_cache.stats()['entries']==1
    module.configure({})
    assert module._rgb_encoding_cache.stats()['entries']==0
    assert module._rgb_encoding_cache.max_bytes==100000
    builder.build(state,memory)
    memory.active_instruction_id='empty_next_task'
    memory.recent_observation_refs.clear()
    builder.build(state,memory)
    assert module._rgb_encoding_cache.stats()['entries']==0


def test_constructed_full_context_and_captured_request_equivalence(tmp_path):
    from tests.test_context_builder import make_state, make_memory
    from src.perception.context_builder import ObservationContextBuilder
    from src.agent.planning.model_action_planner import ModelActionPlanner
    from src.runtime.configs.model_config import ModelConfig
    from src.agent.transcript import AgentTranscript
    from src.tools.registry import default_registry
    state=make_state()
    memory=make_memory(tmp_path,state,observations=[{'frames':{'rgb':frame('blue')}}])
    builder=ObservationContextBuilder()
    captured=[]
    class Stop(Exception): pass
    class Planner(ModelActionPlanner):
        def _request(self,payload):
            captured.append(payload)
            raise Stop
    planner=Planner(ModelConfig(api_key='offline',model_name='offline'),client=object())
    contexts=[]
    for capacity in (0,100000,100000):
        builder.composer.registry.modules['image'].configure({'rgb_encoding_cache_max_bytes':capacity})
        context=builder.build(state,memory)
        # Second build exercises a real cache hit, including full metadata.
        assert context==builder.build(state,memory)
        contexts.append(context)
        transcript=AgentTranscript(SimpleNamespace(write_transcript_event=lambda **kw:None))
        transcript.start_instruction(state.user_instruction,'inst1')
        transcript.append_state(context,'obs_0001')
        with pytest.raises(Stop): planner.plan_turn(transcript,default_registry(),decision_index=6)
    assert contexts[0]==contexts[1]==contexts[2]
    assert captured[0]==captured[1]==captured[2]


def test_depth_mode_bypasses_cache(tmp_path):
    from tests.test_context_builder import make_state, make_memory
    from src.perception.context_builder import ObservationContextBuilder
    state=make_state()
    memory=make_memory(tmp_path,state,observations=[{'frames':{'rgb':frame('red')}}])
    builder=ObservationContextBuilder()
    module=builder.composer.registry.modules['image']
    module.configure({'include_current_depth':True,'rgb_encoding_cache_max_bytes':100000})
    builder.build(state,memory)
    assert module._rgb_encoding_cache.stats()['encodes']==0
