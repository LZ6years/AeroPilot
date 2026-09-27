"""Checks for frozen-frame resolution, honest timing and the cache entry gate."""
import base64
from io import BytesIO

import pytest
from PIL import Image
from src.perception.frame_payload import rgb_data_url
from tests.manual.image_context_probe import eligible, measured_stages, resolve_frame, speed_gate


def test_stages_match_production_bytes():
    out=BytesIO()
    Image.new('RGB',(20,30),(11,27,59)).save(out,format='PNG')
    frame={'base64':base64.b64encode(out.getvalue()).decode(),'width':20,'height':30}
    result, stages, count=measured_stages(frame)
    assert result==rgb_data_url(frame)[0]
    assert count==len(out.getvalue())
    assert set(stages)=={'read','decode_rgb','png','base64','input_assembly'}
    assert all(value>=0 for value in stages.values())


def test_entry_gate_requires_repeat_and_three_slow_samples():
    samples=[{'conversion':{'p50_sec':v}} for v in (.002,.0021,.003,.0001,.0002,.0003)]
    assert eligible(samples,[{'adjacent_repeats':1}])
    assert not eligible(samples,[{'adjacent_repeats':0}])
    samples[0]['conversion']['p50_sec']=.00199
    assert not eligible(samples,[{'adjacent_repeats':1}])


def test_frame_reference_mismatch_rejected(tmp_path):
    records={'observations':[{'id':'obs1','frame_refs':[{'id':'f1','ref':'memory/frames.jsonl#1','modality':'rgb'}]}],
             'frames':[{'id':'future','observation_id':'obs2'}]}
    with pytest.raises(ValueError,match='mismatch'):
        resolve_frame({'checkpoint_observation_id':'obs1'},tmp_path,records)


def test_speed_gate_checks_absolute_relative_tail_and_miss_cost():
    repeated={'baseline':{'p50_sec':.008,'p95_sec':.010},'candidate':{'p50_sec':.001,'p95_sec':.002}}
    missed={'baseline':{'p50_sec':.008},'candidate':{'p50_sec':.0085}}
    assert speed_gate(repeated,missed)
    missed['candidate']['p50_sec']=.010
    assert not speed_gate(repeated,missed)
    missed['candidate']['p50_sec']=.008
    repeated['candidate']['p95_sec']=.012
    assert not speed_gate(repeated,missed)
