"""Experiment-only model selection must match audited requests and preserve inputs."""

import copy
import json

import pytest

from src.llm import ModelResponse
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import encode_request
from src.agent.config import ModelConfig
from src.tools.registry import default_registry
from tests.support.planner import (
    AuditedPlanner,
    guidance_probe as probe,
    make_transcript,
    response_with_calls,
)


def test_model_override_is_immutable_and_optional():
    config = ModelConfig(model_name='original')
    assert probe.model_override(config, None) is config
    assert probe.model_override(config, 'flash').model_name == 'flash'
    assert config.model_name == 'original'
    with pytest.raises(ValueError):
        probe.model_override(config, ' ')


def test_probe_wire_payload_matches_manifest_and_preserves_capture(tmp_path, monkeypatch):
    saved = json.loads(probe.BASELINE.read_text(encoding='utf-8'))
    # This historical prose-only experiment requires a matching historical
    # schema. Do not weaken its drift guard to accept today's retired toggle.
    from types import SimpleNamespace
    monkeypatch.setattr(probe, 'default_registry', lambda: SimpleNamespace(model_specs=lambda: [
        SimpleNamespace(to_provider_tool=lambda t=t: copy.deepcopy(t)) for t in saved['tools']]))
    payload = {'model': 'original', 'messages': [{'role': 'system', 'content': saved['system_prompt']},
               {'role': 'user', 'content': 'recorded observation'}], 'tools': saved['tools'],
               'temperature': 0, 'tool_choice': 'auto'}
    runs = [tmp_path/'aerial', tmp_path/'travel']
    originals = {}
    for run in runs:
        folder = run/'debug/tool_loop/task'
        folder.mkdir(parents=True)
        for index in (1, 6):
            path = folder/f'decision_{index:04}.json'
            value = {'decision_index': index, 'model_rounds': [{'request': payload, 'request_media': []}]}
            path.write_text(json.dumps(value), encoding='utf-8')
            originals[path] = path.read_bytes()
    config = ModelConfig(model_name='original', api_key='test')
    monkeypatch.setattr(probe, 'load_model_config', lambda: config)
    monkeypatch.setattr(probe, 'restore_request', lambda request, media, run: copy.deepcopy(request))
    sent = []

    class Client:
        def __init__(self, settings):
            self.settings = settings

        def request(self, request):
            sent.append(encode_request(self.settings, request))
            response = response_with_calls(('call', 'uav_turn_relative', {'yaw_deg': 10}))
            response['model'] = 'flash-version'
            return ModelResponse(data=response)

    monkeypatch.setattr(probe, 'DefaultModelClient', Client)
    output = tmp_path/'output'
    assert probe.run(runs, output, model='flash') == 'completed'
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    rows = json.loads((output/'results.json').read_text(encoding='utf-8'))['results']
    assert len(sent) == len(rows) == 36
    samples = {s['name']: s for s in manifest['samples']}
    for wire, row in zip(sent, rows):
        sample = samples[row['sample']]
        assert snapshot_request(wire)[0] == sample['snapshots'][row['group']]['request']
        expected = probe.treatments(payload, int(row['sample'].rsplit('_', 1)[1]))[row['group']]
        expected['model'] = 'flash'
        assert wire == expected
        assert row['source_model'] == 'original' and row['returned_model'] == 'flash-version'
    assert all(p.read_bytes() == original for p, original in originals.items())
    assert config.model_name == 'original'
    # Omitted override preserves the original captured/configured model check.
    default_output = tmp_path/'default'
    assert probe.run(runs, default_output, validate_only=True) == 'validated_only'
    default_manifest = json.loads((default_output/'manifest.json').read_text(encoding='utf-8'))
    assert default_manifest['model'] == 'original'
    monkeypatch.setattr(probe, 'load_model_config', lambda: ModelConfig(model_name='other'))
    with pytest.raises(ValueError, match='differs from recorded'):
        probe.run(runs, tmp_path/'mismatch', validate_only=True)
    class IncompatibleClient(Client):
        def request(self, request):
            return ModelResponse(data={'model': 'flash', 'choices': [{'message': {'content': 'text only'}}]})

    monkeypatch.setattr(probe, 'DefaultModelClient', IncompatibleClient)
    failed_output = tmp_path/'incompatible'
    with pytest.raises(ValueError, match='Native tool protocol'):
        probe.run(runs, failed_output, model='flash')
    failed = json.loads((failed_output/'results.json').read_text(encoding='utf-8'))
    assert failed['status'] == 'failed' and len(failed['results']) == 1
    assert (failed_output/'failure.json').is_file()


def test_simulation_audits_returned_model(tmp_path):
    class Client:
        def request(self, request, **kwargs):
            data = response_with_calls(('call', 'uav_turn_relative', {'yaw_deg': 10}))
            data['model'] = 'flash-version'
            return ModelResponse(data=data)

    planner = AuditedPlanner(ModelConfig(model_name='flash', api_key='test'), client=Client())
    planner.plan_turn(make_transcript(tmp_path), default_registry(), decision_index=1)
    assert planner.last_exchange['response']['model_identity'] == {
        'requested_model': 'flash', 'returned_model': 'flash-version'}
