"""Offline regression checks for exact media recovery and response acceptance."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('latency_replay', Path(__file__).parents[1] / 'manual/api_latency_replay.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_restore_and_reject_corrupted_media(tmp_path):
    data = b'original-image-bytes'
    (tmp_path / 'image.png').write_bytes(data)
    sha = probe.digest(data)
    record = {'request': {'messages': [{'url': f'[provider_media sha256={sha}]'}]},
              'request_media': [{'asset_ref': 'image.png', 'sha256': sha,
                                 'request_path': ['messages', 0, 'url'], 'mime_type': 'image/png'}]}
    restored = json.loads(probe.restore_request(record, tmp_path))
    assert restored['messages'][0]['url'].startswith('data:image/png;base64,')
    assert record['request']['messages'][0]['url'].startswith('[provider_media')
    (tmp_path / 'image.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        probe.restore_request(record, tmp_path)


@pytest.mark.parametrize('body,valid', [
    (b'{"choices":[{"finish_reason":"tool_calls"}],"usage":{"total_tokens":4}}', True),
    (b'data: {"choices":[{"finish_reason":"stop"}]}\n\ndata: [DONE]\n', True),
    (b'data: {"choices":[{"finish_reason":null}]}\n\ndata: [DONE]\n', False),
    (b'data: {"error":{"message":"upstream failed"}}\n\ndata: [DONE]\n', False),
    (b'<html>gateway error</html>', False),
])
def test_response_validation(body, valid):
    assert probe.response_info(body)['valid'] is valid
