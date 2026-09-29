"""Review-only native transport preserves CDP payloads and cleans up on errors."""
import io
import json
from types import SimpleNamespace

import pytest

from tests.manual.review_browser_probe import NativeWebSocket


def transport(replies):
    socket = NativeWebSocket.__new__(NativeWebSocket)
    socket.process = SimpleNamespace(
        stdin=io.StringIO(), stdout=io.StringIO(replies))
    return socket


def test_native_transport_preserves_json_and_unicode():
    payload = json.dumps({'id': 7, 'text': '图像\n`$()"'})
    socket = transport(json.dumps({'ok': True}) + '\n' +
                       json.dumps({'data': payload}) + '\n')
    socket.send(payload)
    assert socket.recv() == payload
    commands = list(map(json.loads, socket.process.stdin.getvalue().splitlines()))
    assert commands == [{'op': 'send', 'payload': payload}, {'op': 'recv'}]


@pytest.mark.parametrize('reply, message', [
    ('', 'exited without'), ('{"error":"timeout"}\n', 'timeout')])
def test_native_transport_reports_failure(reply, message):
    with pytest.raises(RuntimeError, match=message):
        transport(reply).recv()


def test_native_transport_terminates_bridge_when_close_fails():
    socket = transport('{"error":"disconnected"}\n')
    calls = []
    socket.process.poll = lambda: None
    socket.process.kill = lambda: calls.append('kill')
    socket.process.communicate = lambda **kwargs: calls.append('reap')
    with pytest.raises(RuntimeError, match='disconnected'):
        socket.close()
    assert calls == ['kill', 'reap']
