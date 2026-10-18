"""Guard checks are offline and never launch a simulator or contact an API."""
from tests.manual.guarded_proxy_campaign import proxy_environment, proxy_ready


def test_proxy_environment_overrides_bypass_without_changing_parent(monkeypatch):
    import os
    monkeypatch.setenv('NO_PROXY', '*')
    env = proxy_environment('http://127.0.0.1:17890')
    assert env['NO_PROXY'] == '127.0.0.1,localhost,::1'
    assert env['HTTPS_PROXY'] == env['https_proxy'] == 'http://127.0.0.1:17890'
    assert os.environ['NO_PROXY'] == '*'


def test_connect_failure_is_not_direct_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('unavailable')
    monkeypatch.setattr('socket.create_connection', fail)
    assert not proxy_ready('http://127.0.0.1:17890', 'https://api.example.test/v1')
