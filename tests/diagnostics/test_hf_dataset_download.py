"""Offline checks for download selection and bounded retry behavior."""
import errno
import json
import sys
from types import SimpleNamespace

import pytest

from tools.deployment.download_hf_dataset import main, positive, retry, selected_files


def test_selection_keeps_all_split_archive_parts():
    files = [SimpleNamespace(rfilename=n) for n in ('closeloop_envs.z01', 'closeloop_envs.zip', 'other.zip')]
    assert selected_files(files, ['closeloop_envs.*']) == files[:2]
    with pytest.raises(ValueError):
        selected_files(files, ['*test*'])


def test_transient_retry_and_limit():
    calls, sleeps = [], []
    def action():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError()
        return 'ok'
    assert retry(action, 3, sleeps.append) == 'ok'
    assert sleeps == [5, 10]
    with pytest.raises(TimeoutError):
        retry(lambda: (_ for _ in ()).throw(TimeoutError()), 2, lambda _: None)


@pytest.mark.parametrize('error', [OSError(errno.ENOSPC, 'full'), PermissionError(errno.EACCES, 'denied'), ValueError('bad')])
def test_permanent_failures_do_not_retry(error):
    sleeps = []
    with pytest.raises(type(error)):
        retry(lambda: (_ for _ in ()).throw(error), 5, sleeps.append)
    assert not sleeps


def test_auth_failure_does_not_retry():
    error = RuntimeError('auth')
    error.response = SimpleNamespace(status_code=401)
    with pytest.raises(RuntimeError):
        retry(lambda: (_ for _ in ()).throw(error), 5, lambda _: pytest.fail('must not retry'))


def test_positive_arguments():
    assert positive('4') == 4
    with pytest.raises(Exception):
        positive('0')


def test_manifest_pins_restart_and_list_only_does_not_download(tmp_path, monkeypatch):
    calls = []
    class Api:
        def __init__(self, **kwargs):
            assert kwargs['token'] is False
        def repo_info(self, repo, **kwargs):
            calls.append(kwargs['revision'])
            return SimpleNamespace(sha='fixed-commit', siblings=[SimpleNamespace(rfilename='env.zip', size=3)])
    def download(**kwargs):
        assert kwargs['revision'] == 'fixed-commit'
        assert kwargs['token'] is False
        (tmp_path / 'env.zip').write_bytes(b'zip')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=Api, snapshot_download=download))
    args = ['--repo', 'owner/env', '--local-dir', str(tmp_path)]
    assert main(args + ['--list-only']) == 0
    assert not (tmp_path / 'env.zip').exists()
    assert main(args) == 0
    assert main(args + ['--endpoint', 'https://huggingface.co']) == 0
    assert calls == ['main', 'main', 'fixed-commit']
    assert json.loads((tmp_path / '.flyagent-download.json').read_text())['commit'] == 'fixed-commit'
    with pytest.raises(ValueError, match='another repository'):
        main(['--repo', 'other/repo', '--local-dir', str(tmp_path)])
