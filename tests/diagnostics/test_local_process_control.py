"""Platform selection and command contracts; no simulator or GPU needed."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.env_client.schemas import MapPackage
from src.sim_server.process_control import LocalSceneControl
from src.sim_server.process_manager import ProcessManager


def package(tmp_path):
    return MapPackage.from_dict(dict(map_id='test', provider='packaged_unreal', scene_id='1',
        root=str(tmp_path), connection={'mode': 'local', 'data_transport': 'direct'},
        launch={'windows': 'scene.exe', 'linux': 'scene.sh'},
        control={'protocol': 'airsim'}, airsim={'vehicle_name': 'Drone_1', 'camera_name': 'front_0'}))


@pytest.mark.parametrize('system,name', [('Windows', 'scene.exe'), ('Linux', 'scene.sh')])
def test_platform_selects_executable(monkeypatch, tmp_path, system, name):
    monkeypatch.setattr('platform.system', lambda: system)
    assert Path(LocalSceneControl.resolve_executable(package(tmp_path))).name == name


def test_missing_and_unsupported_platform(monkeypatch, tmp_path):
    p = package(tmp_path)
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    del p.launch['linux']
    with pytest.raises(ValueError, match='no linux executable'):
        LocalSceneControl.resolve_executable(p)
    monkeypatch.setattr('platform.system', lambda: 'Darwin')
    with pytest.raises(ValueError, match='Unsupported'):
        LocalSceneControl.resolve_executable(p)


@pytest.mark.parametrize('style,suffix', [('double_dash', ['--settings', 'settings path.json']),
                                        ('single_dash_equals', ['-settings=settings path.json'])])
def test_linux_bash_arguments(monkeypatch, tmp_path, style, suffix):
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    script = tmp_path / 'scene script.sh'
    script.touch()
    cmd = ProcessManager().build_command(str(script), ['-GraphicsAdapter=3'], 'linux_bash',
                                        'settings path.json', style)
    assert cmd == ['bash', str(script), '-GraphicsAdapter=3', *suffix]


@pytest.mark.parametrize('system,method', [('Linux', 'wsl_bash'), ('Windows', 'linux_bash')])
def test_wrong_platform_method_rejected(monkeypatch, tmp_path, system, method):
    monkeypatch.setattr('platform.system', lambda: system)
    with pytest.raises(ValueError, match='requires'):
        ProcessManager().build_command(str(tmp_path / 'script'), [], method)


def test_local_control_forwards_explicit_settings_and_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    p = package(tmp_path)
    p.launch.update(method='linux_bash', settings_arg_style='double_dash', args=['-GraphicsAdapter=3'])
    manager = Mock()
    manager.start.return_value = 123
    result = LocalSceneControl(manager).start(p, 'test', tmp_path / 'settings.json', tmp_path / 'log')
    assert result.process_id == 123
    assert manager.start.call_args.kwargs['settings_arg_style'] == 'double_dash'
    assert manager.start.call_args.args[2] == ['-GraphicsAdapter=3']


def test_cleanup_failure_retains_ownership(monkeypatch):
    manager = ProcessManager(terminate_timeout_sec=0, kill_timeout_sec=0)
    proc = Mock()
    manager._processes['test'] = proc
    manager._process_groups['test'] = 987654
    monkeypatch.setattr(manager, '_stop_group', Mock(side_effect=RuntimeError('not stopped')))
    with pytest.raises(RuntimeError, match='not stopped'):
        manager.stop('test')
    assert manager._processes['test'] is proc
    assert manager._process_groups['test'] == 987654


def test_duplicate_launch_preserves_owner():
    manager = ProcessManager()
    manager._processes['test'] = Mock()
    with pytest.raises(RuntimeError, match='already owns'):
        manager.start('test', 'unused', [])
