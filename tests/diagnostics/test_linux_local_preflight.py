"""Preflight path/port failures must remain visible without launching a scene."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('linux_local_preflight', Path(__file__).parents[1] / 'manual/linux_local_preflight.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize('port_free,template_exists,expected', [(True, True, True), (False, True, False), (True, False, False)])
def test_preflight_unavailable_assets_and_ports(monkeypatch, tmp_path, port_free, template_exists, expected):
    monkeypatch.setattr('platform.system', lambda: 'Linux')
    monkeypatch.setattr(probe.PortManager, '_is_available', lambda port: port_free)
    script = tmp_path / 'scene.sh'
    script.write_text('#!/bin/sh\n', encoding='utf-8')
    template = tmp_path / 'settings.json'
    if template_exists:
        template.write_text('{}', encoding='utf-8')
    maps = tmp_path / 'maps.json'
    maps.write_text(json.dumps({'maps': [{'map_id': 'test', 'provider': 'packaged_unreal', 'scene_id': '1',
        'root': str(tmp_path), 'connection': {'mode': 'local', 'data_transport': 'direct'},
        'launch': {'linux': script.name, 'method': 'linux_bash'}, 'control': {'protocol': 'airsim'},
        'airsim': {'vehicle_name': 'Drone_1', 'camera_name': 'front_0', 'settings_template': str(template)}}],
        'port_policy': {'control_port_start': 39131, 'max_scene_num': 1}}), encoding='utf-8')
    config = tmp_path / 'eval.json'
    config.write_text(json.dumps({'map_config_path': str(maps), 'dataset_root': str(tmp_path)}), encoding='utf-8')
    assert probe.inspect_config(config)['valid'] is expected
