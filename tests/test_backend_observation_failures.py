"""Failure-path contracts: never turn failed sensors into normal observations."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.adapters.uav_primitives import PrimitiveResult
from src.adapters.airsim.client import probe_airsim_endpoint
from src.adapters.unrealcv.client import probe_unrealcv_endpoint
from src.adapters.unrealcv.uav import UnrealCvUavAdapter
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend


@pytest.mark.parametrize("backend_type", [AirSimBackend, UnrealCvBackend])
@pytest.mark.parametrize("modality,method", [("pose", "get_pose"), ("rgb", "get_frames")])
def test_failed_sensor_is_not_an_observation(backend_type, modality, method):
    adapter = Mock()
    getattr(adapter, method).return_value = PrimitiveResult(False, "original sensor error")
    backend = backend_type(uav=adapter)
    with pytest.raises(RuntimeError, match="original sensor error"):
        backend.get_observation(SimpleNamespace(scene_instance_id="fixture"), "uav", [modality])


@pytest.mark.parametrize("protocol,probe", [("airsim", probe_airsim_endpoint), ("unrealcv", probe_unrealcv_endpoint)])
@pytest.mark.parametrize("ready", [True, False])
def test_probe_cleanup_does_not_replace_connectivity(monkeypatch, protocol, probe, ready):
    client = Mock()
    client.ping.return_value = client.isconnected.return_value = ready
    client.close.side_effect = client.disconnect.side_effect = RuntimeError("close failure")
    monkeypatch.setitem(sys.modules, protocol, SimpleNamespace(VehicleClient=lambda **kw: client, Client=lambda endpoint: client))
    actual, message = probe("localhost", 1)
    assert actual is ready
    assert "close failure" in message


def test_unrealcv_preserves_connection_and_cleanup_errors(monkeypatch):
    client = Mock()
    original = ConnectionError("primary connect failure")
    client.connect.side_effect = original
    client.disconnect.side_effect = RuntimeError("secondary close failure")
    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace(Client=lambda endpoint: client))
    adapter = UnrealCvUavAdapter("localhost", 1, {})
    with pytest.raises(ConnectionError) as captured:
        adapter.connect()
    assert captured.value is original
    assert captured.value.cleanup_errors[0]["error"] == "secondary close failure"
    assert adapter.client is None


def test_unrealcv_prepare_and_disconnect_failures_preserve_original(monkeypatch):
    import src.adapters.unrealcv.uav as module
    client = Mock()
    original = RuntimeError("prepare failed")
    client.isconnected.return_value = True
    client.disconnect.side_effect = RuntimeError("disconnect failed")
    control = Mock()
    control.prepare.side_effect = original
    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace(Client=lambda endpoint: client))
    monkeypatch.setattr(module, "create_pose_control", lambda *args: control)
    with pytest.raises(RuntimeError) as captured:
        UnrealCvUavAdapter("localhost", 1, {}).connect()
    assert captured.value is original
    assert captured.value.cleanup_errors[0]["error"] == "disconnect failed"
