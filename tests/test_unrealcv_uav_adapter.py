"""Verify the primitive boundary, SDK cleanup, and Backend delegation."""

import ast
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.adapters.unrealcv.uav import UnrealCvUavAdapter
from src.adapters.unrealcv.client import probe_unrealcv_endpoint
from src.adapters.uav_primitives import PrimitiveResult
from tests.support.fake_uav import FakeUavPrimitiveAdapter
from src.sim_server.backends.unrealcv_backend import UnrealCvBackend
from tests.test_unrealcv_control_and_capture import FakeUnrealCvClient, _package


class ConnectedClient(FakeUnrealCvClient):
    """Add transport lifetime to the existing command-level fake."""

    def __init__(self):
        super().__init__()
        self.actor_rotation = [0.0, 0.0, 0.0]
        self.connected = False
        self.closed = 0

    def connect(self):
        self.connected = True

    def isconnected(self):
        return self.connected

    def disconnect(self):
        self.closed += 1
        self.connected = False


def test_adapter_actor_primitives_capture_and_reconnect(monkeypatch, tmp_path):
    clients = []
    def factory(endpoint):
        client = ConnectedClient()
        clients.append(client)
        return client
    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace(Client=factory))
    adapter = UnrealCvUavAdapter("localhost", 9010, _package("actor").unrealcv,
                                artifact_dir=tmp_path, save_artifacts=True)
    assert adapter.connect().ok
    assert adapter.set_pose(1, 2, 3, 170).ok
    assert clients[0].actor_location == [100, 200, 300]
    assert adapter.turn_relative(30).ok
    assert adapter.get_pose().data["yaw_deg"] == pytest.approx(-160)
    assert adapter.move_body(1, 0, 0, 2).ok
    assert adapter.get_pose().data["x"] < 1
    frames = adapter.get_frames(["rgb", "depth"]).data["frames"]
    assert set(frames) == {"rgb", "depth"}
    assert not adapter.land().ok
    adapter.close()
    adapter.close()
    assert clients[0].closed == 1
    assert adapter.get_pose().ok
    assert len(clients) == 2
    adapter.close()


def test_prepare_failure_closes_temporary_connection(monkeypatch):
    client = ConnectedClient()
    def reject(*args, **kwargs):
        raise RuntimeError("prepare command failed")
    client.request = reject
    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace(Client=lambda endpoint: client))
    adapter = UnrealCvUavAdapter("localhost", 9010, _package("actor").unrealcv)
    with pytest.raises(RuntimeError):
        adapter.connect()
    assert client.closed == 1
    assert adapter.client is None


def test_read_only_probe_always_disconnects(monkeypatch):
    client = ConnectedClient()
    monkeypatch.setitem(sys.modules, "unrealcv", SimpleNamespace(Client=lambda endpoint: client))
    assert probe_unrealcv_endpoint("localhost", 9010)[0]
    assert client.commands == []
    assert client.closed == 1


def test_backend_preserves_primitive_failure():
    class FailingAdapter(FakeUavPrimitiveAdapter):
        def turn_relative(self, yaw_deg):
            return PrimitiveResult(False, "blocked", {"failure_kind": "control_no_effect"})
    backend = UnrealCvBackend(uav=FailingAdapter())
    result = backend.execute_action(None, "drone", "turn_relative", {"yaw_deg": 10})
    assert not result.ok
    assert result.data["failure_kind"] == "control_no_effect"


@pytest.mark.parametrize("name", ["airsim_backend", "unrealcv_backend"])
def test_backend_has_no_external_sdk_import(name):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/sim_server/backends" / f"{name}.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not {"airsim", "unrealcv"}.intersection(imported)
