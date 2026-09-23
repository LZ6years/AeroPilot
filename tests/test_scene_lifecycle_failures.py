"""Fault-injection gates for owned resources; never contact an engine or SSH."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.adapters.airsim.client import close_airsim_client
from src.adapters.airsim.uav import AirSimUavAdapter
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.lifecycle import SceneLifecycleError
from src.sim_server.server import SimulationServer
from tests.test_remote_scene_control import (
    FakeRemoteControl, ManagedRemoteBackend, make_remote_package, make_template,
)


def make_server(tmp_path):
    template = tmp_path / "settings.json"
    make_template(template)
    package = make_remote_package(template)
    package.launch["post_launch_wait_sec"] = 0
    remote = FakeRemoteControl()
    backend = ManagedRemoteBackend(remote)
    registry = BackendRegistry()
    registry.register("airsim", lambda context: backend)
    server = SimulationServer([package], run_dir=tmp_path / "run",
                              remote_scene_control=remote, backend_registry=registry)
    return server, remote, backend, package


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_ready_timeout_rolls_back_owned_process(tmp_path, monkeypatch, cleanup_failure):
    server, remote, backend, package = make_server(tmp_path)
    monkeypatch.setattr(server, "_wait_for_backend", Mock(side_effect=TimeoutError("RPC deadline")))
    backend.stop = Mock(side_effect=RuntimeError("close failed") if cleanup_failure else None)
    expected = SceneLifecycleError if cleanup_failure else TimeoutError
    with pytest.raises(expected, match="RPC deadline") as captured:
        server.start_scene(package.map_id)
    assert remote.starts == remote.stops
    assert len(remote.starts) == 1
    assert remote.tunnel.closed
    assert not server._reserved_ports
    backend.stop.assert_called_once()
    if cleanup_failure:
        assert isinstance(captured.value.original_error, TimeoutError)
        assert captured.value.cleanup_errors[0]["stage"] == "backend_closed"


def test_backend_start_and_stop_failures_do_not_skip_process_cleanup(tmp_path):
    server, remote, backend, package = make_server(tmp_path)
    backend.start = Mock(side_effect=RuntimeError("partial connect"))
    backend.stop = Mock(side_effect=RuntimeError("partial close"))
    with pytest.raises(SceneLifecycleError, match="partial connect"):
        server.start_scene(package.map_id)
    assert remote.starts == remote.stops
    assert remote.tunnel.closed


def test_shutdown_tries_tunnel_even_when_backend_and_worker_fail(tmp_path):
    server, remote, backend, package = make_server(tmp_path)
    scene = server.start_scene(package.map_id)
    backend.stop = Mock(side_effect=RuntimeError("backend close"))
    remote.stop = Mock(side_effect=RuntimeError("worker stop"))
    with pytest.raises(SceneLifecycleError) as captured:
        server.stop_scene(scene.scene_instance_id)
    assert len(captured.value.cleanup_errors) == 2
    assert remote.tunnel.closed
    assert not server._reserved_ports
    records = [json.loads(line) for line in next((tmp_path / "run/scenes").glob("*/lifecycle.jsonl")).read_text().splitlines()]
    assert any(r["stage"] == "tunnel_closed" and r["status"] == "ok" for r in records)


@pytest.mark.parametrize("outcome", [True, False, TimeoutError("ping"), RuntimeError("rpc")])
def test_probe_always_closes_nested_transport(monkeypatch, outcome):
    transport = SimpleNamespace(close=Mock())
    client = SimpleNamespace(client=transport, ping=Mock())
    if isinstance(outcome, Exception):
        client.ping.side_effect = outcome
    else:
        client.ping.return_value = outcome
    monkeypatch.setitem(sys.modules, "airsim", SimpleNamespace(VehicleClient=lambda **kwargs: client))
    result = AirSimBackend().probe(None, "unused", 1)
    assert result.ready is (outcome is True)
    transport.close.assert_called_once()


def test_close_prefers_outer_and_adapter_repeat_close_is_safe():
    inner = Mock()
    outer = SimpleNamespace(close=Mock(), client=inner)
    close_airsim_client(outer)
    outer.close.assert_called_once()
    inner.close.assert_not_called()
    adapter = AirSimUavAdapter()
    adapter._client = SimpleNamespace(client=inner)
    adapter.close()
    adapter.close()
    inner.close.assert_called_once()


def test_tunnel_large_stderr_uses_file_and_bounded_diagnostics(tmp_path):
    import subprocess
    from src.sim_server.remote.tunnel import SshTunnel
    log = (tmp_path / "ssh.log").open("w+b")
    process = subprocess.Popen([sys.executable, "-c",
        "import sys; sys.stderr.write('x' * 1000000); sys.stderr.flush()"], stderr=log)
    tunnel = SshTunnel(process, "127.0.0.1", 1, log)
    process.wait(timeout=10)
    with pytest.raises(RuntimeError) as captured:
        tunnel.check_alive()
    assert len(str(captured.value)) < 4200
    tunnel.close()
    tunnel.close()
    assert log.closed


def test_lifecycle_error_keeps_benchmark_original_type():
    from evaluation.benchmarks.core.executor import _error_payload
    error = SceneLifecycleError(TimeoutError("deadline"), [{"stage": "stop", "error": "bad"}])
    assert _error_payload(error) == {"error_type": "TimeoutError", "error": "deadline"}


def test_cleanup_log_failure_does_not_stop_remaining_releases(tmp_path):
    server, remote, backend, package = make_server(tmp_path)
    scene = server.start_scene(package.map_id)
    server.lifecycle[scene.scene_instance_id].record = Mock(side_effect=OSError("disk full"))
    with pytest.raises(SceneLifecycleError):
        server.stop_scene(scene.scene_instance_id)
    assert remote.starts == remote.stops
    assert remote.tunnel.closed
    assert not server._reserved_ports
