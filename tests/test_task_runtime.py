"""Test M4 task runtime dry-run behavior."""

from __future__ import annotations

import json

from src.agent.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


def make_package() -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )


def test_task_runtime_completes_dry_run_and_writes_records(tmp_path):
    run_dir = tmp_path / "task_run"
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([make_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"))

    state = runtime.run_task("m1", "前面右转会有一个黑色的路灯，右转面向路灯后停下")
    recorder.close()

    assert state.completed
    assert state.connected
    assert state.airborne
    assert state.recent_observations
    assert (run_dir / "semantic_plan.json").exists()
    assert (run_dir / "mission_state.json").exists()
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    event_records = [json.loads(line) for line in events]
    event_names = [record["event"] for record in event_records]
    short_horizon_tools = [
        step.get("tool")
        for record in event_records
        if record["event"] == "short_horizon_planned"
        for step in record["payload"]["steps"]
        if step["kind"] == "tool"
    ]
    mission_state = json.loads((run_dir / "mission_state.json").read_text(encoding="utf-8"))

    assert "control_connected" in event_names
    assert "preflight_takeoff_started" in event_names
    assert "preflight_takeoff_completed" in event_names
    assert "preflight_hover_completed" in event_names
    assert "initial_observation_updated" in event_names
    assert "short_horizon_planned" in event_names
    assert "step_completed" in event_names
    assert "observation_updated" in event_names
    assert "uav.connect" not in short_horizon_tools
    assert "uav.takeoff" not in short_horizon_tools
    assert mission_state["connected"] is True
    assert mission_state["airborne"] is True
    assert mission_state["recent_observations"]
