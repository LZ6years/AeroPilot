"""Run a local memory-system demo with two continuous instructions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.agent.semantic.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


LAMP_TASK = "前面右转会有一个黑色的路灯，右转面向路灯后停下"
ENTRANCE_TASK = "左转寻找建筑入口，并停在入口前"


def main() -> None:
    run_dir = Path("runs") / ("memory_system_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer([_map_package()], run_dir=run_dir)
    runtime = TaskRuntime(LocalEnvClient(server), recorder, TaskParser(mode="rule_based"), shutdown_policy="none")

    first_state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    second_state = runtime.run_task("m1", ENTRANCE_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    runtime.finish_reused_scene("m1", second_state)
    recorder.close()

    context = _read_json(run_dir / "contexts" / "latest_context.json")
    flight_memory = _read_json(run_dir / "flight_memory.json")
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = {
        "run_dir": str(run_dir),
        "first_completed": first_state.completed,
        "second_completed": second_state.completed,
        "event_counts": {
            "scene_started": _count(events, "scene_started"),
            "scene_reused": _count(events, "scene_reused"),
            "control_connected": _count(events, "control_connected"),
            "preflight_takeoff_completed": _count(events, "preflight_takeoff_completed"),
            "scene_stopped": _count(events, "scene_stopped"),
        },
        "active_instruction": context["active_instruction"],
        "fragment_names": [item["name"] for item in context["fragments"]],
        "instruction_history_refs": flight_memory["instruction_history"],
        "last_recent_pose_ref": flight_memory["recent_pose_refs"][-1] if flight_memory["recent_pose_refs"] else None,
        "recent_actions": _recent_actions(context),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _map_package() -> MapPackage:
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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(events: list[dict], event_name: str) -> int:
    return sum(1 for item in events if item["event"] == event_name)


def _recent_actions(context: dict) -> list[dict]:
    for fragment in context["fragments"]:
        if fragment["name"] == "action":
            return fragment["content"].get("recent_actions", [])
    return []


if __name__ == "__main__":
    main()
