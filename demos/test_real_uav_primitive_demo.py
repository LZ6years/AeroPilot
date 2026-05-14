"""Manual real-engine UAV primitive smoke demo.

This script is intentionally separate from pytest. It launches or connects to a
configured AirSim scene, runs lifecycle control, performs one bounded vertical
move, records before/after observations, then hovers/lands.

Recommended PowerShell:

    conda activate flyagent
    python test_real_uav_primitive_demo.py --map-id cityenviron_win64 --launch --up-meters 5

Equivalent without persistent shell activation:

    conda run -n flyagent python test_real_uav_primitive_demo.py --map-id cityenviron_win64 --launch --up-meters 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.sim_server.server import LocalSimulationServer
from src.tools.dispatcher import ToolDispatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real AirSim UAV primitive demo")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--map-id", default="cityenviron_win64")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--up-meters", type=float, default=5.0)
    parser.add_argument("--launch", action="store_true", help="Launch the configured scene process")
    parser.add_argument("--connect-existing", action="store_true", help="Connect to an already-running AirSim scene")
    parser.add_argument("--no-land", action="store_true", help="Skip landing before scene shutdown")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    maps, port_policy = load_map_config(args.config)
    action_config = load_uav_action_config(args.action_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == args.map_id)
    run_dir = RunStore(args.run_root).create_run_dir("real_uav_up")
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer(
        maps,
        port_policy,
        Path(run_dir),
        launch_processes=args.launch,
        real_control=args.launch or args.connect_existing,
        action_config=action_config,
    )
    client = LocalEnvClient(server)
    tools = ToolDispatcher(client)
    scene = None
    agent_id = "uav_1"
    try:
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        recorder.write_json("config/uav_action_config.json", action_config.to_dict())
        scene = client.start_scene(args.map_id)
        recorder.write_json("scene_instance.json", scene.to_dict())
        event(recorder, "scene_started", scene.to_dict())

        for tool, tool_args in [
            ("uav.connect", {}),
            ("uav.takeoff", {}),
            ("uav.hover", {}),
        ]:
            call_tool(tools, recorder, scene.scene_instance_id, agent_id, tool, tool_args)

        before = observe(tools, recorder, scene.scene_instance_id, agent_id, "before_up", ["pose"])

        duration = abs(args.up_meters) / action_config.vertical_velocity_mps
        move_args = {"direction": "up", "duration": duration}
        call_tool(
            tools,
            recorder,
            scene.scene_instance_id,
            agent_id,
            "uav.move_body",
            move_args,
            extra={"expected_up_meters": args.up_meters},
        )
        call_tool(tools, recorder, scene.scene_instance_id, agent_id, "uav.hover", {})

        after = observe(tools, recorder, scene.scene_instance_id, agent_id, "after_up", ["pose", "rgb", "depth"])
        if not args.no_land:
            call_tool(tools, recorder, scene.scene_instance_id, agent_id, "uav.land", {})

        summary = build_summary(run_dir, args.map_id, args.up_meters, before.to_dict(), after.to_dict())
        recorder.write_json("real_uav_up_summary.json", summary)
        event(recorder, "real_uav_up_summary", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        if scene is not None:
            client.stop_scene(scene.scene_instance_id)
            event(recorder, "scene_stopped", scene.to_dict())
        recorder.close()


def call_tool(
    tools: ToolDispatcher,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    tool: str,
    args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {"tool": tool, "args": args}
    if extra:
        payload.update(extra)
    event(recorder, "action_started", payload)
    result = tools.call(scene_instance_id, agent_id, tool, args)
    result_data = result.to_dict()
    event(recorder, "action_completed", result_data)
    if not result.ok:
        raise RuntimeError(f"{tool} failed: {result.message}")


def observe(
    tools: ToolDispatcher,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    label: str,
    modalities: list[str],
):
    observation = tools.observe(scene_instance_id, agent_id, modalities)
    payload = observation.to_dict()
    recorder.write_json(f"artifacts/observations/{label}.json", payload)
    event(recorder, f"observation_{label}", payload)
    return observation


def build_summary(
    run_dir: Path,
    map_id: str,
    expected_up_meters: float,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_pos = before.get("data", {}).get("pose", {}).get("position", {})
    after_pos = after.get("data", {}).get("pose", {}).get("position", {})
    delta_z = None
    if "z" in before_pos and "z" in after_pos:
        delta_z = after_pos["z"] - before_pos["z"]
    return {
        "run_dir": str(run_dir),
        "map_id": map_id,
        "expected_up_meters": expected_up_meters,
        "before_position": before_pos,
        "after_position": after_pos,
        "delta_z_ned": delta_z,
        "interpretation": "AirSim uses NED coordinates; upward movement usually makes z smaller/more negative.",
    }


def event(recorder: AsyncRunLogger, name: str, payload: dict[str, Any]) -> None:
    recorder.event(name, payload)
    print(name, json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
