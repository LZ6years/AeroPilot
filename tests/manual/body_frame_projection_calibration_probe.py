r"""Calibrate AirSim body-frame motion against model-facing projection fields.

This manual probe launches or connects to a real UE/AirSim scene, resets the UAV
to the same world position at several yaw angles, executes short body-frame
movement commands, and records whether the current phase-local projection math
matches the observed world pose delta.

Recommended PowerShell:

    $env:PYTHONIOENCODING='utf-8'
    F:\conda_env\flyagent\python.exe tests/manual/body_frame_projection_calibration_probe.py --env-path .env --save-artifacts

Inspect:

    runs/body_frame_calibration_<timestamp>/calibration_report.md
    runs/body_frame_calibration_<timestamp>/calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.uav import pose_measurements
from src.env_client.simulator_client import SimulatorEnvClient
from src.uav.config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.agent.config import load_model_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.sim_server.server import SimulationServer
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry


YAW_CASES = [0.0, 90.0, 180.0, -90.0]
MOVE_CASES = ["forward", "backward", "left", "right", "up", "down"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate body-frame projection against real AirSim motion")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--map-id", default="cityenviron_win64")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--z", type=float, default=-2.0)
    parser.add_argument(
        "--execution",
        choices=["attach", "managed"],
        required=True,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    maps, port_policy = load_map_config(args.config)
    load_model_config(args.env_path)  # Load .env consistently with the other real-engine demos.
    action_config = load_uav_action_config(args.action_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == args.map_id)
    run_dir = RunStore(args.run_root).create_run_dir("body_frame_calibration")
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        execution_mode=args.execution,
        show_display=not args.headless,
        action_config=action_config,
        save_artifacts=args.save_artifacts,
    )
    client = SimulatorEnvClient(server)
    tools = ToolDispatcher(client, default_registry())
    scene = client.start_scene(args.map_id)
    agent_id = "uav_1"
    results: list[dict[str, Any]] = []

    try:
        tools.call(scene.scene_instance_id, agent_id, "uav.connect", {})
        tools.call(scene.scene_instance_id, agent_id, "uav.takeoff", {})
        origin = _pose(tools.observe(scene.scene_instance_id, agent_id, ["pose"]).to_dict())
        origin_x = _position_value(origin, "x")
        origin_y = _position_value(origin, "y")

        for yaw_deg in YAW_CASES:
            for direction in MOVE_CASES:
                # Reset before every movement so each measurement is independent.
                tools.call(
                    scene.scene_instance_id,
                    agent_id,
                    "uav.set_pose",
                    {
                        "x": origin_x,
                        "y": origin_y,
                        "z": args.z,
                        "yaw_deg": yaw_deg,
                        "pitch_deg": 0.0,
                        "roll_deg": 0.0,
                        "ignore_collision": True,
                    },
                )
                tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
                before = _pose(tools.observe(scene.scene_instance_id, agent_id, ["pose"]).to_dict())
                action = {"direction": direction, "duration": args.duration}
                action_result = tools.call(scene.scene_instance_id, agent_id, "uav.move_body", action).to_dict()
                tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
                after = _pose(tools.observe(scene.scene_instance_id, agent_id, ["pose", "rgb", "depth"]).to_dict())
                measurement = pose_measurements(before, after)
                row = {
                    "yaw_deg": yaw_deg,
                    "direction": direction,
                    "action": action,
                    "action_result": action_result,
                    "before_pose": before,
                    "after_pose": after,
                    "measurement": measurement,
                    "expected_sign": expected_sign(direction),
                    "sign_check": sign_check(direction, measurement),
                }
                results.append(row)
                recorder.event("calibration_case_completed", row)
    finally:
        try:
            tools.call(scene.scene_instance_id, agent_id, "uav.hover", {})
        finally:
            client.stop_scene(scene.scene_instance_id)
            recorder.close()

    payload = {
        "map_id": selected_map.map_id,
        "duration": args.duration,
        "action_config": action_config.to_dict(),
        "cases": results,
        "summary": summarize(results),
    }
    (run_dir / "calibration_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "calibration_report.md").write_text(build_markdown(payload), encoding="utf-8")
    print(f"Calibration report: {run_dir / 'calibration_report.md'}")


def _pose(observation: dict[str, Any]) -> dict[str, Any]:
    data = observation.get("data", observation)
    pose = data.get("pose") if isinstance(data, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def _position_value(pose: dict[str, Any], key: str) -> float:
    position = pose.get("position", {}) if isinstance(pose.get("position", {}), dict) else {}
    return float(position.get(key, pose.get(key, 0.0)) or 0.0)


def expected_sign(direction: str) -> dict[str, str]:
    return {
        "forward": {"forward_displacement_m": "positive"},
        "backward": {"forward_displacement_m": "negative"},
        "right": {"right_displacement_m": "positive"},
        "left": {"right_displacement_m": "negative"},
        "up": {"up_displacement_m": "positive"},
        "down": {"up_displacement_m": "negative"},
    }[direction]


def sign_check(direction: str, measurement: dict[str, Any]) -> bool:
    field, sign = next(iter(expected_sign(direction).items()))
    value = float(measurement.get(field, 0.0) or 0.0)
    return value > 0.05 if sign == "positive" else value < -0.05


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["sign_check"])
    return {"total": total, "passed": passed, "failed": total - passed, "pass_rate": round(passed / total, 3) if total else 0.0}


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Body Frame Projection Calibration",
        "",
        f"- Map: `{payload['map_id']}`",
        f"- Duration: `{payload['duration']}`",
        f"- Summary: `{payload['summary']}`",
        "",
        "| yaw_deg | direction | forward | right | up | expected | pass |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for item in payload["cases"]:
        m = item["measurement"]
        lines.append(
            f"| {item['yaw_deg']} | {item['direction']} | {m.get('forward_displacement_m')} | "
            f"{m.get('right_displacement_m')} | {m.get('up_displacement_m')} | "
            f"`{item['expected_sign']}` | {item['sign_check']} |"
        )
    lines.append("")
    lines.append("If any sign check fails, fix `src.uav.motion.pose_measurements()` before trusting phase-local progress.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
