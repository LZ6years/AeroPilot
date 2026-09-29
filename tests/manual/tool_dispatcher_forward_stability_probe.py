"""Manual real-engine ToolDispatcher stability probe for low-level UAV motion.

This manual test intentionally bypasses TaskParser, ModelActionPlanner, memory
modules, and SafetySupervisor. It calls ToolDispatcher directly so we can
inspect whether a motion primitive itself leaves the UAV/camera unstable.

The probe runs several cases such as:

    move_body(forward) -> hover -> poll pose velocity/pitch/roll/yaw
    move_body(left)    -> hover -> poll pose velocity/pitch/roll/yaw
    turn_relative      -> hover -> poll pose velocity/pitch/roll/yaw

It can also run a forward speed sweep where each case targets the same distance
with a different body-frame vx/duration pair. That makes it easier to separate
"the simulator cannot brake after fast forward motion" from higher-level model
planning issues.

The brake strategy comparison mode keeps the same forward command and compares
hover-only, brake-only, and brake-then-hover behavior. Each case starts from a
stabilized pose so the comparison focuses on the post-forward control strategy.

Each poll sample is written to JSON/Markdown so the result can be compared with
what is visible in the simulator window.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_client.simulator_client import SimulatorEnvClient
from src.uav.config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.runtime.configs.stabilization_config import load_uav_stabilization_config
from src.runtime.control import UavStabilizer
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.sim_server.server import SimulationServer
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from src.uav import pose_from_observation, pose_measurements, velocity_norm_mps, yaw_deg
from src.agent.planning.schemas import PlannedStep


DEFAULT_CASES = [
    {"name": "forward_2_5s", "tool": "uav.move_body", "args": {"direction": "forward", "duration": 2.5}},
    {"name": "left_1_0s", "tool": "uav.move_body", "args": {"direction": "left", "duration": 1.0}},
    {"name": "turn_right_40deg", "tool": "uav.turn_relative", "args": {"yaw_deg": 40.0}},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe UAV motion stability through ToolDispatcher")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--stabilization-config", default="configs/uav_stabilization_config.json")
    parser.add_argument("--map-id", default="cityenviron_win64")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--agent-id", default="uav_1")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat the case list this many times.")
    parser.add_argument("--poll-seconds", type=float, default=6.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--speed-sweep",
        action="store_true",
        help="Run forward vx/duration cases that target the same distance.",
    )
    parser.add_argument(
        "--brake-strategy-compare",
        action="store_true",
        help="Compare hover-only, brake-only, and brake-then-hover after the same forward command.",
    )
    parser.add_argument("--target-distance-m", type=float, default=5.0)
    parser.add_argument(
        "--forward-speeds-mps",
        default="2.0,1.5,1.0,0.75,0.5",
        help="Comma-separated forward body-frame speeds for --speed-sweep.",
    )
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
    action_config = load_uav_action_config(args.action_config)
    stabilization_config = load_uav_stabilization_config(args.stabilization_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == args.map_id)
    run_dir = RunStore(args.run_root).create_run_dir("tool_dispatcher_forward_stability")
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
    stabilizer = UavStabilizer(tools, recorder, stabilization_config)
    scene = client.start_scene(args.map_id)
    base_cases = select_cases(args)
    cases = expand_cases(base_cases, args.repeat)
    rows: list[dict[str, Any]] = []

    try:
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        recorder.write_json("config/uav_action_config.json", action_config.to_dict())
        recorder.write_json("config/uav_stabilization_config.json", stabilization_config.to_dict())
        recorder.write_json("state/scene_instance.json", scene.to_dict())
        recorder.event("scene_started", scene.to_dict())

        for tool, tool_args in [("uav.connect", {}), ("uav.takeoff", {}), ("uav.hover", {})]:
            call_tool(tools, recorder, scene.scene_instance_id, args.agent_id, tool, tool_args)

        baseline = observe_pose(tools, scene.scene_instance_id, args.agent_id)
        recorder.debug_event("baseline_pose", baseline)

        for index, case in enumerate(cases, start=1):
            row = run_case(index, case, tools, stabilizer, recorder, scene.scene_instance_id, args.agent_id, args, baseline)
            rows.append(row)
            recorder.event("stability_case_completed", case_summary(row))

        call_tool(tools, recorder, scene.scene_instance_id, args.agent_id, "uav.hover", {})
    finally:
        client.stop_scene(scene.scene_instance_id)
        recorder.event("scene_stopped", scene.to_dict())
        recorder.close()

    report = {
        "run_dir": str(run_dir),
        "map_id": args.map_id,
        "action_config": action_config.to_dict(),
        "stabilization_config": stabilization_config.to_dict(),
        "poll_seconds": args.poll_seconds,
        "poll_interval_sec": args.poll_interval_sec,
        "speed_sweep": args.speed_sweep,
        "brake_strategy_compare": args.brake_strategy_compare,
        "target_distance_m": args.target_distance_m if args.speed_sweep else None,
        "cases": rows,
        "summary": summarize(rows),
    }
    write_json(run_dir / "demo" / "stability_report.json", report)
    (run_dir / "demo" / "stability_report.md").write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "summary": report["summary"]}, indent=2, ensure_ascii=False))


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Select one manual probe scenario; keep modes mutually exclusive."""

    enabled_modes = [bool(args.speed_sweep), bool(args.brake_strategy_compare)]
    if sum(enabled_modes) > 1:
        raise ValueError("Choose only one of --speed-sweep or --brake-strategy-compare.")
    if args.speed_sweep:
        return build_speed_sweep_cases(args)
    if args.brake_strategy_compare:
        return build_brake_strategy_cases(args)
    return DEFAULT_CASES


def expand_cases(cases: list[dict[str, Any]], repeat: int) -> list[dict[str, Any]]:
    expanded = []
    for iteration in range(max(1, repeat)):
        for case in cases:
            expanded.append({**case, "iteration": iteration + 1})
    return expanded


def build_speed_sweep_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build same-distance forward cases with different vx/duration pairs."""

    speeds = [float(item.strip()) for item in args.forward_speeds_mps.split(",") if item.strip()]
    cases: list[dict[str, Any]] = []
    for speed in speeds:
        if speed <= 0:
            raise ValueError(f"forward speed must be positive: {speed}")
        duration = float(args.target_distance_m) / speed
        speed_label = str(speed).replace(".", "_")
        cases.append(
            {
                "name": f"forward_{args.target_distance_m:g}m_vx_{speed_label}",
                "tool": "uav.move_body",
                "args": {"vx": speed, "vy": 0.0, "vz": 0.0, "duration": round(duration, 3)},
                "expected_forward_m": float(args.target_distance_m),
                "commanded_speed_mps": speed,
            }
        )
    return cases


def build_brake_strategy_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build comparable post-forward stabilization strategies."""

    forward_args = {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 5.0}
    return [
        {
            "name": "hover_only",
            "tool": "uav.move_body",
            "args": dict(forward_args),
            "strategy": "hover_only",
            "post_actions": [{"tool": "uav.hover", "args": {}}],
        },
        {
            "name": "brake_only_1s",
            "tool": "uav.move_body",
            "args": dict(forward_args),
            "strategy": "brake_only",
            "post_actions": [{"tool": "uav.move_body", "args": zero_velocity_args(1.0)}],
        },
        {
            "name": "brake_only_2s",
            "tool": "uav.move_body",
            "args": dict(forward_args),
            "strategy": "brake_only",
            "post_actions": [{"tool": "uav.move_body", "args": zero_velocity_args(2.0)}],
        },
        {
            "name": "brake_then_hover_2s",
            "tool": "uav.move_body",
            "args": dict(forward_args),
            "strategy": "brake_then_hover",
            "post_actions": [
                {"tool": "uav.move_body", "args": zero_velocity_args(2.0)},
                {"tool": "uav.hover", "args": {}},
            ],
        },
    ]


def run_case(
    index: int,
    case: dict[str, Any],
    tools: ToolDispatcher,
    stabilizer: UavStabilizer,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    args: argparse.Namespace,
    baseline_pose: dict[str, Any],
) -> dict[str, Any]:
    if case.get("post_actions"):
        return run_brake_strategy_case(index, case, tools, recorder, scene_instance_id, agent_id, args, baseline_pose)

    before = observe_pose(tools, scene_instance_id, agent_id)
    action_result = call_tool(tools, recorder, scene_instance_id, agent_id, case["tool"], case["args"])
    stabilization = stabilizer.wait_after_step(
        scene_instance_id,
        agent_id,
        case["name"],
        PlannedStep.tool_step(case["tool"], dict(case["args"]), "tool dispatcher stability probe"),
    )
    after_stabilization = observe_pose(tools, scene_instance_id, agent_id)
    samples = poll_pose_series(tools, scene_instance_id, agent_id, args.poll_seconds, args.poll_interval_sec)
    after_poll = samples[-1]["pose"] if samples else after_stabilization
    row = {
        "index": index,
        "case": case,
        "before_pose": before,
        "action_result": action_result,
        "stabilization": stabilization,
        "after_stabilization_pose": after_stabilization,
        "poll_samples": samples,
        "after_poll_pose": after_poll,
        "movement_before_to_stabilized": pose_measurements(before, after_stabilization),
        "movement_stabilized_to_after_poll": pose_measurements(after_stabilization, after_poll),
        "movement_before_to_after_poll": pose_measurements(before, after_poll),
        "stability_stats": stability_stats(samples),
    }
    recorder.write_json(f"debug/stability_cases/case_{index:03d}_{safe_name(case['name'])}.json", row)
    return row


def run_brake_strategy_case(
    index: int,
    case: dict[str, Any],
    tools: ToolDispatcher,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    args: argparse.Namespace,
    baseline_pose: dict[str, Any],
) -> dict[str, Any]:
    """Run one forward command followed by explicit post-forward actions."""

    prepare_case_start(tools, recorder, scene_instance_id, agent_id, baseline_pose)
    before = observe_pose(tools, scene_instance_id, agent_id)
    action_result = call_tool(tools, recorder, scene_instance_id, agent_id, case["tool"], case["args"])
    after_forward = observe_pose(tools, scene_instance_id, agent_id)
    post_results = []
    for post_action in case["post_actions"]:
        post_results.append(
            {
                "tool": post_action["tool"],
                "args": post_action["args"],
                "result": call_tool(
                    tools,
                    recorder,
                    scene_instance_id,
                    agent_id,
                    post_action["tool"],
                    post_action["args"],
                ),
            }
        )
    after_stabilization = observe_pose(tools, scene_instance_id, agent_id)
    samples = poll_pose_series(tools, scene_instance_id, agent_id, args.poll_seconds, args.poll_interval_sec)
    after_poll = samples[-1]["pose"] if samples else after_stabilization
    row = {
        "index": index,
        "case": case,
        "before_pose": before,
        "action_result": action_result,
        "after_forward_pose": after_forward,
        "post_action_results": post_results,
        "stabilization": {
            "status": "manual_strategy",
            "strategy": case.get("strategy"),
            "post_actions": case.get("post_actions", []),
        },
        "after_stabilization_pose": after_stabilization,
        "poll_samples": samples,
        "after_poll_pose": after_poll,
        "movement_before_to_after_forward": pose_measurements(before, after_forward),
        "movement_before_to_stabilized": pose_measurements(before, after_stabilization),
        "movement_stabilized_to_after_poll": pose_measurements(after_stabilization, after_poll),
        "movement_before_to_after_poll": pose_measurements(before, after_poll),
        "stability_stats": stability_stats(samples),
    }
    recorder.write_json(f"debug/stability_cases/case_{index:03d}_{safe_name(case['name'])}.json", row)
    return row


def prepare_case_start(
    tools: ToolDispatcher,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    baseline_pose: dict[str, Any],
) -> None:
    """Start each comparison case from a known low-motion state."""

    position = baseline_pose.get("position") or {}
    call_tool(
        tools,
        recorder,
        scene_instance_id,
        agent_id,
        "uav.set_pose",
        {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", -2.0)),
            "yaw_deg": yaw_deg(baseline_pose),
        },
    )
    call_tool(tools, recorder, scene_instance_id, agent_id, "uav.move_body", zero_velocity_args(2.0))
    call_tool(tools, recorder, scene_instance_id, agent_id, "uav.hover", {})
    time.sleep(0.5)


def call_tool(
    tools: ToolDispatcher,
    recorder: AsyncRunLogger,
    scene_instance_id: str,
    agent_id: str,
    tool: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    recorder.debug_event("tool_started", {"tool": tool, "args": args})
    result = tools.call(scene_instance_id, agent_id, tool, args)
    payload = result.to_dict()
    recorder.debug_event("tool_completed", {"tool": tool, "args": args, "result": payload})
    if not result.ok:
        raise RuntimeError(f"{tool} failed: {result.message}")
    return payload


def observe_pose(tools: ToolDispatcher, scene_instance_id: str, agent_id: str) -> dict[str, Any]:
    observation = tools.observe(scene_instance_id, agent_id, ["pose"])
    return pose_from_observation(observation.to_dict().get("data", {}))


def poll_pose_series(
    tools: ToolDispatcher,
    scene_instance_id: str,
    agent_id: str,
    poll_seconds: float,
    poll_interval_sec: float,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    samples = []
    while time.monotonic() - started <= poll_seconds:
        pose = observe_pose(tools, scene_instance_id, agent_id)
        samples.append(
            {
                "elapsed_sec": round(time.monotonic() - started, 3),
                "pose": pose,
                "velocity_norm_mps": velocity_norm_mps(pose),
                "attitude_deg": attitude_deg(pose),
                "yaw_deg": round(yaw_deg(pose), 3),
            }
        )
        time.sleep(max(0.0, poll_interval_sec))
    return samples


def attitude_deg(pose: dict[str, Any]) -> dict[str, float | None]:
    euler = pose.get("euler_rad") if isinstance(pose.get("euler_rad"), dict) else {}
    return {
        "pitch": round(math.degrees(float(euler.get("pitch", 0.0))), 3) if euler else None,
        "roll": round(math.degrees(float(euler.get("roll", 0.0))), 3) if euler else None,
    }


def stability_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    velocities = [float(item["velocity_norm_mps"]) for item in samples if item.get("velocity_norm_mps") is not None]
    pitches = [float(item["attitude_deg"]["pitch"]) for item in samples if item.get("attitude_deg", {}).get("pitch") is not None]
    rolls = [float(item["attitude_deg"]["roll"]) for item in samples if item.get("attitude_deg", {}).get("roll") is not None]
    return {
        "sample_count": len(samples),
        "velocity_first": round(velocities[0], 4) if velocities else None,
        "velocity_last": round(velocities[-1], 4) if velocities else None,
        "velocity_max": round(max(velocities), 4) if velocities else None,
        "pitch_range_deg": round(max(pitches) - min(pitches), 4) if pitches else None,
        "roll_range_deg": round(max(rolls) - min(rolls), 4) if rolls else None,
    }


def case_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row["index"],
        "name": row["case"]["name"],
        "tool": row["case"]["tool"],
        "args": row["case"]["args"],
        "stabilization": row["stabilization"],
        "stability_stats": row["stability_stats"],
        "movement_stabilized_to_after_poll": row["movement_stabilized_to_after_poll"],
        "movement_before_to_after_poll": row["movement_before_to_after_poll"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "timeouts": [case_summary(row) for row in rows if row["stabilization"].get("status") == "timeout"],
        "completed": [case_summary(row) for row in rows if row["stabilization"].get("status") == "completed"],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ToolDispatcher Forward Stability Report",
        "",
        f"- Run: `{report['run_dir']}`",
        f"- Map: `{report['map_id']}`",
        f"- Stabilization: `{report['stabilization_config']}`",
        "",
        "| case | tool | args | wait status | wait velocity | poll last velocity | stabilized forward | final forward | drift after wait | pitch range | roll range |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in report["cases"]:
        stats = row["stability_stats"]
        drift = row["movement_stabilized_to_after_poll"]
        stabilized = row["movement_before_to_stabilized"]
        final = row["movement_before_to_after_poll"]
        lines.append(
            f"| {row['case']['name']}#{row['case'].get('iteration')} | {row['case']['tool']} | "
            f"`{row['case']['args']}` | {row['stabilization'].get('status')} | "
            f"{row['stabilization'].get('last_velocity_norm_mps')} | {stats.get('velocity_last')} | "
            f"{measurement_value(stabilized, 'forward_displacement_m')} | "
            f"{measurement_value(final, 'forward_displacement_m')} | "
            f"`{drift}` | {stats.get('pitch_range_deg')} | {stats.get('roll_range_deg')} |"
        )
    lines.append("")
    lines.append("If forward cases keep a high velocity or large attitude range after hover, the issue is below the model layer.")
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def measurement_value(measurements: dict[str, Any], key: str) -> float | None:
    value = measurements.get(key)
    return round(float(value), 3) if value is not None else None


def zero_velocity_args(duration: float) -> dict[str, float]:
    return {"vx": 0.0, "vy": 0.0, "vz": 0.0, "duration": float(duration)}


if __name__ == "__main__":
    main()
