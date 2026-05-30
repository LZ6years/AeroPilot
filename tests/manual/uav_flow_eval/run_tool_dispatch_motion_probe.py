"""Probe UAV-Flow UnrealCV motion through the normal ToolDispatcher path.

This manual diagnostic intentionally avoids planner/model calls. It starts one
UAV-Flow case, routes primitive commands through ``ToolDispatcher``, records
pose before/after each command, and compares measured local-meter displacement
with the shared UAV action scale config.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.cases import load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.tools.dispatcher import ToolDispatcher
from src.uav import body_velocity_to_world_delta, move_body_velocity_from_args, pose_delta


def run_probe(
    config: UavFlowEvalConfig,
    *,
    class_name: str = "Move",
    output_root: str | Path = "tests/manual_outputs/uav_flow_tool_dispatch_motion_probe",
    real_unrealcv: bool = True,
) -> dict[str, Any]:
    """Run a fixed primitive sequence through ToolDispatcher and save diagnostics."""

    cases = load_cases(
        config.uav_flow_repo_path,
        classes=[class_name],
        limit_per_class=1,
        instruction_field=config.instruction_field,
        test_jsons_path=config.test_jsons_path,
        classified_instr_path=config.classified_instr_path,
    )
    if not cases:
        raise ValueError(f"No UAV-Flow case available for class {class_name!r}")

    case = cases[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    case_dir = Path(output_root) / f"{case.case_id}_{stamp}"
    case_dir.mkdir(parents=True, exist_ok=True)
    action_config = _benchmark_action_config(config)
    client = UavFlowBenchmarkEnvClient(
        case,
        run_dir=case_dir,
        env_id=config.env_id,
        uav_flow_repo_path=config.uav_flow_repo_path,
        real_unrealcv=real_unrealcv,
        time_dilation=config.time_dilation,
        resolution=config.resolution,
        seed=config.seed,
        action_config=action_config,
    )
    tools = ToolDispatcher(client)
    scene = client.start_scene("uav_flow_downtownwest", case.case_id)
    scene_id = scene.scene_instance_id
    agent_id = "uav_flow_drone"

    try:
        steps = [
            ("uav.move_body", {"direction": "forward", "duration": 0.25}),
            ("uav.move_body", {"direction": "forward", "duration": 1.0}),
            ("uav.turn_relative", {"yaw_deg": 90.0}),
            ("uav.move_body", {"direction": "forward", "duration": 1.0}),
            ("uav.move_body", {"direction": "right", "duration": 1.0}),
        ]
        records: list[dict[str, Any]] = []
        initial = tools.observe(scene_id, agent_id, ["rgb", "pose"]).to_dict()
        for index, (tool_name, args) in enumerate(steps, start=1):
            before = tools.observe(scene_id, agent_id, ["pose"]).to_dict()["data"]["pose"]
            expected = _expected_delta(tool_name, args, before, action_config)
            result = tools.call(scene_id, agent_id, tool_name, args).to_dict()
            after = tools.observe(scene_id, agent_id, ["rgb", "pose"]).to_dict()
            after_pose = after["data"]["pose"]
            records.append(
                {
                    "index": index,
                    "tool": tool_name,
                    "args": args,
                    "before_pose": before,
                    "after_pose": after_pose,
                    "pose_delta": pose_delta(before, after_pose),
                    "expected_delta_m": expected,
                    "action_result": result,
                    "world_pose_cm": list(client.world_pose_cm),
                    "rgb": after.get("data", {}).get("frames", {}).get("rgb", {}),
                }
            )
    finally:
        client.stop_scene(scene_id)

    summary = {
        "case_id": case.case_id,
        "class_name": case.class_name,
        "instruction": case.selected_instruction,
        "real_unrealcv": real_unrealcv,
        "action_config": action_config.to_dict(),
        "scene": scene.to_dict(),
        "initial_observation": initial,
        "records": records,
    }
    (case_dir / "tool_dispatch_motion_probe.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "console.txt").write_text(_console(summary), encoding="utf-8")
    return {"case_dir": str(case_dir), "summary": summary}


def _expected_delta(
    tool_name: str,
    args: dict[str, Any],
    before_pose: dict[str, Any],
    action_config: Any,
) -> dict[str, float | None]:
    if tool_name == "uav.turn_relative":
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw_deg": float(args.get("yaw_deg", 0.0))}
    if tool_name != "uav.move_body":
        return {"dx": None, "dy": None, "dz": None, "dyaw_deg": None}
    vx, vy, vz, duration = move_body_velocity_from_args(args, action_config, vertical_sign=1.0)
    delta = body_velocity_to_world_delta(vx, vy, vz, duration, float(before_pose.get("yaw_deg", 0.0) or 0.0))
    return {
        "dx": round(delta["dx"], 3),
        "dy": round(delta["dy"], 3),
        "dz": round(delta["dz"], 3),
        "dyaw_deg": 0.0,
    }


def _benchmark_action_config(config: UavFlowEvalConfig) -> UavActionConfig:
    data = load_uav_action_config().to_dict()
    data.update(config.action_config_overrides)
    return UavActionConfig.from_dict(data)


def _console(summary: dict[str, Any]) -> str:
    lines = [
        "uav_flow_tool_dispatch_motion_probe: ok",
        f"case_id={summary.get('case_id')}",
        f"real_unrealcv={summary.get('real_unrealcv')}",
        f"action_config={summary.get('action_config')}",
    ]
    for record in summary.get("records", []):
        lines.append(
            "step={index} tool={tool} args={args} expected={expected} measured={measured}".format(
                index=record.get("index"),
                tool=record.get("tool"),
                args=record.get("args"),
                expected=record.get("expected_delta_m"),
                measured=record.get("pose_delta"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json",
    )
    parser.add_argument("--class", dest="class_name", default="Move")
    parser.add_argument("--dry-run", action="store_true", help="Use dry benchmark env instead of real UnrealCV.")
    parser.add_argument("--output-root", default="tests/manual_outputs/uav_flow_tool_dispatch_motion_probe")
    args = parser.parse_args()

    config = UavFlowEvalConfig.from_file(args.config)
    result = run_probe(
        config,
        class_name=args.class_name,
        output_root=args.output_root,
        real_unrealcv=not args.dry_run,
    )
    print(json.dumps({"case_dir": result["case_dir"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
