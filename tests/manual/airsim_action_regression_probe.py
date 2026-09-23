"""Measure real primitives through managed scenes; this is not a benchmark score.

Example: python tests/manual/airsim_action_regression_probe.py --config MAP.json
--map-id MAP --pose X Y Z YAW PITCH ROLL --turn -79 --velocity 2 0.5 0
--duration 1.5. JSON retains failures and cleanup evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs.map_config import load_map_config
from src.sim_server import SimulationExecutionMode
from src.sim_server.server import SimulationServer
from src.uav.frames import position_distance_m
from src.perception.frame_payload import load_rgb_frame, load_depth_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--pose", nargs=6, type=float, required=True)
    parser.add_argument("--turn", type=float, required=True)
    parser.add_argument("--velocity", nargs=3, type=float, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="tests/manual_outputs/airsim_action_regression")
    args = parser.parse_args()
    output = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    maps, policy = load_map_config(args.config)
    service = SimulationServer(maps, policy, run_dir=output,
                               execution_mode=SimulationExecutionMode.MANAGED,
                               save_artifacts=args.save_artifacts)
    client = SimulatorEnvClient(service)
    scene = None
    report = {"diagnostic_only": True, "map_id": args.map_id, "actions": []}
    try:
        scene = client.start_scene(args.map_id)
        report["scene"] = scene.to_dict()
        def action(name, arguments):
            result = client.execute_action(scene.scene_instance_id, "Drone_1", name, arguments)
            report["actions"].append({"action": name, "args": arguments, **result.to_dict()})
            return result
        connected = action("connect", {})
        if not connected.ok:
            raise RuntimeError(connected.message)
        pose = dict(zip(("x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"), args.pose))
        placed = action("set_pose", pose)
        if not placed.ok:
            raise RuntimeError(placed.message)
        measured = action("get_pose", {}).data
        report["initial_position_error_m"] = position_distance_m(pose, measured)
        report["first_observation"] = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]).to_dict()
        data = report["first_observation"]["data"]
        frames = data["frames"].get("frames", data["frames"])
        rgb = load_rgb_frame(frames["rgb"])
        depth = load_depth_frame(frames["depth"], require_metric=True)
        import numpy as np
        report["capture_validation"] = {
            "save_artifacts": args.save_artifacts,
            "rgb_size": list(rgb.size), "depth_shape": list(depth.shape),
            "rgb_nonconstant": bool(np.ptp(np.asarray(rgb)) > 0),
            "depth_finite_count": int(np.isfinite(depth).sum()),
            "depth_nonconstant": bool(np.ptp(depth[np.isfinite(depth)]) > 0),
        }
        if not report["capture_validation"]["rgb_nonconstant"] or not report["capture_validation"]["depth_nonconstant"]:
            raise RuntimeError("Degenerate image payload")
        if not args.save_artifacts and list((output / "artifacts").rglob("*")):
            raise RuntimeError("Memory-only capture created backend artifacts")
        action("turn_relative", {"yaw_deg": args.turn})
        action("move_body", dict(zip(("vx", "vy", "vz"), args.velocity)) | {"duration": args.duration})
        report["final_observation"] = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]).to_dict()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if scene is not None:
            try:
                client.stop_scene(scene.scene_instance_id)
                report["cleanup"] = {"ok": True}
            except Exception as exc:
                report["cleanup"] = {"ok": False, "error": str(exc)}
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(path)
    failed_action = any(not row["ok"] for row in report["actions"])
    return 1 if failed_action or report.get("error") or not report.get("cleanup", {}).get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
