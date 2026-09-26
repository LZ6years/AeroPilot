"""Measure real primitives through managed scenes; this is not a benchmark score.

Example: python tests/manual/airsim_action_regression_probe.py --config MAP.json
--map-id MAP --pose X Y Z YAW PITCH ROLL --turn -79 --velocity 2 0.5 0
--duration 1.5. JSON retains failures and cleanup evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs.map_config import load_map_config
from src.sim_server import SimulationExecutionMode
from src.sim_server.server import SimulationServer
from src.uav.frames import position_distance_m
from src.perception.frame_payload import load_rgb_frame, load_depth_frame
from src.runtime.control.stabilization import UavStabilizer
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from src.agent.planning.schemas import PlannedStep
from src.env_client.errors import SceneRecoveryRequired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--pose", nargs=6, type=float)
    parser.add_argument("--turn", type=float, default=15)
    parser.add_argument("--velocity", nargs=3, type=float, default=[1, 0, 0])
    parser.add_argument("--duration", type=float, default=1)
    parser.add_argument("--case-id", help="Use exact TravelUAV dataset pose and target placement")
    parser.add_argument("--sequence-file", type=Path, help="Frozen list of {action,args}")
    parser.add_argument("--reuse-verified-post-action-state", action="store_true")
    parser.add_argument("--diagnostic-snapshots", action="store_true")
    parser.add_argument("--yaw-timeout", type=float, help="Diagnostic-only single-variable deadline experiment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", default="tests/manual_outputs/airsim_action_regression")
    args = parser.parse_args()
    output = args.output_dir or Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    maps, policy = load_map_config(args.config)
    service = SimulationServer(maps, policy, run_dir=output,
                               execution_mode=SimulationExecutionMode.MANAGED,
                               save_artifacts=args.save_artifacts)
    client = SimulatorEnvClient(service)
    scene = None
    report = {"diagnostic_only": True, "final_score_available": False,
              "map_id": args.map_id, "case_id": args.case_id, "actions": [],
              "reuse": args.reuse_verified_post_action_state, "snapshots": [], "windows": []}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    class Recorder:
        def event(self, name, payload):
            with (output / "stabilization.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": name, "payload": payload}) + "\n")
    stabilizer = UavStabilizer(ToolDispatcher(client, default_registry()), Recorder(),
                              UavStabilizationConfig(reuse_verified_post_action_state=args.reuse_verified_post_action_state))
    prepared = None
    if args.case_id:
        from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
        from evaluation.benchmarks.travel_uav.adapter import TravelUavCaseAdapter
        from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController
        cases, provenance = load_travel_uav_index(".external_refs/TravelUAV/data/travel_uav_index.json", split="test_seen")
        case = next(c for c in cases if c.episode_id == args.case_id)
        prepared = TravelUavCaseAdapter(scene_map={case.map_name: args.map_id},
                                       observation_profile=client.observation_profile).prepare_case(case)
        pose = prepared.execution_options.initial_pose.to_action_args()
        report["dataset_sha256"] = provenance.index_sha256
    else:
        if args.pose is None:
            parser.error("--pose or --case-id is required")
        pose = dict(zip(("x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"), args.pose))
    report["requested_initial_pose"] = pose
    sequence = json.loads(args.sequence_file.read_text(encoding="utf-8")) if args.sequence_file else [
        {"action": "turn_relative", "args": {"yaw_deg": args.turn}},
        {"action": "move_body", "args": dict(zip(("vx", "vy", "vz"), args.velocity)) | {"duration": args.duration}}]
    report["sequence"] = sequence
    save()
    try:
        startup = time.monotonic()
        scene = client.start_scene(args.map_id)
        report["startup_seconds"] = time.monotonic() - startup
        report["scene"] = scene.to_dict()
        (output / "state").mkdir()
        (output / "state/scene_instance.json").write_text(json.dumps(scene.to_dict()), encoding="utf-8")
        if prepared:
            TravelUavScenarioController().prepare_episode(scene, prepared.scenario_payload)
        def action(name, arguments):
            result = client.execute_action(scene.scene_instance_id, "Drone_1", name, arguments)
            report["actions"].append({"action": name, "args": arguments, **result.to_dict()})
            save()
            if result.execution_state.get("requires_scene_reset"):
                raise SceneRecoveryRequired(result.message)
            return result
        connected = action("connect", {})
        if not connected.ok:
            raise RuntimeError(connected.message)
        adapter = service.backends[scene.scene_instance_id].uav
        executor = adapter._control._actions
        if args.yaw_timeout is not None:
            executor.policy = replace(executor.policy, yaw_timeout_sec=args.yaw_timeout)
        if args.diagnostic_snapshots:
            freeze = executor._freeze_at_current_pose
            def traced_freeze(sdk):
                report["snapshots"].append({"stage": "before_freeze", "state": executor._multirotor_snapshot(sdk)})
                save()
                return freeze(sdk)
            executor._freeze_at_current_pose = traced_freeze
        placed = action("set_pose", pose)
        if not placed.ok:
            raise RuntimeError(placed.message)
        measured = action("get_pose", {}).data
        if args.diagnostic_snapshots:
            report["snapshots"].append({"stage": "placed", "state": executor._multirotor_snapshot(adapter._client)})
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
        report["initialization_to_observation_seconds"] = time.monotonic() - startup
        if not args.save_artifacts and list((output / "artifacts").rglob("*")):
            raise RuntimeError("Memory-only capture created backend artifacts")
        for item in sequence:
            started = time.monotonic()
            rpc_before = adapter.performance.rpc_count_total
            result = action(item["action"], item["args"])
            window = {"action": item["action"], "args": item["args"], "result": result.to_dict()}
            report["windows"].append(window)
            if result.ok:
                window["stabilization"] = stabilizer.wait_after_step(
                    scene.scene_instance_id, "Drone_1", "diagnostic",
                    PlannedStep.tool_step("uav." + item["action"], item["args"], "fixed diagnostic"),
                    capabilities=scene.capabilities,
                    post_action_state=result.execution_state.get("post_action_state_token"))
            window["observation"] = client.get_observation(
                scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]).to_dict()
            window["action_to_observation_seconds"] = time.monotonic() - started
            window["rpc_count"] = adapter.performance.rpc_count_total - rpc_before
            report["final_observation"] = window["observation"]
            save()
            # A bounded controller timeout with successful cancel/freeze permits
            # a separately configured next action. Other failures stop this probe.
            if not result.ok and not (args.yaw_timeout is not None and
                    result.data.get("failure_kind") == "controller_timeout_or_rejected"):
                break
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["requires_scene_reset"] = isinstance(exc, SceneRecoveryRequired)
    finally:
        if scene is not None:
            try:
                client.stop_scene(scene.scene_instance_id)
                report["cleanup"] = {"ok": True}
            except Exception as exc:
                report["cleanup"] = {"ok": False, "error": str(exc)}
        path = output / "report.json"
        save()
        print(path)
    failed_action = any(not row["ok"] for row in report["actions"])
    return 1 if failed_action or report.get("error") or not report.get("cleanup", {}).get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
