"""Compare physics, estimator and rendered poses in isolated TravelUAV scenes.

No Planner, model calls or score updates run here. The campaign supplies exact
retained initialization/first-turn arguments; a separate map config owns ports,
GPU and remote diagnostic directories. All owned scenes are stopped in finally.
Run with the flyagent interpreter, --campaign, --map-config, repeated --case-id,
and a new --output directory. Raw SDK positions below remain in NED metres.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs.map_config import load_map_config
from src.sim_server import SimulationExecutionMode
from src.sim_server.server import SimulationServer
from src.adapters.airsim.capture import AirSimCapture
from src.runtime.control.stabilization import UavStabilizer
from src.tools.dispatcher import ToolDispatcher
from src.tools.registry import default_registry
from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
from evaluation.benchmarks.travel_uav.adapter import TravelUavCaseAdapter
from evaluation.benchmarks.travel_uav.scenario import TravelUavScenarioController


def plain(value):
    """Retain numeric SDK structures without embedding image transport bytes."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items() if key != "base64"}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    return value


def independent_snapshot(sdk, camera="FrontCamera", vehicle="Drone_1"):
    """Keep independent APIs separate: no source silently replaces another."""
    result = {}
    for key, read in (
        ("estimated", lambda: sdk.getMultirotorState(vehicle_name=vehicle)),
        ("physics", lambda: sdk.simGetGroundTruthKinematics(vehicle_name=vehicle)),
        ("rendered_pose", lambda: sdk.simGetVehiclePose(vehicle_name=vehicle)),
        ("camera", lambda: sdk.simGetCameraInfo(camera, vehicle_name=vehicle)),
        ("collision", lambda: sdk.simGetCollisionInfo(vehicle_name=vehicle)),
        ("paused", lambda: sdk.simIsPause()),
    ):
        try:
            result[key] = plain(read())
        except Exception as exc:
            result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


class DiagnosticCapture(AirSimCapture):
    """Read camera pose from the same image response without an extra RPC."""

    def _frame_metadata(self, modality, capture_source, spec, response):
        value = super()._frame_metadata(modality, capture_source, spec, response)
        value["diagnostic_camera_position_ned"] = plain(getattr(response, "camera_position", None))
        value["diagnostic_camera_orientation_ned"] = plain(getattr(response, "camera_orientation", None))
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--map-config", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-budget", default="30")
    parser.add_argument("--index", default=".external_refs/TravelUAV/data/travel_uav_index.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--minimal-before-turn", action="store_true",
                        help="Preserve placement/image/turn RPC order; no independent pre-turn sampling")
    parser.add_argument("--decision-delay-sec", type=float, default=0,
                        help="Leave the scene paused to reproduce the model decision interval")
    args = parser.parse_args()
    if args.repeat < 1 or not math.isfinite(args.decision_delay_sec) or args.decision_delay_sec < 0:
        parser.error("repeat must be positive and decision-delay-sec finite and non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    environment = {"sys_executable": sys.executable, "torch_version": torch.__version__,
                   "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    rows = list(csv.DictReader((args.campaign / "per_case.csv").open(encoding="utf-8-sig")))
    maps, policy = load_map_config(args.map_config)
    cases, provenance = load_travel_uav_index(args.index, split="test_seen")
    results = []
    for sample_index, case_id in enumerate(args.case_id * args.repeat):
        row = next(r for r in rows if r["dataset"] == "travel_uav" and
                   r["image_budget"] == args.image_budget and r["episode_id"] == case_id)
        source_run = Path(row["run_dir"])
        actions = [json.loads(line) for line in (source_run / "memory/actions.jsonl").open(encoding="utf-8")]
        first_turn = next(a["args"] for a in actions if a["tool"] == "uav.turn_relative")
        map_config = next(m for m in maps if m.scene_id == case_id.split("__")[0])
        case = next(c for c in cases if c.episode_id == case_id)
        prepared = TravelUavCaseAdapter(scene_map={case.map_name: map_config.map_id},
            observation_profile=SimulatorEnvClient(None).observation_profile).prepare_case(case)
        initial = prepared.execution_options.initial_pose.to_action_args()
        output = args.output / case_id
        if args.repeat > 1:
            output = output / f"repeat_{sample_index // len(args.case_id) + 1:02d}"
        output.mkdir(parents=True)
        report = {"case_id": case_id, "source_run": str(source_run),
                  "initial_pose": initial, "first_turn": first_turn, "snapshots": [],
                  "dataset_sha256": provenance.index_sha256,
                  "minimal_before_turn": args.minimal_before_turn,
                  "decision_delay_sec": args.decision_delay_sec}
        def save():
            (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        service = SimulationServer(maps, policy, run_dir=output,
                                  execution_mode=SimulationExecutionMode.MANAGED, save_artifacts=True)
        client = SimulatorEnvClient(service)
        scene = None
        try:
            started = time.monotonic()
            scene = client.start_scene(map_config.map_id)
            report["scene"] = scene.to_dict()
            TravelUavScenarioController().prepare_episode(scene, prepared.scenario_payload)
            connected = client.execute_action(scene.scene_instance_id, "Drone_1", "connect", {})
            if not connected.ok:
                raise RuntimeError(connected.message)
            adapter = service.backends[scene.scene_instance_id].uav
            sdk = adapter._client
            import airsim
            capture = DiagnosticCapture("FrontCamera", "Drone_1", output / "capture", True)
            def snapshot(stage, independent=True):
                item = {"stage": stage, **(independent_snapshot(sdk) if independent else {})}
                item["frames"] = plain(capture.get_frames(sdk, airsim, ["rgb", "depth"]).data)
                report["snapshots"].append(item)
                save()
            if not args.minimal_before_turn:
                snapshot("before_placement")
            report["placement"] = plain(client.execute_action(scene.scene_instance_id, "Drone_1", "set_pose", initial).to_dict())
            if not report["placement"]["ok"]:
                raise RuntimeError("episode placement failed; no turn issued")
            # Match production initialization: hover and zero-velocity brake
            # follow placement before the first RGB/depth observation.
            report["preflight_hover"] = plain(client.execute_action(scene.scene_instance_id, "Drone_1", "hover", {}).to_dict())
            if not report["preflight_hover"]["ok"]:
                raise RuntimeError("preflight hold failed; no turn issued")
            report["preflight_stabilization"] = plain(UavStabilizer(ToolDispatcher(client, default_registry())).wait_after_preflight(
                scene.scene_instance_id, "Drone_1", capabilities=scene.capabilities))
            snapshot("after_placement", independent=not args.minimal_before_turn)
            if not args.minimal_before_turn:
                snapshot("paused_stationary")
            if args.decision_delay_sec:
                report["stage"] = "paused_decision_interval"
                save()
                time.sleep(max(0, args.decision_delay_sec))
            # Read after a fatal action directly for diagnosis, without asking
            # Runtime to continue or issuing any further physical command.
            report["turn"] = plain(adapter.turn_relative(first_turn["yaw_deg"]).__dict__)
            snapshot("after_turn")
            report["stage"] = "turn_completed"
            report["elapsed_sec"] = time.monotonic() - started
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if scene is not None:
                try:
                    client.stop_scene(scene.scene_instance_id)
                    report["cleanup"] = {"ok": True}
                except Exception as exc:
                    report["cleanup"] = {"ok": False, "error": str(exc)}
            save()
        results.append({"case_id": case_id, "report": str(output / "report.json"),
                        "error": report.get("error"), "cleanup": report.get("cleanup")})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    (args.output / "index.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return int(any(r["error"] or not r.get("cleanup", {}).get("ok") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
