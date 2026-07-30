"""Validate local UnrealCV actor control through the frozen EnvClient API.

This is a developer-facing real-engine probe, not a Benchmark runner.  It does
not create targets, place objects, read episode truth, or import UAV-Flow.  A
MapPackage supplies only deployment and primary-vehicle control information.

Safety defaults
---------------
The probe starts and captures a scene but does not move it unless
``--yaw-deg`` or ``--move-forward-m`` is explicitly provided.  Whenever motion
is requested it records the initial canonical pose, restores that pose in a
``finally`` path, and then asks EnvClient to stop only the process owned by the
current SceneInstance.

Example
-------
After copying the portable config example and replacing its local root:

    python tests/manual/unrealcv_local_actor_probe.py \
        --config configs/maps/unrealcv_actor_control.local.json \
        --map-id downtownwest_local_unrealcv \
        --yaw-deg 10 \
        --move-forward-m 0.25 \
        --speed-mps 0.5 \
        --require-visual

Inspect ``runs/manual_local_unrealcv_probe/<run-id>/summary.json`` plus the
RGB, depth, depth-preview, and segmentation artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_client import ObservationProfile, SimulatorEnvClient
from src.perception.raw_observation import RawObservation
from src.runtime.configs.map_config import load_map_config
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


def build_parser() -> argparse.ArgumentParser:
    """Define explicit real-engine inputs and conservative motion defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="MapPackage JSON path")
    parser.add_argument("--map-id", required=True, help="MapPackage map_id")
    parser.add_argument(
        "--execution",
        choices=("attach", "managed"),
        default="managed",
        help="Attach only, or let EnvClient own local process startup/cleanup",
    )
    parser.add_argument("--agent-id", default="drone_1")
    parser.add_argument(
        "--run-root",
        default="runs/manual_local_unrealcv_probe",
        help="Ignored output root for reports, UE logs, and image artifacts",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="Optional relative yaw test; zero performs no turn",
    )
    parser.add_argument(
        "--move-forward-m",
        type=float,
        default=0.0,
        help="Optional forward distance; zero performs no translation",
    )
    parser.add_argument("--speed-mps", type=float, default=0.5)
    parser.add_argument(
        "--require-visual",
        action="store_true",
        help="Reject constant RGB/depth artifacts instead of only saving them",
    )
    return parser


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run one ownership-aware actor-control acceptance sequence.

    External effects:
    - may start one local UE process in MANAGED mode;
    - may turn or move the configured primary actor when explicitly requested;
    - writes only beneath ``run_root``;
    - restores the initial pose and stops only the owned SceneInstance.
    """

    if args.move_forward_m < 0.0:
        raise ValueError("--move-forward-m must be zero or positive")
    if args.speed_mps <= 0.0:
        raise ValueError("--speed-mps must be greater than zero")

    packages, port_policy = load_map_config(args.config)
    package = _select_package(packages, args.map_id)
    if package.control_protocol != "unrealcv":
        raise ValueError("This probe requires control.protocol=unrealcv")
    if package.unrealcv.get("control_target") != "actor":
        raise ValueError("This probe requires unrealcv.control_target=actor")

    run_dir = _new_run_dir(args.run_root)
    _write_json(run_dir / "resolved_map_package.json", package.to_dict())
    profile = ObservationProfile.from_dict(
        {
            "modalities": ["rgb", "depth", "segmentation", "pose"],
            "capture_sources": {
                "rgb": "scene",
                "depth": "depth_perspective",
                "segmentation": "segmentation",
            },
        }
    )
    service = SimulationServer(
        packages,
        port_policy,
        run_dir=run_dir,
        execution_mode=SimulationExecutionMode.parse(args.execution),
        save_artifacts=True,
        show_display=True,
    )
    client = SimulatorEnvClient(service, observation_profile=profile)
    summary: dict[str, Any] = {
        "status": "starting",
        "run_dir": str(run_dir),
        "map_id": package.map_id,
        "execution": args.execution,
        "requested_motion": {
            "yaw_deg": args.yaw_deg,
            "move_forward_m": args.move_forward_m,
            "speed_mps": args.speed_mps,
        },
        "poses": {},
        "checks": {},
        "cleanup": {"attempted": False},
    }
    scene = None
    initial_pose: dict[str, Any] | None = None
    restored = False

    try:
        # The public client sees only map_id/scene_id. Backend selection,
        # ProcessManager, and UnrealCV strategy selection remain below it.
        scene = client.start_scene(package.map_id, package.scene_id)
        _write_json(run_dir / "scene_instance.json", scene.to_dict())
        summary["scene"] = scene.to_dict()
        capabilities = scene.capabilities
        if capabilities is None or capabilities.control_mode != "actor_pose":
            raise AssertionError(
                f"Expected actor_pose capabilities, received {capabilities}"
            )
        connected = client.execute_action(
            scene.scene_instance_id, args.agent_id, "connect", {}
        )
        if not connected.ok:
            raise RuntimeError(connected.message)

        before = _observe(client, scene.scene_instance_id, args.agent_id)
        initial_pose = before.pose
        summary["poses"]["before"] = before.pose
        _write_json(run_dir / "observation_before.json", before.to_dict())
        summary["checks"]["visual_before"] = _visual_quality(
            before, require=args.require_visual
        )

        if args.yaw_deg:
            result = client.execute_action(
                scene.scene_instance_id,
                args.agent_id,
                "turn_relative",
                {"yaw_deg": args.yaw_deg},
            )
            if not result.ok:
                raise RuntimeError(result.message)
            after_turn = _observe(client, scene.scene_instance_id, args.agent_id)
            summary["poses"]["after_turn"] = after_turn.pose
            _write_json(run_dir / "observation_after_turn.json", after_turn.to_dict())
            summary["checks"]["turn"] = {
                "position_drift_m": _position_distance(before.pose, after_turn.pose),
                "yaw_delta_deg": _yaw_delta(
                    _yaw(before.pose), _yaw(after_turn.pose)
                ),
            }

        if args.move_forward_m:
            duration = args.move_forward_m / args.speed_mps
            result = client.execute_action(
                scene.scene_instance_id,
                args.agent_id,
                "move_body",
                {
                    "vx": args.speed_mps,
                    "vy": 0.0,
                    "vz": 0.0,
                    "duration": duration,
                },
            )
            if not result.ok:
                raise RuntimeError(result.message)
            after_move = _observe(client, scene.scene_instance_id, args.agent_id)
            summary["poses"]["after_move"] = after_move.pose
            _write_json(run_dir / "observation_after_move.json", after_move.to_dict())
            motion_origin = (
                summary["poses"].get("after_turn") or summary["poses"]["before"]
            )
            summary["checks"]["move"] = {
                "position_delta_m": _position_distance(
                    motion_origin, after_move.pose
                )
            }

        # Explicit set_pose restoration is the portable UnrealCV episode
        # primitive. The Backend intentionally does not advertise a fake soft
        # reset because scenario objects belong to the Benchmark layer.
        if args.yaw_deg or args.move_forward_m:
            restore_result = client.execute_action(
                scene.scene_instance_id,
                args.agent_id,
                "set_pose",
                _set_pose_args(initial_pose),
            )
            if not restore_result.ok:
                raise RuntimeError(restore_result.message)
            restored_observation = _observe(
                client, scene.scene_instance_id, args.agent_id
            )
            restored = True
            summary["poses"]["restored"] = restored_observation.pose
            _write_json(
                run_dir / "observation_restored.json",
                restored_observation.to_dict(),
            )
            summary["checks"]["restore"] = {
                "position_error_m": _position_distance(
                    initial_pose, restored_observation.pose
                ),
                "yaw_error_deg": abs(
                    _yaw_delta(_yaw(initial_pose), _yaw(restored_observation.pose))
                ),
            }
            summary["checks"]["visual_restored"] = _visual_quality(
                restored_observation, require=args.require_visual
            )

        _assert_acceptance(summary, args)
        summary["status"] = "completed"
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Restoration is attempted again if an exception occurred after the
        # first movement. Cleanup must proceed even when restoration fails.
        if scene is not None:
            summary["cleanup"]["attempted"] = True
            if initial_pose is not None and not restored and (
                args.yaw_deg or args.move_forward_m
            ):
                try:
                    client.execute_action(
                        scene.scene_instance_id,
                        args.agent_id,
                        "set_pose",
                        _set_pose_args(initial_pose),
                    )
                    summary["cleanup"]["emergency_restore"] = True
                except Exception as restore_error:
                    summary["cleanup"]["emergency_restore"] = False
                    summary["cleanup"]["restore_error"] = str(restore_error)
            try:
                ownership = scene.process_ownership
                client.stop_scene(scene.scene_instance_id)
                summary["cleanup"].update(
                    {
                        "ok": True,
                        "process_ownership": ownership,
                        "owned_process_stopped": ownership == "local",
                    }
                )
            except Exception as cleanup_error:
                summary["cleanup"].update(
                    {"ok": False, "error": str(cleanup_error)}
                )
        _write_json(run_dir / "poses.json", summary.get("poses", {}))
        _write_json(run_dir / "summary.json", summary)


def _observe(
    client: SimulatorEnvClient,
    scene_instance_id: str,
    agent_id: str,
) -> RawObservation:
    observation = client.get_observation(
        scene_instance_id,
        agent_id,
        ["rgb", "depth", "segmentation", "pose"],
    )
    return RawObservation.from_env_observation(
        observation, source="manual_local_unrealcv_probe"
    )


def _visual_quality(
    observation: RawObservation,
    *,
    require: bool,
) -> dict[str, Any]:
    """Validate saved artifacts without putting scene judgment in Backend."""

    import numpy as np
    from PIL import Image

    rgb_path = Path(str(observation.frames.get("rgb", {}).get("artifact_path", "")))
    depth_path = Path(
        str(observation.frames.get("depth", {}).get("artifact_path", ""))
    )
    segmentation_path = Path(
        str(
            observation.frames.get("segmentation", {}).get(
                "artifact_path", ""
            )
        )
    )
    for label, path in (
        ("RGB", rgb_path),
        ("depth", depth_path),
        ("segmentation", segmentation_path),
    ):
        if not path.is_file():
            raise AssertionError(f"{label} artifact is missing: {path}")

    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    with Image.open(segmentation_path) as image:
        segmentation = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = {
        "rgb": {
            "path": str(rgb_path),
            "shape": list(rgb.shape),
            "std": float(rgb.std()),
            "unique_colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
        },
        "depth": {
            "path": str(depth_path),
            "shape": list(depth.shape),
            "finite": bool(np.isfinite(depth).all()),
            "std": float(depth.std()),
            "unit": "meter",
        },
        "segmentation": {
            "path": str(segmentation_path),
            "shape": list(segmentation.shape),
            "unique_colors": int(
                len(np.unique(segmentation.reshape(-1, 3), axis=0))
            ),
        },
    }
    result["ok"] = bool(
        result["rgb"]["std"] > 1.0
        and result["rgb"]["unique_colors"] > 16
        and result["depth"]["finite"]
        and result["depth"]["std"] > 1e-6
    )
    if require and not result["ok"]:
        raise AssertionError(f"Visual artifacts are degenerate: {result}")
    return result


def _assert_acceptance(summary: dict[str, Any], args: argparse.Namespace) -> None:
    if args.yaw_deg:
        turn = summary["checks"]["turn"]
        if turn["position_drift_m"] > 0.05:
            raise AssertionError(f"Turn position drift exceeded 0.05 m: {turn}")
        if abs(_yaw_delta(args.yaw_deg, turn["yaw_delta_deg"])) > 2.0:
            raise AssertionError(f"Turn yaw error exceeded 2 deg: {turn}")
    if args.move_forward_m:
        actual = summary["checks"]["move"]["position_delta_m"]
        if not 0.20 <= actual <= 0.30:
            raise AssertionError(
                f"Forward displacement must be 0.20-0.30 m, received {actual}"
            )
    if args.yaw_deg or args.move_forward_m:
        restore = summary["checks"]["restore"]
        if restore["position_error_m"] > 0.05 or restore["yaw_error_deg"] > 2.0:
            raise AssertionError(f"Pose restoration exceeded tolerance: {restore}")


def _set_pose_args(pose: dict[str, Any]) -> dict[str, float | bool]:
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "pitch_deg": float(pose.get("pitch_deg", 0.0)),
        "yaw_deg": float(pose.get("yaw_deg", 0.0)),
        "roll_deg": float(pose.get("roll_deg", 0.0)),
        "ignore_collision": True,
    }


def _position_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.sqrt(
        sum(
            (float(second[axis]) - float(first[axis])) ** 2
            for axis in ("x", "y", "z")
        )
    )


def _yaw(pose: dict[str, Any]) -> float:
    return float(pose.get("yaw_deg", 0.0))


def _yaw_delta(first: float, second: float) -> float:
    return (float(second) - float(first) + 180.0) % 360.0 - 180.0


def _select_package(packages, map_id: str):
    for package in packages:
        if package.map_id == map_id:
            return package
    available = ", ".join(package.map_id for package in packages)
    raise ValueError(f"Unknown map_id {map_id!r}; available: {available}")


def _new_run_dir(run_root: str | Path) -> Path:
    run_dir = Path(run_root) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    try:
        return asdict(value)
    except TypeError as exc:
        raise TypeError(
            f"Value is not JSON serializable: {type(value).__name__}"
        ) from exc


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_probe(args)
    except Exception as exc:
        print(f"UnrealCV local actor probe failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Inspect: {Path(summary['run_dir']) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
