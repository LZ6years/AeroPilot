"""Run the real remote AirSim safety loop through SimulatorEnvClient.

The probe attaches to a healthy UE or starts it through the uploaded one-shot
Worker, captures observations, moves 0.5 m, restores the initial pose, resets,
and stops only a process owned by this SceneInstance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_client.simulator_client import SimulatorEnvClient
from src.env_client.observation_profile import load_observation_profile
from src.perception.raw_observation import RawObservation
from src.runtime.configs.map_config import load_map_config
from src.sim_server import SimulationExecutionMode
from src.sim_server.server import SimulationServer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument(
        "--run-root", default="runs/manual_remote_ssh_probe"
    )
    parser.add_argument(
        "--observation-profile",
        default="configs/observations/airsim_default.example.json",
        help=(
            "Run-level capture-source selection; this remains separate from "
            "the deployment MapPackage"
        ),
    )
    parser.add_argument(
        "--probe-pose-canonical",
        default=None,
        type=_parse_canonical_pose,
        help=(
            "Optional safe map pose as x,y,z,yaw_deg,pitch_deg,roll_deg in "
            "the canonical positive-up frame. Use a dataset start pose when "
            "the packaged map's origin is not a renderable camera location."
        ),
    )
    parser.add_argument(
        "--pose-settle-sec",
        type=float,
        default=1.0,
        help="Settle time after applying an explicit probe pose.",
    )
    args = parser.parse_args()
    probe_pose = args.probe_pose_canonical
    if args.pose_settle_sec < 0:
        parser.error("--pose-settle-sec must be non-negative")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    maps, policy = load_map_config(args.config)
    selected = next(package for package in maps if package.map_id == args.map_id)
    (run_dir / "map_package.json").write_text(
        json.dumps(selected.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    service = SimulationServer(
        maps,
        policy,
        run_dir=run_dir,
        execution_mode=SimulationExecutionMode.MANAGED,
        save_artifacts=True,
    )
    observation_profile = load_observation_profile(args.observation_profile)
    client = SimulatorEnvClient(
        service,
        observation_profile=observation_profile,
    )
    (run_dir / "observation_profile.json").write_text(
        json.dumps(
            observation_profile.to_dict(), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    scene = client.start_scene(args.map_id)
    scene_payload = scene.to_dict()
    (run_dir / "scene_instance.json").write_text(
        json.dumps(scene_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    source_settings = (
        run_dir
        / "scenes"
        / scene.scene_instance_id
        / "source.settings.json"
    )
    report: dict = {
        "scene": scene_payload,
        "deployment": {
            "map_id": selected.map_id,
            "scene_id": selected.scene_id,
            "executable": selected.launch.get("linux_executable"),
            "remote_endpoint": {
                "host": selected.airsim.get("remote_host", "127.0.0.1"),
                "port": selected.airsim.get("port"),
            },
        },
        "settings": {
            "source_path": selected.airsim.get("settings_template"),
            "rendered_sha256": scene.settings_sha256,
            "source_sha256_before": (
                _sha256(source_settings) if source_settings.is_file() else None
            ),
        },
        "cleanup": {},
    }
    settings_mismatch: str | None = None
    cleanup_mismatch: str | None = None
    try:
        if scene.process_ownership == "external":
            raise AssertionError(
                "Remote probe refuses to exercise an external AirSim process"
            )
        connected = client.execute_action(
            scene.scene_instance_id, "Drone_1", "connect", {}
        )
        if not connected.ok:
            raise RuntimeError(connected.message)
        initial_pose_result = None
        if probe_pose is not None:
            initial_pose_result = client.execute_action(
                scene.scene_instance_id,
                "Drone_1",
                "set_pose",
                probe_pose,
            )
            if not initial_pose_result.ok:
                raise RuntimeError(initial_pose_result.message)
            if args.pose_settle_sec:
                time.sleep(args.pose_settle_sec)
        before = client.get_observation(
            scene.scene_instance_id,
            "Drone_1",
            ["rgb", "depth", "pose"],
        )
        raw_before = RawObservation.from_env_observation(
            before, source="remote_probe_before"
        )
        # Persist the earliest control/readback evidence before validating
        # pixels so a failed real-engine gate still explains whether set_pose
        # reached AirSim and which pose the camera reported.
        report.update(
            {
                "connect": connected.to_dict(),
                "probe_pose_canonical": probe_pose,
                "initial_pose_result": (
                    initial_pose_result.to_dict()
                    if initial_pose_result is not None
                    else None
                ),
                "before": before.to_dict(),
            }
        )
        _validate_frames(raw_before.frames)
        report["frame_contract"] = _validate_dimensions(
            raw_before.frames,
            source_settings,
            vehicle_name=str(selected.airsim["vehicle_name"]),
            camera_name=str(selected.airsim["camera_name"]),
        )
        initial = _pose_args(raw_before.pose)

        moved = client.execute_action(
            scene.scene_instance_id,
            "Drone_1",
            "move_body",
            {"vx": 0.5, "vy": 0.0, "vz": 0.0, "duration": 1.0},
        )
        if not moved.ok:
            raise RuntimeError(moved.message)
        after_move = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]
        )
        raw_after_move = RawObservation.from_env_observation(
            after_move, source="remote_probe_after_move"
        )
        displacement = _distance(
            initial, _pose_args(raw_after_move.pose)
        )
        if displacement < 0.2:
            raise AssertionError(
                f"Expected pose displacement >=0.2m, got {displacement:.3f}m"
            )

        restored = client.execute_action(
            scene.scene_instance_id,
            "Drone_1",
            "set_pose",
            initial,
        )
        if not restored.ok:
            raise RuntimeError(restored.message)
        after_restore = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["pose"]
        )
        raw_after_restore = RawObservation.from_env_observation(
            after_restore, source="remote_probe_after_restore"
        )
        restore_error = _distance(
            initial, _pose_args(raw_after_restore.pose)
        )
        if restore_error > 0.1:
            raise AssertionError(
                f"Pose restore error exceeds 0.1m: {restore_error:.3f}m"
            )

        reset_scene = client.reset_scene(scene.scene_instance_id)
        reset_pose_result = None
        if probe_pose is not None:
            # AirSim reset returns the ComputerVision vehicle to the map
            # origin, which is not guaranteed to be a renderable camera pose.
            # Reapply the verified episode pose before validating capture.
            reset_pose_result = client.execute_action(
                scene.scene_instance_id,
                "Drone_1",
                "set_pose",
                initial,
            )
            if not reset_pose_result.ok:
                raise RuntimeError(reset_pose_result.message)
            if args.pose_settle_sec:
                time.sleep(args.pose_settle_sec)
        after_reset = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]
        )
        raw_after_reset = RawObservation.from_env_observation(
            after_reset, source="remote_probe_after_reset"
        )
        _validate_frames(raw_after_reset.frames)
        report.update(
            {
                "move": moved.to_dict(),
                "after_move": after_move.to_dict(),
                "displacement_m": displacement,
                "restore": restored.to_dict(),
                "after_restore": after_restore.to_dict(),
                "restore_error_m": restore_error,
                "reset": reset_scene.to_dict(),
                "reset_pose_result": (
                    reset_pose_result.to_dict()
                    if reset_pose_result is not None
                    else None
                ),
                "after_reset": after_reset.to_dict(),
            }
        )
        return 0
    finally:
        try:
            client.stop_scene(scene.scene_instance_id)
            report["cleanup"] = {
                "ok": True,
                "process_ownership": scene.process_ownership,
            }
        except Exception as exc:
            report["cleanup"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if (
            selected.connection_mode == "remote"
            and scene.process_ownership == "remote_managed"
        ):
            try:
                released = _remote_port_released(service, selected)
                report["cleanup"]["remote_port_released"] = released
                if not released:
                    cleanup_mismatch = (
                        "Remote AirSim port remains reachable after cleanup"
                    )
            except Exception as exc:
                report["cleanup"]["port_verification_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                cleanup_mismatch = "Remote AirSim port release was not verified"
        if (
            selected.connection_mode == "remote"
            and selected.airsim.get("settings_template_source") == "remote"
        ):
            try:
                source_after = run_dir / "source.settings.after.json"
                service.remote_scene_control.download_settings_template(
                    selected, source_after
                )
                before_hash = report["settings"]["source_sha256_before"]
                after_hash = _sha256(source_after)
                unchanged = bool(before_hash and before_hash == after_hash)
                report["settings"].update(
                    {
                        "source_sha256_after": after_hash,
                        "remote_source_unchanged": unchanged,
                    }
                )
                if not unchanged:
                    settings_mismatch = (
                        "Remote source settings SHA-256 changed during probe"
                    )
            except Exception as exc:
                report["settings"]["post_cleanup_verification_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        output = run_dir / "remote_probe_report.json"
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(run_dir), **report}, indent=2, ensure_ascii=False))
        if settings_mismatch is not None:
            raise AssertionError(settings_mismatch)
        if cleanup_mismatch is not None:
            raise AssertionError(cleanup_mismatch)


def _pose_args(pose: dict) -> dict[str, float | bool]:
    position = pose.get("position", pose)
    euler = pose.get("euler_rad", {})
    return {
        "x": float(position["x"]),
        "y": float(position["y"]),
        "z": float(position["z"]),
        "pitch_deg": math.degrees(float(euler.get("pitch", 0.0))),
        "roll_deg": math.degrees(float(euler.get("roll", 0.0))),
        "yaw_deg": math.degrees(float(euler.get("yaw", 0.0))),
        "ignore_collision": True,
    }


def _parse_canonical_pose(
    value: str | None,
) -> dict[str, float | bool] | None:
    """Parse an explicit reusable map pose without benchmark coupling."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "probe pose must contain x,y,z,yaw_deg,pitch_deg,roll_deg"
        )
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "probe pose values must be numeric"
        ) from exc
    if not all(math.isfinite(number) for number in numbers):
        raise argparse.ArgumentTypeError("probe pose values must be finite")
    x, y, z, yaw, pitch, roll = numbers
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "ignore_collision": True,
    }


def _distance(first: dict, second: dict) -> float:
    return math.sqrt(
        sum(
            (float(second[axis]) - float(first[axis])) ** 2
            for axis in ("x", "y", "z")
        )
    )


def _validate_frames(frames: dict) -> None:
    rgb = frames.get("rgb", {})
    depth = frames.get("depth", {})
    if int(rgb.get("width", 0)) <= 0 or int(rgb.get("height", 0)) <= 0:
        raise AssertionError(f"Invalid RGB dimensions: {rgb}")
    if int(rgb.get("bytes", 0)) <= 0:
        raise AssertionError(f"RGB payload is empty: {rgb}")
    if int(depth.get("width", 0)) <= 0 or int(depth.get("height", 0)) <= 0:
        raise AssertionError(f"Invalid depth dimensions: {depth}")
    if int(depth.get("float_count", 0)) <= 0:
        raise AssertionError(f"Depth payload is empty: {depth}")
    rgb_artifact = Path(str(rgb.get("artifact_path", "")))
    depth_artifact = Path(str(depth.get("artifact_path", "")))
    if not rgb_artifact.is_file() or rgb_artifact.stat().st_size <= 0:
        raise AssertionError(f"RGB artifact is missing or empty: {rgb_artifact}")
    if not depth_artifact.is_file() or depth_artifact.stat().st_size <= 0:
        raise AssertionError(
            f"Depth artifact is missing or empty: {depth_artifact}"
        )
    import numpy as np
    from PIL import Image

    with Image.open(rgb_artifact) as image:
        rgb_values = np.asarray(image.convert("RGB"), dtype=np.uint8)
    unique_colors = np.unique(rgb_values.reshape(-1, 3), axis=0)
    if float(rgb_values.std()) <= 1.0 or len(unique_colors) <= 16:
        raise AssertionError(
            "RGB artifact is degenerate: "
            f"std={float(rgb_values.std()):.6f}, "
            f"unique_colors={len(unique_colors)}"
        )
    depth_values = np.load(depth_artifact)
    if depth_values.size <= 0 or not bool(np.isfinite(depth_values).all()):
        raise AssertionError("Depth artifact contains empty or non-finite data")
    if float(depth_values.std()) <= 1e-6:
        raise AssertionError(
            f"Depth artifact is constant: std={float(depth_values.std()):.9f}"
        )


def _validate_dimensions(
    frames: dict,
    settings_path: Path,
    *,
    vehicle_name: str,
    camera_name: str,
) -> dict[str, list[int]]:
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    captures = settings["Vehicles"][vehicle_name]["Cameras"][camera_name][
        "CaptureSettings"
    ]
    expected = {
        str(item["ImageType"]): [int(item["Width"]), int(item["Height"])]
        for item in captures
    }
    actual = {
        "rgb": [
            int(frames["rgb"]["width"]),
            int(frames["rgb"]["height"]),
        ],
        "depth": [
            int(frames["depth"]["width"]),
            int(frames["depth"]["height"]),
        ],
    }
    if actual["rgb"] != expected.get("0"):
        raise AssertionError(
            f"RGB dimensions {actual['rgb']} do not match settings "
            f"{expected.get('0')}"
        )
    if actual["depth"] != expected.get("2"):
        raise AssertionError(
            f"Depth dimensions {actual['depth']} do not match settings "
            f"{expected.get('2')}"
        )
    return actual


def _remote_port_released(service, package) -> bool:
    target = service.remote_scene_control.target(package)
    output = service.remote_scene_control.runner.run(
        target,
        [
            "python3",
            "-c",
            (
                "import json,socket,sys; "
                "s=socket.socket(); s.settimeout(0.5); "
                "r=s.connect_ex((sys.argv[1],int(sys.argv[2]))); s.close(); "
                "print(json.dumps({'listening':r==0}))"
            ),
            str(package.airsim.get("remote_host", "127.0.0.1")),
            str(package.airsim["port"]),
        ],
    )
    return not bool(json.loads(output.strip().splitlines()[-1])["listening"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
