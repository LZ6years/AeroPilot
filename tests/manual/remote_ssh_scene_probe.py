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
    args = parser.parse_args()

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
        "settings": {
            "rendered_sha256": scene.settings_sha256,
            "source_sha256_before": (
                _sha256(source_settings) if source_settings.is_file() else None
            ),
        },
        "cleanup": {},
    }
    settings_mismatch: str | None = None
    try:
        connected = client.execute_action(
            scene.scene_instance_id, "Drone_1", "connect", {}
        )
        if not connected.ok:
            raise RuntimeError(connected.message)
        before = client.get_observation(
            scene.scene_instance_id,
            "Drone_1",
            ["rgb", "depth", "pose"],
        )
        raw_before = RawObservation.from_env_observation(
            before, source="remote_probe_before"
        )
        _validate_frames(raw_before.frames)
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
        after_reset = client.get_observation(
            scene.scene_instance_id, "Drone_1", ["rgb", "depth", "pose"]
        )
        raw_after_reset = RawObservation.from_env_observation(
            after_reset, source="remote_probe_after_reset"
        )
        _validate_frames(raw_after_reset.frames)
        report.update(
            {
                "connect": connected.to_dict(),
                "before": before.to_dict(),
                "move": moved.to_dict(),
                "after_move": after_move.to_dict(),
                "displacement_m": displacement,
                "restore": restored.to_dict(),
                "after_restore": after_restore.to_dict(),
                "restore_error_m": restore_error,
                "reset": reset_scene.to_dict(),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
