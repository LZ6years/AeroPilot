"""Demonstrate every simulator deployment through one SimulatorEnvClient.

This is a user-facing workflow demo, not a low-level diagnostic probe and not
an Agent/Benchmark runner. Its purpose is to make three independent decisions
visible:

1. MapPackage chooses local versus remote deployment.
2. MapPackage chooses AirSim versus UnrealCV protocol.
3. ``--execution`` chooses whether this run only attaches to an existing UE or
   may manage an owned UE process.
4. ``--observation-profile`` chooses run-level capture sources without changing
   the MapPackage or simulator deployment.

The default execution policy is ``managed``. It may start the configured UE
process, although movement remains disabled until ``--move-forward-m`` is
provided. To inspect configuration without SDK, SSH, tunnel, or process side
effects, use:

    python demos/simulator_env_client_demo.py --list-maps

Useful examples:

    # List the MapPackages contained in the example configuration.
    python demos/simulator_env_client_demo.py --list-maps

    # Connect to an already-running simulator, move 0.5 m, then restore pose.
    python demos/simulator_env_client_demo.py \
      --config configs/maps/remote_ssh.example.json \
      --map-id airsim_remote_env_1 \
      --execution attach \
      --move-forward-m 0.5 \
      --save-artifacts

Inspect the generated ``execution_summary.json`` first. It records the route,
capabilities, actions, ownership, output paths, and cleanup result without
requiring the reader to inspect internal Server objects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# A demo is commonly run as ``python demos/<file>.py``. In that form Python
# puts ``demos/`` rather than the repository root on sys.path, so add the root
# explicitly before importing the real project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_client import (
    ActionResult,
    SceneCapabilities,
    SceneInstance,
    SimulatorEnvClient,
    load_observation_profile,
)
from src.perception.raw_observation import RawObservation
from src.runtime.configs.map_config import load_map_config
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


DEFAULT_CONFIG = "configs/maps/simulator_control_demo.example.json"
DEFAULT_MAP_ID = "demo_local_airsim_cv"
DEFAULT_OBSERVATION_PROFILE = (
    "configs/observations/airsim_default.example.json"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the user-facing CLI without exposing Server implementation flags."""

    parser = argparse.ArgumentParser(
        description=(
            "Use one SimulatorEnvClient with Local/Remote and "
            "AirSim/UnrealCV MapPackages"
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--map-id", default=DEFAULT_MAP_ID)
    parser.add_argument(
        "--observation-profile",
        default=DEFAULT_OBSERVATION_PROFILE,
        help=(
            "Run-level modalities and capture sources. This is intentionally "
            "separate from MapPackage deployment configuration."
        ),
    )
    parser.add_argument(
        "--execution",
        choices=[mode.value for mode in SimulationExecutionMode],
        default=SimulationExecutionMode.MANAGED.value,
        help=(
            "attach connects but never owns UE; managed may start and stop "
            "an owned UE"
        ),
    )
    parser.add_argument("--agent-id", default="uav_1")
    parser.add_argument(
        "--run-root", default="runs/simulator_env_client_demo"
    )
    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="Print configured maps and exit without creating a scene",
    )
    parser.add_argument(
        "--move-forward-m",
        type=float,
        default=0.0,
        help=(
            "Optional bounded motion. The default 0 performs no movement; "
            "a positive value is restored to the initial pose afterward"
        ),
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=0.5,
        help="Positive body-frame forward speed used with --move-forward-m",
    )
    parser.add_argument(
        "--soft-reset",
        action="store_true",
        help="Request reset only when SceneCapabilities advertises support",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save real RGB/depth payloads when the selected Backend supports it",
    )
    parser.add_argument(
        "--require-visual",
        action="store_true",
        help=(
            "Developer acceptance mode: retry and reject constant RGB/depth "
            "artifacts instead of treating non-empty payloads as success"
        ),
    )
    parser.add_argument(
        "--visual-timeout-sec",
        type=float,
        default=30.0,
        help="Maximum wait for non-degenerate visual artifacts",
    )
    return parser


def list_maps(config_path: str | Path) -> list[dict[str, str]]:
    """Return the routing axes for each configured MapPackage.

    This function only parses local JSON. It does not create SimulationServer,
    so even a Remote MapPackage cannot trigger SSH or simulator access.
    """

    packages, _ = load_map_config(config_path)
    rows = []
    for package in packages:
        rows.append(
            {
                "map_id": package.map_id,
                "connection": package.connection_mode,
                "protocol": package.control_protocol,
                "control_mode": _configured_control_mode(package),
            }
        )
    return rows


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Run one complete EnvClient lifecycle and return its human-facing report.

    External effects are controlled exclusively by ``args.execution``. The
    function always records cleanup in ``execution_summary.json`` and always
    calls ``stop_scene`` for a created scene. ``stop_scene`` closes local
    clients/tunnels for attached scenes but stops UE only when ownership says
    this SceneInstance created that process.
    """

    if args.move_forward_m < 0:
        raise ValueError("--move-forward-m must be zero or positive")
    if args.speed_mps <= 0:
        raise ValueError("--speed-mps must be greater than zero")
    if args.visual_timeout_sec <= 0:
        raise ValueError("--visual-timeout-sec must be greater than zero")
    if args.require_visual and not args.save_artifacts:
        raise ValueError("--require-visual requires --save-artifacts")

    packages, port_policy = load_map_config(args.config)
    package = _select_package(packages, args.map_id)
    observation_profile = load_observation_profile(args.observation_profile)
    execution = SimulationExecutionMode.parse(args.execution)
    run_dir = _new_run_dir(args.run_root)

    # Persist the resolved package before touching a scene. This makes failures
    # reproducible while keeping execution mode outside MapPackage, where it
    # belongs.
    _write_json(run_dir / "resolved_map_package.json", package.to_dict())
    _write_json(
        run_dir / "resolved_observation_profile.json",
        observation_profile.to_dict(),
    )
    report: dict[str, Any] = {
        "status": "starting",
        "run_dir": str(run_dir),
        "route": {
            "map_id": package.map_id,
            "scene_id": package.scene_id,
            "connection_mode": package.connection_mode,
            "control_protocol": package.control_protocol,
            "configured_control_mode": _configured_control_mode(package),
            "execution_mode": execution.value,
        },
        "requested_flow": {
            "modalities": list(observation_profile.modalities),
            "capture_sources": dict(observation_profile.capture_sources),
            "move_forward_m": args.move_forward_m,
            "speed_mps": args.speed_mps,
            "soft_reset": args.soft_reset,
            "save_artifacts": args.save_artifacts,
            "require_visual": args.require_visual,
            "visual_timeout_sec": args.visual_timeout_sec,
        },
        "actions": [],
        "outputs": {},
        "visual_quality": {},
        "cleanup": {"attempted": False},
    }

    # The Server composes deployment control and Backend selection. The caller
    # still receives one stable EnvClient regardless of the two MapPackage axes.
    server = SimulationServer(
        packages,
        port_policy,
        run_dir=run_dir,
        execution_mode=execution,
        save_artifacts=args.save_artifacts,
    )
    client = SimulatorEnvClient(
        server,
        observation_profile=observation_profile,
    )
    scene: SceneInstance | None = None
    took_off = False

    try:
        scene = client.start_scene(package.map_id, package.scene_id)
        _write_json(run_dir / "scene_instance.json", scene.to_dict())
        report["scene"] = scene.to_dict()
        report["capabilities"] = _capabilities_dict(scene.capabilities)

        # connect is a common Backend action. AirSim and UnrealCV implement it
        # differently, but the demo never branches on the protocol name.
        _record_action(
            report,
            _require_ok(
                client.execute_action(
                    scene.scene_instance_id,
                    args.agent_id,
                    "connect",
                    {},
                )
            ),
        )

        before, quality = _observe(
            client,
            scene,
            args.agent_id,
            run_dir / "observation_before.json",
            list(observation_profile.modalities),
            require_visual=args.require_visual,
            timeout_sec=args.visual_timeout_sec,
        )
        report["visual_quality"]["before"] = quality
        report["outputs"]["observation_before"] = str(
            run_dir / "observation_before.json"
        )

        if args.move_forward_m > 0:
            capabilities = _require_capabilities(scene)

            # A Multirotor scene must take off before moving. ComputerVision
            # and UnrealCV camera-pose scenes explicitly advertise that no
            # airborne lifecycle exists, so sending a fake takeoff would be a
            # contract violation.
            if capabilities.requires_takeoff:
                took_off = True
                _record_action(
                    report,
                    _require_ok(
                        client.execute_action(
                            scene.scene_instance_id,
                            args.agent_id,
                            "takeoff",
                            {},
                        )
                    ),
                )

            duration = args.move_forward_m / args.speed_mps
            _record_action(
                report,
                _require_ok(
                    client.execute_action(
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
                ),
            )
            if capabilities.supports("hover"):
                _record_action(
                    report,
                    _require_ok(
                        client.execute_action(
                            scene.scene_instance_id,
                            args.agent_id,
                            "hover",
                            {},
                        )
                    ),
                )

            _, quality = _observe(
                client,
                scene,
                args.agent_id,
                run_dir / "observation_after.json",
                list(observation_profile.modalities),
                require_visual=args.require_visual,
                timeout_sec=args.visual_timeout_sec,
            )
            report["visual_quality"]["after"] = quality
            report["outputs"]["observation_after"] = str(
                run_dir / "observation_after.json"
            )

            # Restoring the initial pose makes an explicitly requested motion
            # reproducible and avoids leaving an attached simulator displaced.
            if capabilities.supports("set_pose"):
                _record_action(
                    report,
                    _require_ok(
                        client.execute_action(
                            scene.scene_instance_id,
                            args.agent_id,
                            "set_pose",
                            _set_pose_args(before.pose),
                        )
                    ),
                )
                restored, quality = _observe(
                    client,
                    scene,
                    args.agent_id,
                    run_dir / "observation_restored.json",
                    list(observation_profile.modalities),
                    require_visual=args.require_visual,
                    timeout_sec=args.visual_timeout_sec,
                )
                report["visual_quality"]["restored"] = quality
                report["outputs"]["observation_restored"] = str(
                    run_dir / "observation_restored.json"
                )
                report["restore_position_error_m"] = _position_distance(
                    before.pose, restored.pose
                )

        if args.soft_reset:
            capabilities = _require_capabilities(scene)
            if capabilities.supports_soft_reset:
                reset_scene = client.reset_scene(scene.scene_instance_id)
                report["soft_reset"] = {
                    "status": "completed",
                    "scene": reset_scene.to_dict(),
                }
                after_reset, quality = _observe(
                    client,
                    scene,
                    args.agent_id,
                    run_dir / "observation_after_reset.json",
                    list(observation_profile.modalities),
                    require_visual=args.require_visual,
                    timeout_sec=args.visual_timeout_sec,
                )
                report["outputs"]["observation_after_reset"] = str(
                    run_dir / "observation_after_reset.json"
                )
                report["visual_quality"]["after_reset"] = quality
            else:
                report["soft_reset"] = {
                    "status": "skipped",
                    "reason": (
                        f"{capabilities.protocol}/"
                        f"{capabilities.control_mode} does not advertise "
                        "supports_soft_reset"
                    ),
                }

        report["status"] = "completed"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        # Cleanup is intentionally ownership-aware. A managed scene may stop
        # its own UE process, while an attached scene only disconnects the SDK
        # client and closes its tunnel. The Demo never calls SSH or ProcessManager
        # directly and therefore cannot bypass those ownership checks.
        report["cleanup"]["attempted"] = scene is not None
        if scene is not None:
            try:
                capabilities = scene.capabilities
                if (
                    took_off
                    and capabilities is not None
                    and capabilities.supports("land")
                ):
                    landing = client.execute_action(
                        scene.scene_instance_id,
                        args.agent_id,
                        "land",
                        {},
                    )
                    _record_action(report, _require_ok(landing))
                ownership = scene.process_ownership
                client.stop_scene(scene.scene_instance_id)
                report["cleanup"].update(
                    {
                        "ok": True,
                        "process_ownership": ownership,
                        "meaning": (
                            "owned process stopped"
                            if ownership in {"local", "remote_managed"}
                            else "external process preserved; client resources closed"
                        ),
                    }
                )
            except Exception as cleanup_error:
                report["cleanup"].update(
                    {
                        "ok": False,
                        "error_type": type(cleanup_error).__name__,
                        "error": str(cleanup_error),
                    }
                )
        _write_json(run_dir / "execution_summary.json", report)


def _observe(
    client: SimulatorEnvClient,
    scene: SceneInstance,
    agent_id: str,
    output_path: Path,
    modalities: list[str],
    *,
    require_visual: bool,
    timeout_sec: float,
) -> tuple[RawObservation, dict[str, Any]]:
    """Capture one observation and optionally wait for meaningful artifacts.

    Strict visual checks belong to this acceptance workflow rather than the
    Backend.  A simulator may legitimately return a dark scene, while a real
    engine validation run must reject constant output to avoid false success.
    """

    deadline = time.monotonic() + timeout_sec
    last_quality: dict[str, Any] = {}
    while True:
        observation = client.get_observation(
            scene.scene_instance_id,
            agent_id,
            modalities,
        )

        # Backends may wrap primitive results as ``pose.pose`` or
        # ``frames.frames``. RawObservation is the existing project boundary
        # that removes those transport wrappers without interpreting the scene.
        raw = RawObservation.from_env_observation(
            observation, source="simulator_env_client_demo"
        )
        last_quality = _visual_quality(raw, modalities)
        if not require_visual or last_quality["ok"]:
            _write_json(output_path, raw.to_dict())
            return raw, last_quality
        if time.monotonic() >= deadline:
            raise AssertionError(
                "Visual observation remained degenerate within "
                f"{timeout_sec:.1f}s: {json.dumps(last_quality, ensure_ascii=False)}"
            )
        time.sleep(1.0)


def _visual_quality(
    observation: RawObservation,
    modalities: list[str],
) -> dict[str, Any]:
    """Measure saved RGB/depth artifacts without affecting Backend behavior."""

    import numpy as np
    from PIL import Image

    result: dict[str, Any] = {"ok": True, "frames": {}}
    requested = set(modalities)

    if "rgb" in requested:
        rgb = observation.frames.get("rgb", {})
        path = Path(str(rgb.get("artifact_path", "")))
        if not path.is_file():
            result["frames"]["rgb"] = {
                "ok": False,
                "reason": f"RGB artifact missing: {path}",
            }
            result["ok"] = False
        else:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            colors = np.unique(array.reshape(-1, 3), axis=0)
            stats = {
                "ok": bool(float(array.std()) > 1.0 and len(colors) > 16),
                "path": str(path),
                "shape": list(array.shape),
                "min": int(array.min()),
                "max": int(array.max()),
                "std": float(array.std()),
                "unique_colors": int(len(colors)),
            }
            result["frames"]["rgb"] = stats
            result["ok"] = bool(result["ok"] and stats["ok"])

    if "depth" in requested:
        depth = observation.frames.get("depth", {})
        path = Path(str(depth.get("artifact_path", "")))
        if not path.is_file():
            result["frames"]["depth"] = {
                "ok": False,
                "reason": f"Depth artifact missing: {path}",
            }
            result["ok"] = False
        else:
            values = np.asarray(np.load(path), dtype=np.float32)
            finite = bool(values.size and np.isfinite(values).all())
            spread = float(values.std()) if finite else 0.0
            stats = {
                "ok": bool(finite and spread > 1e-6),
                "path": str(path),
                "shape": list(values.shape),
                "finite": finite,
                "min": float(values.min()) if finite else None,
                "max": float(values.max()) if finite else None,
                "std": spread,
            }
            preview = depth.get("preview_path")
            if preview:
                stats["preview_path"] = str(preview)
            result["frames"]["depth"] = stats
            result["ok"] = bool(result["ok"] and stats["ok"])

    if not ({"rgb", "depth"} & requested):
        result["ok"] = False
        result["reason"] = "Strict visual validation requires rgb or depth"
    return result


def _set_pose_args(pose: dict[str, Any]) -> dict[str, float | bool]:
    """Convert canonical observation pose into the EnvClient set_pose contract."""

    position = pose.get("position", pose)
    euler_rad = pose.get("euler_rad", {})
    euler_deg = pose.get("euler_deg", {})
    return {
        "x": float(position.get("x", 0.0)),
        "y": float(position.get("y", 0.0)),
        "z": float(position.get("z", 0.0)),
        "pitch_deg": float(
            euler_deg.get(
                "pitch",
                math.degrees(float(euler_rad.get("pitch", 0.0))),
            )
        ),
        "roll_deg": float(
            euler_deg.get(
                "roll",
                math.degrees(float(euler_rad.get("roll", 0.0))),
            )
        ),
        "yaw_deg": float(
            pose.get(
                "yaw_deg",
                euler_deg.get(
                    "yaw",
                    math.degrees(float(euler_rad.get("yaw", 0.0))),
                ),
            )
        ),
        "ignore_collision": True,
    }


def _position_distance(
    first: dict[str, Any], second: dict[str, Any]
) -> float:
    """Return Euclidean position error for flat or nested canonical poses."""

    first_position = first.get("position", first)
    second_position = second.get("position", second)
    return math.sqrt(
        sum(
            (
                float(second_position.get(axis, 0.0))
                - float(first_position.get(axis, 0.0))
            )
            ** 2
            for axis in ("x", "y", "z")
        )
    )


def _require_capabilities(scene: SceneInstance) -> SceneCapabilities:
    """Return advertised capabilities or fail instead of guessing behavior."""

    if scene.capabilities is None:
        raise RuntimeError("Scene started without SceneCapabilities")
    return scene.capabilities


def _require_ok(result: ActionResult) -> ActionResult:
    """Turn a failed primitive result into one actionable Demo failure."""

    if not result.ok:
        raise RuntimeError(f"{result.action} failed: {result.message}")
    return result


def _record_action(report: dict[str, Any], result: ActionResult) -> None:
    """Append the stable EnvClient action result to the execution report."""

    report["actions"].append(result.to_dict())


def _select_package(packages, map_id: str):
    """Select one MapPackage and report available IDs on mistakes."""

    for package in packages:
        if package.map_id == map_id:
            return package
    available = ", ".join(package.map_id for package in packages)
    raise ValueError(f"Unknown map_id {map_id!r}; available: {available}")


def _configured_control_mode(package) -> str:
    """Describe the configured strategy without influencing Backend routing."""

    if package.control_protocol == "airsim":
        return str(package.airsim.get("mode", "Multirotor"))
    target = str(package.unrealcv.get("control_target", "camera")).lower()
    return "actor_pose" if target == "actor" else "camera_pose"


def _capabilities_dict(
    capabilities: SceneCapabilities | None,
) -> dict[str, Any] | None:
    """Serialize capability dataclass for the human-facing summary."""

    return asdict(capabilities) if capabilities is not None else None


def _new_run_dir(run_root: str | Path) -> Path:
    """Create one collision-resistant run directory for Demo artifacts."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON while accepting project dataclasses and enums."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    """Provide a narrow JSON fallback for Demo-owned report values."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def main() -> int:
    """Run the selected workflow and print the artifact location."""

    args = build_parser().parse_args()
    if args.list_maps:
        print(json.dumps(list_maps(args.config), indent=2, ensure_ascii=False))
        return 0
    try:
        report = run_demo(args)
    except Exception as exc:
        print(f"Simulator EnvClient demo failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Inspect: {Path(report['run_dir']) / 'execution_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
