"""Safely validate TaskRuntime initialization against a real simulator.

This probe exercises the same unified path for AirSim ComputerVision and
UnrealCV:

    MapPackage -> SimulationServer -> SimulatorEnvClient -> TaskRuntime

It intentionally uses a deterministic planner that completes without flight
actions. By default, the probe first reads the simulator's current canonical
pose and then asks TaskRuntime to apply that same pose as the episode initial
pose. That validates episode placement without moving the UAV somewhere new.

Examples:

    python tests/manual/task_runtime_episode_options_probe.py \
        --config configs/maps/remote_ssh.example.json \
        --map-id airsim_remote_env_1 \
        --observation-profile configs/observations/airsim_default.example.json \
        --execution managed

    python tests/manual/task_runtime_episode_options_probe.py \
        --config path/to/local_unrealcv_map.json \
        --map-id downtownwest_actor \
        --observation-profile configs/observations/unrealcv_default.example.json \
        --execution managed

Explicit ``--x --y --z`` values remain available for a deliberate episode
placement test. They must be supplied together. The probe never calls an LLM
and never issues move_body, turn_relative, takeoff, or land from its planner.
Runtime itself still performs capability-aware connection and safe preflight.
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

from src.env_client import SimulatorEnvClient, load_observation_profile
from src.runtime.configs import InitialPose, TaskExecutionOptions
from src.runtime.configs.map_config import load_map_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import SimulationServer
from src.uav import pose_value, yaw_deg


PROBE_INSTRUCTION = "前面右转会有一个黑色的路灯，右转面向路灯后停下"


from tests.manual.benchmark_probe_support import CompleteProbePlanner as CompleteWithoutFlightPlanner


def build_parser() -> argparse.ArgumentParser:
    """Build the command line without opening a simulator."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate TaskRuntime capabilities, episode pose, and observations "
            "on a real AirSim or UnrealCV scene."
        )
    )
    parser.add_argument("--config", required=True, help="MapPackage JSON path.")
    parser.add_argument("--map-id", required=True, help="MapPackage map_id.")
    parser.add_argument("--scene-id", help="Optional scene selection.")
    parser.add_argument(
        "--observation-profile",
        "--profile",
        dest="observation_profile",
        help="Run-level ObservationProfile JSON path.",
    )
    parser.add_argument(
        "--execution",
        choices=("attach", "managed"),
        default="managed",
        help="Use managed by default; attach never starts or stops an external UE.",
    )
    parser.add_argument(
        "--run-root",
        default="runs/manual_task_runtime_probe",
    )
    parser.add_argument("--agent-id", default="uav_1")
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--yaw-deg", type=float)
    parser.add_argument("--pitch-deg", type=float)
    parser.add_argument("--roll-deg", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.set_defaults(save_artifacts=True)
    parser.add_argument(
        "--save-artifacts",
        dest="save_artifacts",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-artifacts",
        dest="save_artifacts",
        action="store_false",
    )
    return parser


def _explicit_pose_from_args(args: argparse.Namespace) -> InitialPose | None:
    """Return an explicit pose, enforcing the all-or-none XYZ contract."""

    supplied = [args.x is not None, args.y is not None, args.z is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("--x, --y, and --z must be supplied together")
    if not all(supplied):
        return None
    return InitialPose(
        x=args.x,
        y=args.y,
        z=args.z,
        yaw_deg=args.yaw_deg or 0.0,
        pitch_deg=args.pitch_deg or 0.0,
        roll_deg=args.roll_deg or 0.0,
    )


def _initial_pose_from_observation(observation: dict[str, Any]) -> InitialPose:
    """Convert a canonical EnvClient pose into TaskExecutionOptions."""

    data = observation.get("data", observation)
    pose = data.get("pose", {}) if isinstance(data, dict) else {}
    x = pose_value(pose, "x")
    y = pose_value(pose, "y")
    z = pose_value(pose, "z")
    if x is None or y is None or z is None:
        raise RuntimeError(
            "Current observation does not contain a complete canonical pose"
        )
    return InitialPose(
        x=x,
        y=y,
        z=z,
        yaw_deg=yaw_deg(pose),
        pitch_deg=pose_value(pose, "pitch_deg") or 0.0,
        roll_deg=pose_value(pose, "roll_deg") or 0.0,
    )


def _capabilities_dict(scene) -> dict[str, Any] | None:
    """Serialize only deployment-neutral simulator capabilities."""

    capabilities = scene.capabilities
    if capabilities is None:
        return None
    return {
        "protocol": capabilities.protocol,
        "control_mode": capabilities.control_mode,
        "supported_actions": list(capabilities.supported_actions),
        "observation_modalities": list(
            capabilities.observation_modalities
        ),
        "requires_takeoff": capabilities.requires_takeoff,
        "supports_soft_reset": capabilities.supports_soft_reset,
    }


def main() -> int:
    """Run the real-engine initialization probe and always clean up its scene."""

    args = build_parser().parse_args()

    # Profile and TaskExecutionOptions validation happen before SimulationServer
    # starts UE. In particular, Runtime v1 requires pose in every profile.
    profile = load_observation_profile(args.observation_profile)
    explicit_pose = _explicit_pose_from_args(args)
    TaskExecutionOptions.from_observation_profile(
        profile,
        initial_pose=explicit_pose,
    )

    maps, port_policy = load_map_config(args.config)
    selected = next(
        (package for package in maps if package.map_id == args.map_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Unknown map_id: {args.map_id}")

    run_dir = RunStore(args.run_root).create_run_dir("task_runtime_probe")
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        execution_mode=args.execution,
        show_display=not args.headless,
        save_artifacts=args.save_artifacts,
    )
    client = SimulatorEnvClient(server, observation_profile=profile)
    runtime = TaskRuntime(
        client,
        recorder,
        planner=CompleteWithoutFlightPlanner(),
        shutdown_policy="none",
    )

    recorder.write_json("config/map_package.json", selected.to_dict())
    recorder.write_json("config/observation_profile.json", profile.to_dict())

    state = None
    scene = None
    current_observation: dict[str, Any] | None = None
    cleanup: dict[str, Any] = {
        "attempted": False,
        "completed": False,
        "error": None,
    }
    summary: dict[str, Any] | None = None
    exit_code = 1
    try:
        # Register a reusable scene with TaskRuntime before the direct pose
        # read. This ensures its finally path can safely close the exact same
        # SceneInstance, respecting attach versus managed ownership.
        scene = runtime.ensure_reused_scene(args.map_id, args.scene_id)
        recorder.write_json("state/scene_capabilities.json", _capabilities_dict(scene))

        initial_pose = explicit_pose
        if initial_pose is None:
            connect = client.execute_action(
                scene.scene_instance_id,
                args.agent_id,
                "connect",
                {},
            )
            if not connect.ok:
                raise RuntimeError(
                    f"Initial pose probe could not connect: {connect.message}"
                )
            current = client.get_observation(
                scene.scene_instance_id,
                args.agent_id,
                ["pose"],
            )
            current_observation = current.to_dict()
            initial_pose = _initial_pose_from_observation(current_observation)

        options = TaskExecutionOptions.from_observation_profile(
            profile,
            initial_pose=initial_pose,
        )
        recorder.write_json(
            "config/task_execution_options.json",
            options.to_dict(),
        )
        if current_observation is not None:
            recorder.write_json(
                "state/initial_pose_probe.json",
                current_observation,
            )

        state = runtime.run_task(
            args.map_id,
            PROBE_INSTRUCTION,
            agent_id=args.agent_id,
            scene_policy="reuse_running",
            stop_scene_on_finish=False,
            scene_id=args.scene_id,
            execution_options=options,
        )
        summary = {
            "status": "completed" if state.completed else "incomplete",
            "run_dir": str(Path(run_dir).resolve()),
            "map_id": args.map_id,
            "scene_id": args.scene_id,
            "process_ownership": scene.process_ownership,
            "observation_profile": profile.to_dict(),
            "scene_capabilities": _capabilities_dict(scene),
            "requested_options": options.to_dict(),
            "initial_pose_observation": current_observation,
            "final_pose": state.latest_observation.get("pose"),
            "cleanup": cleanup,
        }
        exit_code = 0 if state.completed else 1
    finally:
        cleanup["attempted"] = True
        try:
            runtime.finish_reused_scene(
                args.map_id,
                state,
                agent_id=args.agent_id,
                scene_id=args.scene_id,
            )
            cleanup["completed"] = True
        except Exception as exc:
            cleanup["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if summary is not None:
                recorder.write_json("summary.json", summary)
                print(json.dumps(summary, indent=2, ensure_ascii=False))
            recorder.write_json("cleanup.json", cleanup)
            recorder.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
