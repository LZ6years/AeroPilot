"""Real-engine probe for TaskRuntime episode pose and observation options.

Run this against an AirSim ComputerVision map after the UE scene is available:

    python tests/manual/task_runtime_episode_options_probe.py \
        --config configs/maps/simulator_control_demo.example.json \
        --map-id demo_local_airsim_cv \
        --profile configs/observations/airsim_default.example.json \
        --x 12 --y -4 --z 8 --yaw-deg 90

The probe uses a deterministic completion planner so it tests Runtime
initialization and observation plumbing without requiring a model call. Inspect
the printed first/final pose plus ``debug/debug_events.jsonl`` under the run
directory for ``episode_initial_pose_*`` ordering and capture artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.phase import COMPLETE_TASK, PhaseTransition
from src.agent.planning.schemas import CoupledActionPhasePlan
from src.agent.semantic.task_parser import TaskParser
from src.env_client import SimulatorEnvClient, load_observation_profile
from src.runtime.configs import InitialPose, TaskExecutionOptions
from src.runtime.configs.map_config import load_map_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import SimulationServer


class CompleteWithoutFlightPlanner:
    """Complete semantic phases without issuing Agent flight actions."""

    planner_mode = "manual_probe"

    def plan_next_with_phase_decision(self, state, context=None):
        return CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=PhaseTransition(
                COMPLETE_TASK,
                "manual probe validates initialization only",
            ),
            steps=[],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TaskRuntime initial_pose and observation modalities on real AirSim ComputerVision."
    )
    parser.add_argument("--config", required=True, help="MapPackage JSON path.")
    parser.add_argument("--map-id", required=True, help="ComputerVision map id.")
    parser.add_argument("--profile", help="ObservationProfile JSON path.")
    parser.add_argument("--execution", choices=("attach", "managed"), default="attach")
    parser.add_argument("--run-root", default="runs/manual_task_execution_options")
    parser.add_argument("--agent-id", default="uav_1")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    maps, port_policy = load_map_config(args.config)
    selected = next(
        (package for package in maps if package.map_id == args.map_id),
        None,
    )
    if selected is None:
        raise ValueError(f"Unknown map_id: {args.map_id}")
    mode = str(selected.airsim.get("mode", "")).strip().lower()
    if mode not in {"computervision", "computer_vision", "cv"}:
        raise ValueError(
            "This probe requires an AirSim ComputerVision MapPackage"
        )

    profile = load_observation_profile(args.profile)
    initial_pose = InitialPose(
        x=args.x,
        y=args.y,
        z=args.z,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )
    options = TaskExecutionOptions.from_observation_profile(
        profile,
        initial_pose=initial_pose,
    )
    run_dir = RunStore(args.run_root).create_run_dir(
        "task_execution_options_probe"
    )
    recorder = AsyncRunLogger(run_dir)
    server = SimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        execution_mode=args.execution,
        show_display=not args.headless,
    )
    runtime = TaskRuntime(
        SimulatorEnvClient(server, observation_profile=profile),
        recorder,
        TaskParser(mode="rule_based"),
        planner=CompleteWithoutFlightPlanner(),
        shutdown_policy="hover",
    )

    try:
        state = runtime.run_task(
            args.map_id,
            "保持当前位置并完成初始化检查",
            agent_id=args.agent_id,
            execution_options=options,
        )
        summary = {
            "ok": state.completed,
            "run_dir": str(Path(run_dir).resolve()),
            "requested_options": options.to_dict(),
            "final_observation": state.latest_observation,
            "inspect": [
                "debug/debug_events.jsonl",
                "debug/latest_observation.json",
                "state/flight_memory.json",
            ],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if state.completed else 1
    finally:
        recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
