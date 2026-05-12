"""Command line entry point for FlyAgent runtime checks."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from src.agent.model_action_planner import ModelActionPlanner
from src.agent.short_horizon_planner import ShortHorizonPlanner
from src.agent.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.map_config import build_remote_airsim_config, load_map_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.agent_runtime import AgentRuntime
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


def build_parser() -> argparse.ArgumentParser:
    # Build the CLI argument parser for smoke-test commands.
    parser = argparse.ArgumentParser(description="FlyAgent CLI")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--env-path", default=".env", help="Environment file for model configuration")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--launch", action="store_true", help="Actually launch scene process")
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Use real control protocol against an already running UE process",
    )
    parser.add_argument("--remote-host", default=None, help="Remote AirSim host for remote-airsim-smoke-test")
    parser.add_argument("--remote-port", type=int, default=41451, help="Remote AirSim RPC port")
    parser.add_argument("--vehicle-name", default="Drone_1", help="AirSim vehicle name")
    parser.add_argument("--camera-name", default="front_0", help="AirSim camera name")
    parser.add_argument("--ready-timeout-sec", type=int, default=30, help="Remote endpoint readiness timeout")
    parser.add_argument("--instruction", default=None, help="User instruction for parse-task")
    parser.add_argument(
        "--parser-mode",
        choices=["model", "rule_based"],
        default=None,
        help="Override FLYAGENT_TASK_PARSER_MODE for parse-task",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["model", "rule_based"],
        default="rule_based",
        help="Short-horizon planner mode for run-task",
    )
    parser.add_argument(
        "--shutdown-policy",
        choices=["hover", "land", "none"],
        default="hover",
        help="Final task shutdown action before stopping the scene",
    )
    parser.add_argument(
        "command",
        choices=[
            "smoke-test",
            "remote-airsim-smoke-test",
            "remote-airsim-cv-smoke-test",
            "parse-task",
            "run-task",
        ],
    )
    return parser


def main() -> None:
    # Compose config, local C/S objects, and runtime command execution.
    args = build_parser().parse_args()
    if args.command == "parse-task":
        run_parse_task(args)
        return
    if args.command == "run-task":
        run_task(args)
        return
    if args.command in {"remote-airsim-smoke-test", "remote-airsim-cv-smoke-test"}:
        if not args.remote_host:
            raise ValueError(f"{args.command} requires --remote-host")
        maps, port_policy = build_remote_airsim_config(
            args.remote_host,
            args.remote_port,
            args.vehicle_name,
            args.camera_name,
            args.ready_timeout_sec,
            "ComputerVision" if args.command == "remote-airsim-cv-smoke-test" else "Multirotor",
        )
        map_id = "remote_linux_airsim"
    else:
        maps, port_policy = load_map_config(args.config)
        map_id = args.map_id or maps[0].map_id
    run_dir = RunStore(args.run_root).create_run_dir("smoke")
    recorder = AsyncRunLogger(run_dir)
    selected_map = next(pkg for pkg in maps if pkg.map_id == map_id)
    try:
        recorder.info("run started", command=args.command, map_id=map_id)
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        if args.command in {"remote-airsim-smoke-test", "remote-airsim-cv-smoke-test"}:
            probe = probe_tcp_endpoint(args.remote_host, args.remote_port, args.ready_timeout_sec)
            recorder.write_json("remote_probe.json", probe)
            if not probe["reachable"]:
                recorder.event("run_failed", {"error_type": "ConnectionError", "error": probe["error"]})
                raise ConnectionError(
                    f"Remote AirSim endpoint is not reachable: "
                    f"{args.remote_host}:{args.remote_port}; {probe['error']}"
                )
        server = LocalSimulationServer(
            maps,
            port_policy,
            Path(run_dir),
            launch_processes=args.launch,
            real_control=args.launch
            or args.connect_existing
            or args.command in {"remote-airsim-smoke-test", "remote-airsim-cv-smoke-test"},
        )
        runtime = AgentRuntime(LocalEnvClient(server), recorder)
        try:
            if args.command == "remote-airsim-cv-smoke-test":
                runtime.run_observation_smoke_test(map_id)
            else:
                runtime.run_smoke_test(map_id)
            recorder.info("run completed", run_dir=str(run_dir))
            print(f"Smoke test complete. Run record: {run_dir}")
        except ModuleNotFoundError as exc:
            recorder.event(
                "run_failed",
                {
                    "error": str(exc),
                    "hint": "Install AirSim support with: pip install -r requirements.txt; "
                    "pip install --no-build-isolation -r requirements-airsim.txt",
                },
            )
            raise
        except Exception as exc:
            recorder.event("run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
            recorder.error("run failed", error_type=type(exc).__name__, error=str(exc))
            raise
    finally:
        recorder.close()


def run_parse_task(args: argparse.Namespace) -> None:
    # Parse a user instruction into a semantic plan without touching any environment.
    if not args.instruction:
        raise ValueError("parse-task requires --instruction")
    run_dir = RunStore(args.run_root).create_run_dir("parse_task")
    recorder = AsyncRunLogger(run_dir)
    model_config = load_model_config(args.env_path)
    parser_mode = args.parser_mode or model_config.task_parser_mode
    try:
        recorder.info("parse task started", parser_mode=parser_mode)
        recorder.write_json(
            "config/model_config.json",
            {"model_config": model_config.redacted_summary(), "parser_mode": parser_mode},
        )
        parser = TaskParser(mode=parser_mode, model_config=model_config)
        result = parser.parse(args.instruction)
        payload = result.to_dict()
        recorder.write_json("semantic_plan.json", payload)
        recorder.event("task_parsed", payload)
        recorder.info("parse task completed", run_dir=str(run_dir))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"Run record: {run_dir}")
    except Exception as exc:
        recorder.event("task_parse_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        recorder.error("parse task failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        recorder.close()


def run_task(args: argparse.Namespace) -> None:
    # Run one M4 short-horizon task loop through EnvClient and ToolDispatcher.
    if not args.instruction:
        raise ValueError("run-task requires --instruction")
    maps, port_policy = load_map_config(args.config)
    map_id = args.map_id or maps[0].map_id
    run_dir = RunStore(args.run_root).create_run_dir("task")
    recorder = AsyncRunLogger(run_dir)
    model_config = load_model_config(args.env_path)
    parser_mode = args.parser_mode or model_config.task_parser_mode
    action_config = load_uav_action_config(args.action_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == map_id)
    try:
        recorder.info(
            "task run started",
            map_id=map_id,
            parser_mode=parser_mode,
            planner_mode=args.planner_mode,
            shutdown_policy=args.shutdown_policy,
        )
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        recorder.write_json(
            "config/model_config.json",
            {
                "model_config": model_config.redacted_summary(),
                "parser_mode": parser_mode,
                "planner_mode": args.planner_mode,
                "shutdown_policy": args.shutdown_policy,
            },
        )
        recorder.write_json("config/uav_action_config.json", action_config.to_dict())
        server = LocalSimulationServer(
            maps,
            port_policy,
            Path(run_dir),
            launch_processes=args.launch,
            real_control=args.launch or args.connect_existing,
            action_config=action_config,
        )
        runtime = TaskRuntime(
            LocalEnvClient(server),
            recorder,
            TaskParser(mode=parser_mode, model_config=model_config),
            planner=build_planner(args.planner_mode, model_config, action_config),
            shutdown_policy=args.shutdown_policy,
        )
        state = runtime.run_task(map_id, args.instruction)
        recorder.info("task run completed", run_dir=str(run_dir), completed=state.completed)
        print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
        print(f"Run record: {run_dir}")
    except Exception as exc:
        recorder.event("task_run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        recorder.error("task run failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        recorder.close()


def build_planner(planner_mode: str, model_config, action_config):
    if planner_mode == "rule_based":
        return ShortHorizonPlanner(action_config)
    if planner_mode == "model":
        return ModelActionPlanner(model_config)
    raise ValueError(f"Unsupported planner mode: {planner_mode}")


def probe_tcp_endpoint(host: str, port: int, timeout_sec: int) -> dict:
    # Check TCP reachability before the AirSim client performs protocol calls.
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return {"host": host, "port": port, "reachable": True}
    except OSError as exc:
        return {
            "host": host,
            "port": port,
            "reachable": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


if __name__ == "__main__":
    main()
