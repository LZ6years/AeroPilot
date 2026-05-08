"""Command line entry point for FlyAgent runtime checks."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.map_config import build_remote_airsim_config, load_map_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.agent_runtime import AgentRuntime
from src.sim_server.server import LocalSimulationServer


def build_parser() -> argparse.ArgumentParser:
    # Build the CLI argument parser for smoke-test commands.
    parser = argparse.ArgumentParser(description="FlyAgent CLI")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--run-root", default="runs")
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
    parser.add_argument(
        "command",
        choices=["smoke-test", "remote-airsim-smoke-test", "remote-airsim-cv-smoke-test"],
    )
    return parser


def main() -> None:
    # Compose config, local C/S objects, and runtime command execution.
    args = build_parser().parse_args()
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
