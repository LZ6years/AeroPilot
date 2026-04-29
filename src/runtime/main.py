"""Command line entry point for FlyAgent runtime checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.env_client.local_client import LocalEnvClient
from src.runtime.agent_runtime import AgentRuntime
from src.runtime.config import load_map_config
from src.runtime.record_writer import RecordWriter
from src.runtime.run_store import RunStore
from src.sim_server.server import LocalSimulationServer


def build_parser() -> argparse.ArgumentParser:
    # Build the CLI argument parser for smoke-test commands.
    parser = argparse.ArgumentParser(description="FlyAgent CLI")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--launch", action="store_true", help="Actually launch scene process")
    parser.add_argument("command", choices=["smoke-test"])
    return parser


def main() -> None:
    # Compose config, local C/S objects, and runtime command execution.
    args = build_parser().parse_args()
    maps, port_policy = load_map_config(args.config)
    map_id = args.map_id or maps[0].map_id
    run_dir = RunStore(args.run_root).create_run_dir("smoke")
    recorder = RecordWriter(run_dir)
    selected_map = next(pkg for pkg in maps if pkg.map_id == map_id)
    recorder.write_json("map_package.json", selected_map.to_dict())
    server = LocalSimulationServer(maps, port_policy, Path(run_dir), launch_processes=args.launch)
    runtime = AgentRuntime(LocalEnvClient(server), recorder)
    if args.command == "smoke-test":
        runtime.run_smoke_test(map_id)
        print(f"Smoke test complete. Run record: {run_dir}")


if __name__ == "__main__":
    main()
