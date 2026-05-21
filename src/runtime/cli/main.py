"""Command line entry point for FlyAgent runtime checks."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.semantic.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.map_config import build_remote_airsim_config, load_map_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.agent_runtime import AgentRuntime
from src.runtime.runners.interactive_runtime import InteractiveTaskSession, compact_turn_summary
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


ANSI_BLUE = "\033[34m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    # Build the CLI argument parser for smoke-test commands.
    parser = argparse.ArgumentParser(description="FlyAgent CLI")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--env-path", default=".env", help="Environment file for model configuration")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--safety-config", default="configs/uav_safety_config.json")
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
    parser.add_argument("--scene-id", default=None, help="Scene id for chat mode")
    parser.add_argument("--agent-id", default="uav_1", help="Agent id for chat mode")
    parser.add_argument("--no-launch", action="store_true", help="Do not launch the configured scene process in chat mode")
    parser.add_argument("--headless", action="store_true", help="Hide the engine display when the map supports it")
    parser.set_defaults(save_artifacts=True)
    parser.add_argument("--save-artifacts", action="store_true", help="Save RGB/depth artifacts under the run dir")
    parser.add_argument("--no-save-artifacts", dest="save_artifacts", action="store_false", help="Do not save RGB/depth artifacts")
    parser.add_argument("--stop-on-error", action="store_true", help="Exit chat after an instruction fails")
    parser.add_argument("--ready-timeout-sec", type=int, default=30, help="Remote endpoint readiness timeout")
    parser.add_argument("--instruction", default=None, help="User instruction for parse-task")
    parser.add_argument(
        "--parser-mode",
        choices=["model", "rule_based"],
        default=None,
        help="Override FLYAGENT_TASK_PARSER_MODE for parse-task",
    )
    parser.add_argument(
        "--shutdown-policy",
        choices=["hover", "land", "none"],
        default="hover",
        help="Final task shutdown action before stopping the scene",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="chat",
        choices=[
            "smoke-test",
            "remote-airsim-smoke-test",
            "remote-airsim-cv-smoke-test",
            "parse-task",
            "run-task",
            "chat",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Compose config, local C/S objects, and runtime command execution.
    args = build_parser().parse_args(argv)
    if args.command == "chat":
        return run_chat(args)
    if args.command == "parse-task":
        run_parse_task(args)
        return 0
    if args.command == "run-task":
        run_task(args)
        return 0
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
            return 0
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
    return 0


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
        recorder.write_json("state/semantic_plan.json", payload)
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
            planner="ModelActionPlanner",
            shutdown_policy=args.shutdown_policy,
        )
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        recorder.write_json(
            "config/model_config.json",
            {
                "model_config": model_config.redacted_summary(),
                "parser_mode": parser_mode,
                "planner": "ModelActionPlanner",
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
            planner=ModelActionPlanner(model_config),
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


def run_chat(args: argparse.Namespace) -> int:
    # Run an interactive natural-language session while reusing one scene.
    args.map_id = select_chat_map_id(args)
    session = create_chat_session(args)
    if not args.no_launch:
        print(color_text(f"Opening map: {args.map_id}", ANSI_DIM))
        print_json(session.start_scene())
    print_chat_banner(args, session)
    try:
        return chat_repl(args, session)
    finally:
        if not session.closed:
            session.shutdown()


def create_chat_session(args: argparse.Namespace) -> InteractiveTaskSession:
    maps, port_policy = load_map_config(args.config)
    map_id = args.map_id or default_chat_map_id(maps)
    selected_map = next((pkg for pkg in maps if pkg.map_id == map_id), None)
    if selected_map is None:
        known = ", ".join(pkg.map_id for pkg in maps)
        raise ValueError(f"Unknown map_id {map_id!r}. Available maps: {known}")
    model_config = load_model_config(args.env_path)
    action_config = load_uav_action_config(args.action_config)
    safety_config = load_uav_safety_config(args.safety_config)
    launch_processes = not args.no_launch and not args.connect_existing
    real_control = launch_processes or args.connect_existing
    parser_mode = args.parser_mode or "model"
    run_dir = RunStore(args.run_root).create_run_dir("interactive_cli")
    recorder = AsyncRunLogger(run_dir)
    server = LocalSimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        launch_processes=launch_processes,
        real_control=real_control,
        show_display=not args.headless,
        action_config=action_config,
        save_artifacts=args.save_artifacts,
    )
    recorder.write_json("config/map_package.json", selected_map.to_dict())
    recorder.write_json("config/uav_action_config.json", action_config.to_dict())
    recorder.write_json("config/uav_safety_config.json", safety_config.to_dict())
    recorder.write_json(
        "config/model_config.json",
        {
            "model_config": model_config.redacted_summary(),
            "parser_mode": parser_mode,
            "planner": "ModelActionPlanner",
            "shutdown_policy": args.shutdown_policy,
            "save_artifacts": args.save_artifacts,
            "interactive_cli": True,
        },
    )
    runtime = TaskRuntime(
        LocalEnvClient(server),
        recorder,
        TaskParser(mode=parser_mode, model_config=model_config),
        planner=ModelActionPlanner(model_config),
        safety=SafetySupervisor(safety_config),
        shutdown_policy=args.shutdown_policy,
        action_config=action_config,
    )
    return InteractiveTaskSession(
        runtime,
        map_id,
        agent_id=args.agent_id,
        scene_id=args.scene_id,
        available_map_ids=[pkg.map_id for pkg in maps],
    )


def select_chat_map_id(args: argparse.Namespace) -> str:
    maps, _ = load_map_config(args.config)
    if args.map_id:
        return args.map_id
    default_id = default_chat_map_id(maps)
    print("Available maps:")
    for index, package in enumerate(maps, start=1):
        marker = " (default)" if package.map_id == default_id else ""
        print(f"  {index}. {package.map_id}{marker}")
    if not sys.stdin.isatty():
        print(f"Using default map_id={default_id}")
        return default_id
    choice = input(f"Select map by number or id [default: {default_id}]: ").strip()
    if not choice:
        return default_id
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(maps):
            return maps[index - 1].map_id
    known = {package.map_id for package in maps}
    if choice in known:
        return choice
    raise ValueError(f"Unknown map selection {choice!r}; available maps: {', '.join(sorted(known))}")


def default_chat_map_id(maps) -> str:
    known = [package.map_id for package in maps]
    return "cityenviron_win64" if "cityenviron_win64" in known else known[0]


def chat_repl(args: argparse.Namespace, session: InteractiveTaskSession) -> int:
    while True:
        try:
            text = input(color_text("flyagent> ", ANSI_BLUE))
        except EOFError:
            print()
            session.shutdown()
            return 0
        command = text.strip()
        if not command:
            continue
        if command.startswith("/"):
            should_continue = handle_slash_command(command, session)
            if not should_continue:
                return 0
            continue
        summary = session.submit_instruction(command)
        print_summary(summary)
        if args.stop_on_error and summary.get("error"):
            session.shutdown()
            return 1


def handle_slash_command(command: str, session: InteractiveTaskSession) -> bool:
    name = command.split(maxsplit=1)[0].lower()
    if name in {"/quit", "/exit", "/q"}:
        print_json(session.shutdown())
        return False
    if name == "/help":
        print_chat_help()
        return True
    if name == "/clear":
        clear_screen()
        return True
    if name == "/status":
        print_json(session.status())
        return True
    if name == "/summary":
        print_json(compact_turn_summary(session.last_summary))
        return True
    if name == "/context":
        print_json(
            {
                "latest_context": str(session.run_dir / "contexts" / "latest_context.json"),
                "latest_phase_transition_context": str(session.run_dir / "contexts" / "latest_phase_transition_context.json"),
                "latest_state": str(session.run_dir / "state" / "latest_state.json"),
                "flight_memory": str(session.run_dir / "state" / "flight_memory.json"),
                "semantic_plan": str(session.run_dir / "state" / "semantic_plan.json"),
                "scene_instance": str(session.run_dir / "state" / "scene_instance.json"),
                "latest_observation_debug": str(session.run_dir / "debug" / "latest_observation.json"),
                "debug_events": str(session.run_dir / "debug" / "debug_events.jsonl"),
            }
        )
        return True
    if name == "/maps":
        print_json({"current_map_id": session.map_id, "available_map_ids": list(session.available_map_ids)})
        return True
    if name == "/use":
        print("Map selection happens before the flight session starts. Use /quit and restart the CLI to switch maps.")
        return True
    if name == "/hover":
        print_json(session.hover())
        return True
    if name == "/land":
        print_json(session.shutdown("land"))
        return False
    print(f"Unknown command: {command}")
    print_chat_help()
    return True


def print_chat_banner(args: argparse.Namespace, session: InteractiveTaskSession) -> None:
    parser_mode = args.parser_mode or "model"
    map_id = args.map_id or session.map_id
    print("FlyAgent interactive CLI")
    print(f"map_id={map_id}, parser_mode={parser_mode}, shutdown_policy={args.shutdown_policy}")
    print(f"launch_processes={not args.no_launch and not args.connect_existing}, save_artifacts={args.save_artifacts}")
    print(f"run_dir={session.run_dir}")
    print_chat_help()


def print_chat_help() -> None:
    lines = [
        "FlyAgent commands:",
        "  /help     show this help",
        "  /clear    clear the terminal screen",
        "  /status   show current session status",
        "  /summary  show the latest instruction summary",
        "  /context  show debug artifact paths",
        "  /maps     show available maps",
        "  /hover    send a best-effort hover command",
        "  /land     land and exit",
        "  /quit     exit with the configured shutdown policy",
        "Natural language example: 向前飞行5米，然后悬停",
    ]
    for line in lines:
        print(color_text(line, ANSI_DIM))


def print_summary(summary: dict[str, Any]) -> None:
    print_json(compact_turn_summary(summary))


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def color_text(text: str, color: str) -> str:
    # Keep redirected output and tests plain while making the interactive shell easier to scan.
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{ANSI_RESET}"


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
        return
    print("")


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
    raise SystemExit(main())

