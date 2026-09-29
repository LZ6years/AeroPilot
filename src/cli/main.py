"""Command line entry point for FlyAgent.

This module is the user-facing shell for FlyAgent. It owns command parsing and
interactive terminal behavior, while the runtime package owns task execution,
scene control, memory updates, action planning, safety, and observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.cli.display import ANSI_BLUE, ANSI_DIM, CliNarrativePanel, clear_screen, color_text, print_json, print_summary
from src.env_client import load_observation_profile
from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs import TaskExecutionOptions
from src.uav.config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.agent.config import load_model_config
from src.uav.config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.task.interactive_session import InteractiveTaskSession, compact_turn_summary
from src.runtime.runners.task_runtime import TaskRuntime
from src.runtime.factory import RuntimeFactory
from src.sim_server.server import SimulationServer


def build_parser() -> argparse.ArgumentParser:
    # Keep one parser for the user-facing chat command and engineering checks so
    # `flyagent`, `flyagent start`, and `python -m src.cli ...` share behavior.
    parser = argparse.ArgumentParser(description="FlyAgent CLI")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--env-path", default=".env", help="Environment file for model configuration")
    parser.add_argument("--map-id", default=None)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--safety-config", default="configs/uav_safety_config.json")
    parser.add_argument(
        "--observation-profile",
        default=None,
        help=(
            "Run-level observation profile JSON. The same profile controls "
            "EnvClient capture sources and TaskRuntime modalities."
        ),
    )
    parser.add_argument(
        "--execution",
        choices=["attach", "managed"],
        default="managed",
        help=(
            "Simulator process ownership: attach only connects to an existing "
            "UE; managed may start and stop an owned UE"
        ),
    )
    parser.add_argument("--scene-id", default=None, help="Scene id for chat mode")
    parser.add_argument("--agent-id", default="uav_1", help="Agent id for chat mode")
    parser.add_argument(
        "--defer-scene-start",
        action="store_true",
        help="Open chat without creating its scene until the session needs it",
    )
    parser.add_argument("--headless", action="store_true", help="Hide the engine display when the map supports it")
    parser.set_defaults(save_artifacts=True)
    parser.add_argument("--save-artifacts", action="store_true", help="Save RGB/depth artifacts under the run dir")
    parser.add_argument("--no-save-artifacts", dest="save_artifacts", action="store_false", help="Do not save RGB/depth artifacts")
    parser.add_argument("--stop-on-error", action="store_true", help="Exit chat after an instruction fails")
    parser.add_argument("--instruction", default=None, help="User instruction for parse-task")
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
            "start",
            "smoke-test",
            "observation-smoke-test",
            "run-task",
            "chat",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Compose config, local C/S objects, and runtime command execution.
    args = build_parser().parse_args(argv)
    if args.command in {"chat", "start"}:
        return run_chat(args)
    if args.command == "run-task":
        run_task(args)
        return 0
    maps, port_policy = load_map_config(args.config)
    observation_profile = load_observation_profile(args.observation_profile)
    execution_options = TaskExecutionOptions.from_observation_profile(
        observation_profile
    )
    map_id = args.map_id or maps[0].map_id
    run_dir = RunStore(args.run_root).create_run_dir("smoke")
    recorder = AsyncRunLogger(run_dir)
    selected_map = next(pkg for pkg in maps if pkg.map_id == map_id)
    try:
        recorder.info("run started", command=args.command, map_id=map_id)
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        _write_runtime_environment_config(
            recorder,
            args,
            observation_profile,
            execution_options,
            map_id,
        )
        server = SimulationServer(
            maps,
            port_policy,
            Path(run_dir),
            execution_mode=args.execution,
            show_display=not args.headless,
            save_artifacts=args.save_artifacts,
        )
        runtime = RuntimeFactory().create_smoke_runtime(
            SimulatorEnvClient(
                server,
                observation_profile=observation_profile,
            ),
            recorder,
            observation_modalities=execution_options.observation_modalities,
        )
        try:
            if args.command == "observation-smoke-test":
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


def run_task(args: argparse.Namespace) -> None:
    # Run one M4 short-horizon task loop through EnvClient and ToolDispatcher.
    if not args.instruction:
        raise ValueError("run-task requires --instruction")
    maps, port_policy = load_map_config(args.config)
    observation_profile = load_observation_profile(args.observation_profile)
    execution_options = TaskExecutionOptions.from_observation_profile(
        observation_profile
    )
    map_id = args.map_id or maps[0].map_id
    run_dir = RunStore(args.run_root).create_run_dir("task")
    recorder = AsyncRunLogger(run_dir)
    model_config = load_model_config(args.env_path)
    action_config = load_uav_action_config(args.action_config)
    safety_config = load_uav_safety_config(args.safety_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == map_id)
    try:
        recorder.info(
            "task run started",
            map_id=map_id,

            planner="ModelActionPlanner",
            shutdown_policy=args.shutdown_policy,
        )
        recorder.write_json("config/map_package.json", selected_map.to_dict())
        recorder.write_json(
            "config/model_config.json",
            {
                "model_config": model_config.redacted_summary(),
                "planner": "ModelActionPlanner",
                "shutdown_policy": args.shutdown_policy,
            },
        )
        recorder.write_json("config/uav_action_config.json", action_config.to_dict())
        recorder.write_json("config/uav_safety_config.json", safety_config.to_dict())
        _write_runtime_environment_config(
            recorder,
            args,
            observation_profile,
            execution_options,
            map_id,
        )
        server = SimulationServer(
            maps,
            port_policy,
            Path(run_dir),
            execution_mode=args.execution,
            show_display=not args.headless,
            action_config=action_config,
            save_artifacts=args.save_artifacts,
        )
        runtime = RuntimeFactory().create_task_runtime(
            SimulatorEnvClient(
                server,
                observation_profile=observation_profile,
            ),
            recorder,
            planner=ModelActionPlanner(model_config),
            safety=SafetySupervisor(safety_config),
            shutdown_policy=args.shutdown_policy,
            model_config=model_config,
        )
        state = runtime.run_task(
            map_id,
            args.instruction,
            agent_id=args.agent_id,
            scene_id=args.scene_id,
            execution_options=execution_options,
        )
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
    if not args.defer_scene_start:
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
    observation_profile = load_observation_profile(args.observation_profile)
    execution_options = TaskExecutionOptions.from_observation_profile(
        observation_profile
    )
    run_dir = RunStore(args.run_root).create_run_dir("interactive_cli")
    recorder = AsyncRunLogger(run_dir)
    live_printer = CliNarrativePanel()
    server = SimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        execution_mode=args.execution,
        show_display=not args.headless,
        action_config=action_config,
        save_artifacts=args.save_artifacts,
    )
    recorder.write_json("config/map_package.json", selected_map.to_dict())
    recorder.write_json("config/uav_action_config.json", action_config.to_dict())
    recorder.write_json("config/uav_safety_config.json", safety_config.to_dict())
    _write_runtime_environment_config(
        recorder,
        args,
        observation_profile,
        execution_options,
        map_id,
    )
    recorder.write_json(
        "config/model_config.json",
        {
            "model_config": model_config.redacted_summary(),
            "planner": "ModelActionPlanner",
            "shutdown_policy": args.shutdown_policy,
            "save_artifacts": args.save_artifacts,
            "interactive_cli": True,
        },
    )
    runtime = RuntimeFactory().create_task_runtime(
        SimulatorEnvClient(
            server,
            observation_profile=observation_profile,
        ),
        recorder,
        planner=ModelActionPlanner(model_config),
        safety=SafetySupervisor(safety_config),
        shutdown_policy=args.shutdown_policy,
        event_sink=live_printer,
        model_config=model_config,
    )
    return InteractiveTaskSession(
        runtime,
        map_id,
        agent_id=args.agent_id,
        scene_id=args.scene_id,
        available_map_ids=[pkg.map_id for pkg in maps],
        event_sink=live_printer,
        execution_options=execution_options,
    )


def _write_runtime_environment_config(
    recorder,
    args: argparse.Namespace,
    observation_profile,
    execution_options: TaskExecutionOptions,
    map_id: str,
) -> None:
    """Persist the one resolved environment contract shared by CLI layers.

    Keeping this snapshot beside the map package makes it possible to explain
    both halves of an observation later: the profile selected the simulator
    capture source, while TaskExecutionOptions selected what Runtime requested.
    """

    recorder.write_json(
        "config/observation_profile.json",
        observation_profile.to_dict(),
    )
    recorder.write_json(
        "config/runtime_environment.json",
        {
            "map_id": map_id,
            "scene_id": args.scene_id,
            "agent_id": args.agent_id,
            "execution": args.execution,
            "show_display": not args.headless,
            "save_artifacts": args.save_artifacts,
            "task_execution_options": execution_options.to_dict(),
        },
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
        event_sink = getattr(session, "event_sink", None)
        if hasattr(event_sink, "finish_turn"):
            event_sink.finish_turn(summary)
        else:
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
        print_summary(session.last_summary or {}, include_messages=True)
        return True
    if name == "/context":
        print_json(
            {
                "latest_checkpoint": str(session.run_dir / "contexts" / "latest_checkpoint.json"),
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
    map_id = args.map_id or session.map_id
    print("FlyAgent interactive CLI")
    print(f"map_id={map_id}, shutdown_policy={args.shutdown_policy}")
    print(
        f"execution={args.execution}, "
        f"defer_scene_start={args.defer_scene_start}, "
        f"save_artifacts={args.save_artifacts}"
    )
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
        "Natural language example: \u5411\u524d\u98de\u884c5\u7c73\uff0c\u7136\u540e\u60ac\u505c",
    ]
    for line in lines:
        print(color_text(line, ANSI_DIM))
if __name__ == "__main__":
    raise SystemExit(main())

