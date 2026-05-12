"""Manual demo for the current FlyAgent M3-M6 task pipeline.

This file is not a pytest unit test. It is a teaching/demo script that shows
what has been built so far and how the pieces connect:

    M3: natural language -> SemanticPlan
    M4: SemanticPlan -> runtime phase loop with preflight
    M5: raw observation -> model-ready context JSON
    M6: context -> short-horizon action plan -> SafetySupervisor -> tools

Default offline run, no .env/model/AirSim required:

    python test_agent_pipeline_demo.py

Real model parser and real model action planner:

    $env:PYTHONIOENCODING='utf-8'
    python test_agent_pipeline_demo.py --parser-mode model --planner-mode model --env-path .env

Run one custom instruction:

    python test_agent_pipeline_demo.py --instruction "左转寻找建筑入口，并停在入口前"

Only show M3 semantic parsing:

    python test_agent_pipeline_demo.py --mode parse-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agent.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.short_horizon_planner import ShortHorizonPlanner
from src.agent.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.env_client.schemas import MapPackage, PortPolicy
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


DEFAULT_INSTRUCTION = "前面右转会有一个黑色的路灯，右转面向路灯后停下"
INSTRUCTION_EXAMPLES = [
    DEFAULT_INSTRUCTION,
    "左转寻找建筑入口，并停在入口前",
    "前方寻找红色车辆，靠近但保持安全距离",
    "观察右侧区域，如果目标不明确就悬停等待确认",
    "沿着道路向前观察，找到蓝色指示牌后面向它悬停",
    "Search ahead for a white delivery truck, face it, and hover.",
]

HELP_EPILOG = """
Common commands:

  Offline full pipeline:
    python test_agent_pipeline_demo.py

  Offline parse only:
    python test_agent_pipeline_demo.py --mode parse-only

  Real model parse only:
    $env:PYTHONIOENCODING='utf-8'
    python test_agent_pipeline_demo.py --mode parse-only --parser-mode model --env-path .env

  Real model parse + real model action planning:
    $env:PYTHONIOENCODING='utf-8'
    python test_agent_pipeline_demo.py --parser-mode model --planner-mode model --env-path .env

  Try the entrance case that exercises the consecutive-yaw safety guard:
    python test_agent_pipeline_demo.py --parser-mode model --planner-mode model --env-path .env --instruction "左转寻找建筑入口，并停在入口前"
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyAgent M3-M6 pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--mode", choices=["parse-only", "run-task"], default="run-task")
    parser.add_argument("--parser-mode", choices=["rule_based", "model"], default="rule_based")
    parser.add_argument("--planner-mode", choices=["rule_based", "model"], default="rule_based")
    parser.add_argument("--shutdown-policy", choices=["hover", "land", "none"], default="hover")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument("--verbose-state", action="store_true", help="Print full MissionState JSON instead of summary.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    action_config = load_uav_action_config(args.action_config)
    model_config = load_model_config(args.env_path)
    task_parser = TaskParser(mode=args.parser_mode, model_config=model_config)

    print_header("FlyAgent Current Pipeline Demo")
    print_current_summary(args)
    if args.show_examples:
        print_instruction_examples()
    print(f"User instruction: {args.instruction}\n")

    print_header("M3 Semantic Plan")
    parse_result = task_parser.parse(args.instruction)
    print("M3 only understands the task. It does not output primitive UAV tools.")
    print_json("task_parse_result", parse_result.to_dict())

    if args.mode == "parse-only":
        print("parse-only selected, so runtime, observation context, and action planning are skipped.")
        return

    print_header("M4-M6 Runtime Dry Run")
    print("The runtime uses Null UAV by default, so this is safe without AirSim or UE.")
    print("Preflight is runtime-owned: connect -> takeoff -> hover -> initial observe.")
    print("Then every semantic phase produces one short-horizon window and one fresh observation.\n")

    state, run_dir = run_pipeline(args, action_config, model_config, parse_result)
    print_header("Final MissionState")
    if args.verbose_state:
        print_json("mission_state", state.to_dict())
    else:
        print_json("mission_state_summary", summarize_state(state))
    print_header("Event Summary")
    summarize_events(run_dir)
    print_header("Generated Files")
    explain_artifacts(run_dir)


def print_current_summary(args: argparse.Namespace) -> None:
    print("Current development state:")
    print("- M3 parses user instructions into open semantic phases with index/total.")
    print("- M4 runs one short-horizon window per phase through ToolDispatcher and Null UAV.")
    print("- M5 normalizes raw rgb/depth/pose observations and writes model-ready context JSON.")
    print("- M6 can use a real model to propose 1-3 short actions, then Safety reviews them.")
    print("- Safety clips large moves/yaw and blocks consecutive blind yaw scans.")
    print()
    print(f"Parser mode: {args.parser_mode}")
    print(f"Planner mode: {args.planner_mode}")
    print(f"Shutdown policy: {args.shutdown_policy}")
    if args.parser_mode == "model" or args.planner_mode == "model":
        print(f"Env path: {args.env_path}")
        print("Model API key is required but never printed or written to run artifacts.")
    else:
        print("Offline rule_based mode does not require .env, API key, AirSim, or UE.")
    print()


def run_pipeline(
    args: argparse.Namespace,
    action_config: UavActionConfig,
    model_config,
    parse_result,
):
    run_dir = RunStore(args.run_root).create_run_dir("agent_pipeline_demo")
    logger = AsyncRunLogger(run_dir)
    package = make_demo_map_package()
    server = LocalSimulationServer(
        [package],
        PortPolicy(server_port=30000, airsim_port_start=30001, max_scene_num=1),
        run_dir=run_dir,
        action_config=action_config,
    )
    logger.write_json(
        "config/model_config.json",
        {
            "model_config": model_config.redacted_summary(),
            "parser_mode": args.parser_mode,
            "planner_mode": args.planner_mode,
            "shutdown_policy": args.shutdown_policy,
        },
    )
    logger.write_json("config/map_package.json", package.to_dict())
    logger.write_json("config/uav_action_config.json", action_config.to_dict())
    runtime = TaskRuntime(
        LocalEnvClient(server),
        logger,
        TaskParser(mode=args.parser_mode, model_config=model_config),
        planner=build_planner(args.planner_mode, model_config, action_config),
        safety=SafetySupervisor(action_config),
        shutdown_policy=args.shutdown_policy,
    )
    try:
        state = runtime.run_parsed_task(package.map_id, parse_result)
        logger.info("agent pipeline demo completed", run_dir=str(run_dir), completed=state.completed)
        return state, run_dir
    finally:
        logger.close()


def build_planner(planner_mode: str, model_config, action_config: UavActionConfig):
    if planner_mode == "rule_based":
        return ShortHorizonPlanner(action_config)
    if planner_mode == "model":
        return ModelActionPlanner(model_config)
    raise ValueError(f"Unsupported planner mode: {planner_mode}")


def summarize_events(run_dir: Path) -> None:
    events = load_events(run_dir / "events.jsonl")
    counts = {
        "task_parsed": count_events(events, "task_parsed"),
        "scene_started": count_events(events, "scene_started"),
        "initial_observation_updated": count_events(events, "initial_observation_updated"),
        "model_action_planned": count_events(events, "model_action_planned"),
        "short_horizon_planned": count_events(events, "short_horizon_planned"),
        "safety_intervened": count_events(events, "safety_intervened"),
        "observation_updated": count_events(events, "observation_updated"),
        "scene_stopped": count_events(events, "scene_stopped"),
        "shutdown_hover_completed": count_events(events, "shutdown_hover_completed"),
        "shutdown_land_completed": count_events(events, "shutdown_land_completed"),
    }
    print_json("event_counts", counts)
    print("Phase windows after Safety review:")
    for event in events:
        if event.get("event") == "short_horizon_planned":
            payload = event.get("payload", {})
            steps = [summarize_step(step) for step in payload.get("steps", [])]
            print(f"- {payload.get('phase')}: {steps}")
    interventions = [event.get("payload", {}) for event in events if event.get("event") == "safety_intervened"]
    if interventions:
        print()
        print_json("safety_interventions", {"items": interventions})
    print(f"Run record: {run_dir}")


def summarize_state(state) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "completed": state.completed,
        "current_phase": state.current_phase_name,
        "connected": state.connected,
        "airborne": state.airborne,
        "recent_action_count": len(state.recent_actions),
        "recent_observation_count": len(state.recent_observations),
        "last_action": state.recent_actions[-1] if state.recent_actions else None,
        "last_observation_source": state.recent_observations[-1].get("source") if state.recent_observations else None,
        "errors": list(state.errors),
        "safety": {"hover_required": state.hover_required},
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def count_events(events: list[dict[str, Any]], name: str) -> int:
    return sum(1 for event in events if event.get("event") == name)


def summarize_step(step: dict[str, Any]) -> str:
    if step.get("kind") == "observe":
        return "observe(rgb,depth,pose)"
    tool = step.get("tool")
    args = step.get("args") or {}
    if args:
        return f"{tool}{args}"
    return str(tool)


def explain_artifacts(run_dir: Path) -> None:
    artifacts = {
        "semantic_plan.json": "M3 output: validated task intent, goal, phases, and uncertainties.",
        "mission_state.json": "Runtime state: current phase, recent actions, observations, errors, safety flags.",
        "observations/latest_observation.json": "M5 normalized raw observation snapshot from rgb/depth/pose.",
        "contexts/latest_context.json": "M5 context that M6 reads before proposing the next short action window.",
        "events.jsonl": "Replayable timeline: parse, preflight, planning, safety, tool calls, observations.",
        "config/model_config.json": "Redacted model config summary; API key is not written.",
        "config/map_package.json": "The map/simulation package used for this demo run.",
        "config/uav_action_config.json": "Yaw, velocity, and duration limits used by planner and Safety.",
        "scenes/<scene_id>/settings.json": "Generated scene settings; present even in Null UAV dry-run.",
    }
    for relative_path, meaning in artifacts.items():
        exists = artifact_exists(run_dir, relative_path)
        marker = "exists" if exists else "missing"
        print(f"- {relative_path} [{marker}]: {meaning}")
        if relative_path != "scenes/<scene_id>/settings.json":
            print(f"  path: {run_dir / relative_path}")


def artifact_exists(run_dir: Path, relative_path: str) -> bool:
    if relative_path == "scenes/<scene_id>/settings.json":
        return any(run_dir.glob("scenes/*/settings.json"))
    return (run_dir / relative_path).exists()


def make_demo_map_package() -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "agent_pipeline_demo_manual",
            "provider": "packaged_unreal",
            "scene_id": "agent_pipeline_demo",
            "root": None,
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {
                "mode": "Multirotor",
                "vehicle_name": "Drone_1",
                "camera_name": "front_0",
                "image_width": 224,
                "image_height": 224,
                "depth_width": 256,
                "depth_height": 256,
            },
            "resources": {"dry_run": True},
        }
    )


def print_instruction_examples() -> None:
    print("Instruction examples:")
    for index, instruction in enumerate(INSTRUCTION_EXAMPLES, start=1):
        print(f"  {index}. {instruction}")
    print()


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_json(label: str, data: dict[str, Any]) -> None:
    print(f"{label}:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()


if __name__ == "__main__":
    main()
