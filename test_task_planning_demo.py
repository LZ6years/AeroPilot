"""Manual demo for M3 semantic planning and M4 short-horizon execution.

This file is not a pytest unit test. It is a small walkthrough you can run by
hand to see what the new Agent planning layers do before any real VLM or UE
environment is required.

What this demo shows:

    M3: instruction -> SemanticPlan JSON
    M4: SemanticPlan phases -> short-horizon dry-run windows -> run record files

Offline rule-based examples, no .env/model/AirSim required:

    python test_task_planning_demo.py
    python test_task_planning_demo.py --mode parse-only
    python test_task_planning_demo.py --mode run-task
    python test_task_planning_demo.py --instruction "前面右转会有一个黑色的路灯，右转面向路灯后停下"

Real model examples, after configuring .env:

    $env:PYTHONIOENCODING='utf-8'
    python test_task_planning_demo.py --parser-mode model --env-path .env --mode parse-only
    python test_task_planning_demo.py --parser-mode model --env-path .env --mode run-task

More instruction examples:

    python test_task_planning_demo.py --parser-mode model --env-path .env --instruction "前方寻找红色车辆，靠近但保持安全距离"
    python test_task_planning_demo.py --parser-mode model --env-path .env --instruction "左转寻找建筑入口，并停在入口前"
    python test_task_planning_demo.py --parser-mode model --env-path .env --instruction "观察右侧区域，如果目标不明确就悬停等待确认"
    python test_task_planning_demo.py --parser-mode model --env-path .env --instruction "Search ahead for a white delivery truck, face it, and hover."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
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
    "前方寻找红色车辆，靠近但保持安全距离",
    "左转寻找建筑入口，并停在入口前",
    "观察右侧区域，如果目标不明确就悬停等待确认",
    "沿着道路向前观察，找到蓝色指示牌后面向它悬停",
    "Search ahead for a white delivery truck, face it, and hover.",
    "Turn left, find the building entrance, and hover in front of it.",
    "右側のエリアを観察して、目標が不明確なら停止して確認してください",
]

HELP_EPILOG = """
Operation examples:

  Offline M3+M4 dry-run, no API key:
    python test_task_planning_demo.py --parser-mode rule_based --mode run-task

  Real model M3 parse only:
    $env:PYTHONIOENCODING='utf-8'
    python test_task_planning_demo.py --parser-mode model --env-path .env --mode parse-only

  Real model M3 plus M4 Null UAV dry-run:
    $env:PYTHONIOENCODING='utf-8'
    python test_task_planning_demo.py --parser-mode model --env-path .env --mode run-task

  Real model with a custom instruction:
    python test_task_planning_demo.py --parser-mode model --env-path .env --instruction "前方寻找红色车辆，靠近但保持安全距离"

Instruction examples:
  - 前面右转会有一个黑色的路灯，右转面向路灯后停下
  - 前方寻找红色车辆，靠近但保持安全距离
  - 左转寻找建筑入口，并停在入口前
  - 观察右侧区域，如果目标不明确就悬停等待确认
  - 沿着道路向前观察，找到蓝色指示牌后面向它悬停
  - Search ahead for a white delivery truck, face it, and hover.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FlyAgent M3/M4 planning demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--mode", choices=["parse-only", "run-task"], default="run-task")
    parser.add_argument(
        "--parser-mode",
        choices=["rule_based", "model"],
        default="rule_based",
        help="Use rule_based for offline tests or model for the configured real provider.",
    )
    parser.add_argument("--env-path", default=".env", help="Path to model .env when --parser-mode model is used.")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--show-examples", action="store_true", help="Print instruction examples before running.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    action_config = load_uav_action_config(args.action_config)
    model_config = load_model_config(args.env_path)
    task_parser = TaskParser(mode=args.parser_mode, model_config=model_config)
    print_header("FlyAgent M3/M4 Planning Demo")
    print("This demo explains M3 semantic planning and M4 short-horizon execution.")
    print(f"Parser mode: {args.parser_mode}")
    if args.parser_mode == "model":
        print(f"Env path: {args.env_path}")
        print("A configured model API key is required. The key is never printed or written to the run record.")
    else:
        print("rule_based mode does not require .env, a model API key, AirSim, or UE.")
    print("M4 still uses Null UAV dry-run by default, so it is safe to run without AirSim/UE.\n")
    if args.show_examples:
        print_instruction_examples()
    print(f"User instruction: {args.instruction}\n")

    parse_result = show_m3_parse(args.instruction, task_parser, args.parser_mode)
    show_short_horizon_preview(parse_result.plan, args.instruction, action_config)
    if args.mode == "parse-only":
        print("\nMode parse-only selected, so M4 runtime execution is skipped.")
        return
    state, run_dir = run_m4_task(args, action_config, model_config, parse_result)
    show_final_state(state)
    explain_run_files(run_dir)


def show_m3_parse(instruction: str, task_parser: TaskParser, parser_mode: str):
    print_header("M3: Task Parsing And Semantic Planning")
    print("M3 turns natural language into a semantic plan.")
    print("The plan describes intent, goal, phases, and uncertainty.")
    print("Phase names are open semantic labels from the parser, not fixed action dispatch keys.")
    print("It must not contain primitive UAV commands such as uav.turn_relative.\n")
    print(f"Parsing with: {parser_mode}\n")
    parse_result = task_parser.parse(instruction)
    print_json("semantic_plan", parse_result.plan.to_dict())
    return parse_result


def show_short_horizon_preview(plan, instruction: str, action_config: UavActionConfig) -> None:
    print_header("M4 Preview: Short-Horizon Windows")
    print("TaskRuntime performs preflight before these phase windows:")
    print("  uav.connect -> uav.takeoff -> uav.hover -> initial observe")
    print("Those lifecycle steps are not part of model/phase action planning.\n")
    print("M4 consumes the semantic phases one window at a time.")
    print("Each window is reviewed by SafetySupervisor before any tool call can run.\n")
    planner = ShortHorizonPlanner(action_config)
    safety = SafetySupervisor(action_config)
    state = MissionState("preview", instruction, plan)
    state.connected = True
    state.airborne = True
    state.record_observation({"preview": True, "source": "preflight", "modalities": ["rgb", "depth", "pose"]})
    for _ in plan.phases:
        raw_plan = planner.plan_next(state)
        decision = safety.review(raw_plan)
        print_json(f"phase:{decision.plan.phase}", decision.plan.to_dict())
        if decision.interventions:
            print_json("safety_interventions", {"interventions": decision.interventions})
        simulate_preview_state_updates(state, decision.plan)
        state.advance_phase()


def run_m4_task(
    args: argparse.Namespace,
    action_config: UavActionConfig,
    model_config,
    parse_result,
) -> tuple[MissionState, Path]:
    print_header("M4: Dry-Run Task Runtime")
    print("Now the same instruction runs through TaskRuntime.")
    print("Path: TaskRuntime -> ToolDispatcher -> LocalEnvClient -> LocalSimulationServer -> NullUAV\n")
    run_dir = RunStore(args.run_root).create_run_dir("task_planning_demo")
    logger = AsyncRunLogger(run_dir)
    package = make_demo_map_package()
    server = LocalSimulationServer(
        [package],
        PortPolicy(server_port=30000, airsim_port_start=30001, max_scene_num=1),
        run_dir=run_dir,
        action_config=action_config,
    )
    logger.write_json("config/model_config.json", {"model_config": model_config.redacted_summary()})
    logger.write_json("config/map_package.json", package.to_dict())
    logger.write_json("config/uav_action_config.json", action_config.to_dict())
    runtime = TaskRuntime(
        LocalEnvClient(server),
        logger,
        TaskParser(mode=args.parser_mode, model_config=model_config),
        planner=ShortHorizonPlanner(action_config),
        safety=SafetySupervisor(action_config),
    )
    try:
        state = runtime.run_parsed_task(package.map_id, parse_result)
        logger.info("planning demo completed", run_dir=str(run_dir), completed=state.completed)
        return state, run_dir
    finally:
        logger.close()


def print_instruction_examples() -> None:
    print("Instruction examples:")
    for index, instruction in enumerate(INSTRUCTION_EXAMPLES, start=1):
        print(f"  {index}. {instruction}")
    print()


def simulate_preview_state_updates(state: MissionState, plan) -> None:
    # Keep the preview readable without actually calling EnvClient.
    for step in plan.steps:
        if step.kind == "observe":
            state.record_observation({"preview": True, "modalities": step.modalities})
        elif step.tool:
            state.record_action(step.tool, step.args, {"ok": True, "preview": True})


def make_demo_map_package() -> MapPackage:
    return MapPackage.from_dict(
        {
            "map_id": "planning_demo_manual",
            "provider": "packaged_unreal",
            "scene_id": "planning_demo",
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


def show_final_state(state: MissionState) -> None:
    print_header("Final MissionState")
    print("MissionState is the M4 snapshot of phase progress, recent actions, observations, and safety state.\n")
    print_json("mission_state", state.to_dict())


def explain_run_files(run_dir: Path) -> None:
    print_header("Generated Run Files")
    files = {
        "semantic_plan.json": "M3 structured semantic plan generated from the user instruction.",
        "mission_state.json": "Final M4 task state after short-horizon windows finish.",
        "observations/latest_observation.json": "M5 normalized raw RGB/depth/pose observation snapshot.",
        "contexts/latest_context.json": "M5 model-ready context for the future M6 action planner.",
        "events.jsonl": "Structured event timeline for replaying every phase, step, action, and observation.",
        "system.log": "Human-readable run log with start, completion, and error records.",
        "config/model_config.json": "Model config summary; API key is never written here.",
        "config/map_package.json": "The map/simulation package used by this demo run.",
        "config/uav_action_config.json": "Primitive action scale limits used by planner and safety.",
        "scenes/<scene_id>/settings.json": "Generated AirSim settings for the scene, even in dry-run mode.",
    }
    print(f"Run record: {run_dir}")
    for relative_path, meaning in files.items():
        display_path = run_dir / relative_path
        exists = path_exists_for_display(run_dir, relative_path)
        marker = "exists" if exists else "not found"
        print(f"- {relative_path} [{marker}]: {meaning}")
        if relative_path != "scenes/<scene_id>/settings.json":
            print(f"  path: {display_path}")


def path_exists_for_display(run_dir: Path, relative_path: str) -> bool:
    if relative_path == "scenes/<scene_id>/settings.json":
        return any(run_dir.glob("scenes/*/settings.json"))
    return (run_dir / relative_path).exists()


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
