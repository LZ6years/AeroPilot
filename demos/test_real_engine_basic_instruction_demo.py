r"""Manual headed real-engine regression demo for basic language instructions.

By default this launches one visible UE/AirSim scene, then sends all
natural-language instructions continuously through TaskRuntime.run_task(...)
with scene_policy="reuse_running". The demo does not manually pass SceneInstance
objects around.

Recommended PowerShell:

    $env:PYTHONIOENCODING='utf-8'
    F:\conda_env\flyagent\python.exe test_real_engine_basic_instruction_demo.py

Pause before every instruction so you can watch the engine window:

    F:\conda_env\flyagent\python.exe test_real_engine_basic_instruction_demo.py --confirm-each

Use one instruction only:

    F:\conda_env\flyagent\python.exe test_real_engine_basic_instruction_demo.py --instruction "向前飞行5米，然后悬停" --save-artifacts --only 1

Restart UE for every instruction, matching the older behavior:

    F:\conda_env\flyagent\python.exe test_real_engine_basic_instruction_demo.py --separate-scenes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.semantic.task_parser import TaskParser
from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import LocalSimulationServer


BASIC_INSTRUCTIONS = [
    "向前飞行5米，然后悬停",
    "向后退3米，然后悬停",
    "向左平移2米，然后悬停",
    "向右平移2米，然后悬停",
    "向上飞行5米，然后悬停",
    "向下飞行2米，然后悬停",
    "原地悬停等待",
    "向右转30度，然后悬停",
    "向左转30度，然后悬停",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headed real-engine basic instruction regression demo")
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    parser.add_argument("--safety-config", default="configs/uav_safety_config.json")
    parser.add_argument("--map-id", default="cityenviron_win64")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--parser-mode", choices=["model", "rule_based"], default="model")
    parser.add_argument("--shutdown-policy", choices=["hover", "land", "none"], default="hover")
    parser.add_argument("--connect-existing", action="store_true", help="Use an already-running scene instead of launch")
    parser.add_argument("--no-launch", action="store_true", help="Do not launch the configured scene process")
    parser.add_argument("--save-artifacts", action="store_true", help="Save real RGB/depth frame artifacts under the run directory")
    parser.add_argument("--headless", action="store_true", help="Keep the engine display hidden if the map supports it")
    parser.add_argument(
        "--separate-scenes",
        action="store_true",
        help="Restart the scene for every instruction; default keeps one UE scene open",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Abort the session on the first instruction error")
    parser.add_argument("--confirm-each", action="store_true", help="Wait for Enter before each instruction")
    parser.add_argument("--only", type=int, default=None, help="Run only the 1-based instruction index")
    parser.add_argument("--instruction", action="append", help="Custom instruction; can be passed multiple times")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instructions = select_instructions(args)
    maps, port_policy = load_map_config(args.config)
    model_config = load_model_config(args.env_path)
    action_config = load_uav_action_config(args.action_config)
    safety_config = load_uav_safety_config(args.safety_config)
    selected_map = next(pkg for pkg in maps if pkg.map_id == args.map_id)
    launch_processes = not args.no_launch and not args.connect_existing
    real_control = launch_processes or args.connect_existing
    run_prefix = "real_engine_basic_instruction" if args.separate_scenes else "real_engine_basic_instruction_session"
    run_dir = RunStore(args.run_root).create_run_dir(run_prefix)
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
    runtime = make_runtime(args, LocalEnvClient(server), recorder, model_config, action_config, safety_config)
    summaries: list[dict[str, Any]] = []
    last_state = None

    print("Real engine basic instruction demo")
    print(f"map_id={args.map_id}")
    print(f"parser_mode={args.parser_mode}, planner=ModelActionPlanner")
    print(f"shutdown_policy={args.shutdown_policy}")
    print(f"launch_processes={launch_processes}, headed={not args.headless}")
    print(f"continuous_scene={not args.separate_scenes}")
    print(f"run_dir={run_dir}")
    print()

    try:
        write_common_config(recorder, selected_map, model_config, action_config, safety_config, args)
        for index, instruction in enumerate(instructions, start=1):
            if args.confirm_each:
                input(f"[{index}/{len(instructions)}] Press Enter to run: {instruction}")
            print(f"\n[{index}/{len(instructions)}] {instruction}")
            try:
                state = runtime.run_task(
                    args.map_id,
                    instruction,
                    scene_policy="isolated" if args.separate_scenes else "reuse_running",
                    stop_scene_on_finish=args.separate_scenes,
                )
                last_state = state
                summary = build_state_summary(run_dir, instruction, state.to_dict())
            except Exception as exc:
                recorder.event(
                    "continuous_instruction_failed",
                    {"instruction": instruction, "error_type": type(exc).__name__, "error": str(exc)},
                )
                runtime.safe_hover_reused_scene(args.map_id)
                summary = build_error_summary(run_dir, instruction, last_state, exc)
                if args.stop_on_error:
                    raise
            summaries.append(summary)
            recorder.event("continuous_instruction_completed", summary)
            print(json.dumps(compact_summary(summary), indent=2, ensure_ascii=False))
    finally:
        if not args.separate_scenes:
            runtime.finish_reused_scene(args.map_id, last_state)
        recorder.close()

    if summaries:
        summaries[-1]["shutdown_events"] = read_shutdown_events(run_dir)
    session_summary = build_session_summary(run_dir, summaries)
    write_json_file(run_dir / "demo" / "summary.json", session_summary)
    summary_path = Path(args.run_root) / "real_engine_basic_instruction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run summary saved to: {run_dir / 'demo' / 'summary.json'}")
    print(f"\nFull summary saved to: {summary_path}")


def select_instructions(args: argparse.Namespace) -> list[str]:
    instructions = args.instruction or BASIC_INSTRUCTIONS
    if args.only is not None:
        if args.only < 1 or args.only > len(instructions):
            raise ValueError(f"--only must be between 1 and {len(instructions)}")
        return [instructions[args.only - 1]]
    return instructions


def make_runtime(args: argparse.Namespace, client, recorder, model_config, action_config, safety_config) -> TaskRuntime:
    return TaskRuntime(
        client,
        recorder,
        TaskParser(mode=args.parser_mode, model_config=model_config),
        planner=ModelActionPlanner(model_config),
        safety=SafetySupervisor(safety_config),
        shutdown_policy=args.shutdown_policy,
        action_config=action_config,
    )


def write_common_config(recorder, selected_map, model_config, action_config, safety_config, args: argparse.Namespace) -> None:
    recorder.write_json("config/map_package.json", selected_map.to_dict())
    recorder.write_json(
        "config/model_config.json",
        {
            "model_config": model_config.redacted_summary(),
            "parser_mode": args.parser_mode,
            "planner": "ModelActionPlanner",
            "shutdown_policy": args.shutdown_policy,
            "continuous_scene": not args.separate_scenes,
            "save_artifacts": args.save_artifacts,
        },
    )
    recorder.write_json("config/uav_action_config.json", action_config.to_dict())
    recorder.write_json("config/uav_safety_config.json", safety_config.to_dict())


def build_state_summary(run_dir: Path, instruction: str, state: dict[str, Any]) -> dict[str, Any]:
    latest_observation = state.get("latest_observation", {})
    progress = state.get("current_phase_progress", {}) if isinstance(state.get("current_phase_progress", {}), dict) else {}
    initial_pose = dict(progress.get("start_pose", {}))
    final_pose = dict(state.get("latest_pose", {})) or pose_from_observation(latest_observation)
    events = load_events(run_dir / "events.jsonl")
    observations = observations_from_events(events) or ([latest_observation] if latest_observation else [])
    return {
        "instruction": instruction,
        "run_dir": str(run_dir),
        "completed": state.get("completed"),
        "connected": state.get("connected"),
        "airborne": state.get("airborne"),
        "tools": action_sequence_from_events(events),
        "initial_pose": initial_pose,
        "final_pose": final_pose,
        "pose_delta": pose_delta(initial_pose, final_pose),
        "phase_execution_summary": progress,
        "observation_summary": observation_summary(observations),
        "artifact_inventory": artifact_inventory_from_observations(observations),
        "shutdown_events": [],
    }


def build_error_summary(run_dir: Path, instruction: str, last_state, exc: Exception) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "run_dir": str(run_dir),
        "completed": False,
        "connected": last_state.connected if last_state is not None else None,
        "airborne": last_state.airborne if last_state is not None else None,
        "tools": [],
        "initial_pose": {},
        "final_pose": {},
        "pose_delta": pose_delta({}, {}),
        "observation_summary": {"has_pose": False, "has_rgb": False, "has_depth": False, "sources": []},
        "phase_execution_summary": {},
        "artifact_inventory": {"rgb": [], "depth": [], "other": [], "counts": {"rgb": 0, "depth": 0, "other": 0}},
        "shutdown_events": [],
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "instruction": summary["instruction"],
        "run_dir": summary["run_dir"],
        "completed": summary["completed"],
        "airborne_after_shutdown": summary["airborne"],
        "tools": summary["tools"],
        "pose_delta": summary["pose_delta"],
        "phase_execution_summary": summary.get("phase_execution_summary", {}),
        "observation_summary": summary.get("observation_summary", {}),
        "artifact_counts": summary.get("artifact_inventory", {}).get("counts", {}),
        "shutdown_events": summary["shutdown_events"],
    }
    if "error" in summary:
        compact["error_type"] = summary["error_type"]
        compact["error"] = summary["error"]
    return compact


def build_session_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    events = load_events(run_dir / "events.jsonl")
    return {
        "run_dir": str(run_dir),
        "instructions": summaries,
        "latest_task_parse_result": latest_event_payload(events, "task_parsed"),
        "latest_model_action_plan": latest_event_payload(events, "model_action_planned"),
        "latest_validated_plan": latest_event_payload(events, "short_horizon_planned"),
        "latest_safety_intervention": latest_event_payload(events, "safety_intervened"),
        "latest_phase_transition_context": latest_event_payload(events, "phase_transition_judge_context_built"),
        "model_plan_recorded": latest_event_payload(events, "model_action_planned") is not None,
        "validated_plan_recorded": latest_event_payload(events, "short_horizon_planned") is not None,
        "automatic_observation_recorded": latest_event_payload(events, "automatic_observation_updated") is not None,
        "action_sequence": action_sequence_from_events(events),
    }


def latest_event_payload(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == event_name:
            payload = event.get("payload")
            return dict(payload) if isinstance(payload, dict) else {}
    return None


def observation_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "has_pose": any(bool(obs.get("pose")) for obs in observations if isinstance(obs, dict)),
        "has_rgb": any("rgb" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "has_depth": any("depth" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "sources": [str(obs.get("source", "")) for obs in observations if isinstance(obs, dict) and obs.get("source")],
    }


def action_sequence_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for event in events:
        if event.get("event") != "step_completed":
            continue
        payload = event.get("payload", {})
        step = payload.get("step", {}) if isinstance(payload, dict) else {}
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        actions.append(
            {
                "phase": payload.get("phase"),
                "tool": step.get("tool"),
                "args": step.get("args", {}),
                "ok": result.get("ok"),
            }
        )
    return actions


def observations_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for event in events:
        if event.get("event") not in {"initial_observation_updated", "instruction_start_observation_updated", "automatic_observation_updated"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            observations.append(payload)
    return observations


def artifact_inventory_from_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, list[dict[str, Any]]] = {"rgb": [], "depth": [], "other": []}
    for obs_index, observation in enumerate(observations):
        frames = observation.get("frames", {}) if isinstance(observation, dict) else {}
        if not isinstance(frames, dict):
            continue
        for modality, metadata in frames.items():
            if not isinstance(metadata, dict) or "artifact_path" not in metadata:
                continue
            normalized = "rgb" if str(modality).lower() in {"rgb", "scene", "color"} else str(modality).lower()
            bucket = normalized if normalized in {"rgb", "depth"} else "other"
            path = Path(str(metadata["artifact_path"]))
            inventory[bucket].append(
                {
                    "observation_index": obs_index,
                    "modality": str(modality),
                    "path": str(path),
                    "exists": path.exists(),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                }
            )
    return {
        **inventory,
        "counts": {name: len(items) for name, items in inventory.items()},
    }


def read_shutdown_events(run_dir: Path) -> list[str]:
    return [
        event["event"]
        for event in load_events(run_dir / "events.jsonl")
        if str(event.get("event", "")).startswith("shutdown_")
    ]


def pose_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    pose = observation.get("pose") if isinstance(observation, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def pose_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    return {
        "dx": subtract(pose_value(after, "x"), pose_value(before, "x")),
        "dy": subtract(pose_value(after, "y"), pose_value(before, "y")),
        "dz": subtract(pose_value(after, "z"), pose_value(before, "z")),
        "dyaw_deg": subtract(pose_value(after, "yaw_deg"), pose_value(before, "yaw_deg")),
    }


def pose_value(pose: dict[str, Any], key: str) -> Any:
    if key in pose:
        return pose.get(key)
    if key in {"x", "y", "z"}:
        position = pose.get("position")
        if isinstance(position, dict):
            return position.get(key)
    orientation = pose.get("orientation")
    if isinstance(orientation, dict):
        return orientation.get(key)
    return None


def subtract(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    try:
        return round(float(after) - float(before), 3)
    except (TypeError, ValueError):
        return None


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
