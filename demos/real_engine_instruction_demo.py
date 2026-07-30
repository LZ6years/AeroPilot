r"""Manual headed real-engine regression demo for basic language instructions.

By default this launches one visible UE/AirSim scene, then sends all
natural-language instructions continuously through TaskRuntime.run_task(...)
with scene_policy="reuse_running". The demo does not manually pass SceneInstance
objects around.

Recommended PowerShell:

    $env:PYTHONIOENCODING='utf-8'
    F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py

Pause before every instruction so you can watch the engine window:

    F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py --confirm-each

Use one instruction only:

    F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py --instruction "向前飞行5米，然后悬停" --save-artifacts --only 1

Restart UE for every instruction, matching the older behavior:

    F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py --separate-scenes
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
from src.env_client import SimulatorEnvClient
from src.runtime.configs.action_config import load_uav_action_config
from src.runtime.configs.map_config import load_map_config
from src.runtime.configs.model_config import load_model_config
from src.runtime.configs.safety_config import load_uav_safety_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.server import SimulationServer
from src.uav import pose_delta, pose_from_observation


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
    parser.add_argument(
        "--execution",
        choices=["attach", "managed"],
        default="managed",
        help=(
            "Simulator execution policy; this real-engine demo defaults to "
            "managed, while attach never starts or stops an external UE"
        ),
    )
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
    parser.add_argument("--task-suite", help="Load instructions from an evaluation task suite JSON")
    parser.add_argument(
        "--suite-group",
        choices=["single", "continuous"],
        default="single",
        help="Task suite group to run when --task-suite is provided",
    )
    parser.add_argument("--task-id", help="Run a specific task id from --task-suite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Instruction selection is separated from execution so the same demo can
    # run built-in smoke instructions, ad-hoc --instruction values, or a
    # standard evaluation task suite.
    instructions = select_instructions(args)

    # These configs define the real runtime boundary: map/scene launch,
    # model-backed parser/planner, nominal action scale, and hard safety limits.
    maps, port_policy = load_map_config(args.config)
    model_config = load_model_config(args.env_path)
    action_config = load_uav_action_config(args.action_config)
    safety_config = load_uav_safety_config(args.safety_config)

    selected_map = next(pkg for pkg in maps if pkg.map_id == args.map_id)
    # A continuous session keeps one UE scene alive for multiple instructions.
    # Separate-scene mode is better for regression tests that need isolation.
    run_prefix = "real_engine_basic_instruction" if args.separate_scenes else "real_engine_basic_instruction_session"
    run_dir = RunStore(args.run_root).create_run_dir(run_prefix)
    recorder = AsyncRunLogger(run_dir)

    # SimulationServer owns scene lifecycle and the concrete UAV backend.
    # TaskRuntime talks to it only through SimulatorEnvClient/ToolDispatcher,
    # which
    # keeps high-level planning independent from AirSim/UE details.
    server = SimulationServer(
        maps,
        port_policy,
        run_dir=run_dir,
        execution_mode=args.execution,
        show_display=not args.headless,
        action_config=action_config,
        save_artifacts=args.save_artifacts,
    )

    # This is the full production pipeline used by the demo:
    # TaskParser(model) -> context builder -> ModelActionPlanner(model) ->
    # coupled phase decision -> ToolContractValidator -> SafetySupervisor ->
    # ToolDispatcher/UAV -> automatic observation when the phase continues.
    runtime = make_runtime(args, SimulatorEnvClient(server), recorder, model_config, action_config, safety_config)
    summaries: list[dict[str, Any]] = []
    last_state = None

    print("Real engine basic instruction demo")
    print(f"map_id={args.map_id}")
    print(f"parser_mode={args.parser_mode}, planner=ModelActionPlanner")
    print(f"shutdown_policy={args.shutdown_policy}")
    print(f"execution={args.execution}, headed={not args.headless}")
    print(f"continuous_scene={not args.separate_scenes}")
    print(f"run_dir={run_dir}")
    print()

    try:
        # Write the resolved configuration once at the beginning. These files
        # make each run reproducible without exposing the raw API key.
        write_common_config(recorder, selected_map, model_config, action_config, safety_config, args)
        for index, instruction in enumerate(instructions, start=1):
            if args.confirm_each:
                input(f"[{index}/{len(instructions)}] Press Enter to run: {instruction}")
            print(f"\n[{index}/{len(instructions)}] {instruction}")
            try:
                # run_task is where the real work happens. The demo does not
                # manually call planner/safety/UAV methods; it lets TaskRuntime
                # produce the same artifacts as CLI use.
                state = runtime.run_task(
                    args.map_id,
                    instruction,
                    scene_policy="isolated" if args.separate_scenes else "reuse_running",
                    stop_scene_on_finish=args.separate_scenes,
                )
                last_state = state
                summary = build_state_summary(
                    run_dir,
                    instruction,
                    state.to_dict(),
                    agent_messages=runtime.latest_turn_agent_messages(),
                )
            except Exception as exc:
                # Continuous sessions should try to leave the UAV in a safe
                # hover even when one instruction fails, so later analysis can
                # distinguish runtime failure from unsafe shutdown.
                recorder.event(
                    "continuous_instruction_failed",
                    {"instruction": instruction, "error_type": type(exc).__name__, "error": str(exc)},
                )
                runtime.safe_hover_reused_scene(args.map_id)
                summary = build_error_summary(
                    run_dir,
                    instruction,
                    last_state,
                    exc,
                    agent_messages=runtime.latest_turn_agent_messages(),
                )
                if args.stop_on_error:
                    raise
            summaries.append(summary)
            recorder.event("continuous_instruction_completed", summary)
            print(json.dumps(compact_summary(summary), indent=2, ensure_ascii=False))
    finally:
        # In continuous mode the scene is intentionally kept open across
        # instructions, so it must be closed once after the final instruction.
        if not args.separate_scenes:
            runtime.finish_reused_scene(args.map_id, last_state)
        recorder.close()

    # demo/summary.json is the user-facing rollup. Raw events, contexts, memory,
    # and state files remain in their dedicated run folders for deeper reading.
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
    """Resolve the natural-language instruction list for this run.

    Priority is: task suite -> explicit --instruction values -> built-in smoke
    list. This keeps the demo convenient for both manual exploration and
    repeatable evaluation.
    """

    if args.task_suite:
        instructions = instructions_from_task_suite(Path(args.task_suite), args.suite_group, args.task_id)
        if args.only is not None:
            if args.only < 1 or args.only > len(instructions):
                raise ValueError(f"--only must be between 1 and {len(instructions)}")
            return [instructions[args.only - 1]]
        return instructions
    instructions = args.instruction or BASIC_INSTRUCTIONS
    if args.only is not None:
        if args.only < 1 or args.only > len(instructions):
            raise ValueError(f"--only must be between 1 and {len(instructions)}")
        return [instructions[args.only - 1]]
    return instructions


def instructions_from_task_suite(path: Path, group: str, task_id: str | None) -> list[str]:
    """Load standard task instructions without interpreting their metrics.

    Metrics are consumed later by the evaluator. This demo only needs the
    natural-language text so the normal runtime pipeline remains unchanged.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    key = "continuous_instruction_tasks" if group == "continuous" else "single_instruction_tasks"
    tasks = list(data.get(key, []))
    if task_id:
        tasks = [task for task in tasks if task.get("id") == task_id]
    if not tasks:
        raise ValueError(f"No tasks selected from {path} group={group!r} task_id={task_id!r}")
    instructions: list[str] = []
    for task in tasks:
        if group == "continuous":
            instructions.extend(str(item) for item in task.get("instructions", []))
        else:
            instructions.append(str(task.get("instruction", "")))
    return [instruction for instruction in instructions if instruction]


def make_runtime(args: argparse.Namespace, client, recorder, model_config, action_config, safety_config) -> TaskRuntime:
    """Construct the same runtime stack used by interactive operation.

    The demo injects concrete parser/planner/safety dependencies here so the
    rest of the file can stay focused on session orchestration and reporting.
    """

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
    """Persist resolved configuration for later debugging.

    Model config is redacted, while map/action/safety configs are written in
    full because they explain why a particular task used certain durations,
    limits, ports, and stopping behavior.
    """

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
            "task_suite": getattr(args, "task_suite", None),
            "suite_group": getattr(args, "suite_group", None),
            "task_id": getattr(args, "task_id", None),
        },
    )
    recorder.write_json("config/uav_action_config.json", action_config.to_dict())
    recorder.write_json("config/uav_safety_config.json", safety_config.to_dict())


def build_state_summary(
    run_dir: Path,
    instruction: str,
    state: dict[str, Any],
    *,
    agent_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one compact per-instruction summary from runtime artifacts.

    This summary is for humans reading the demo output. It deliberately points
    back to the canonical files rather than becoming a second source of truth:
    latest state comes from state/latest_state.json, history comes from events
    and memory JSONL, and image/depth files come from artifact refs.
    """

    latest_observation = state.get("latest_observation", {})
    progress = state.get("current_phase_progress", {}) if isinstance(state.get("current_phase_progress", {}), dict) else {}
    initial_pose = {}
    final_pose = pose_from_observation(latest_observation)
    events = load_events(run_dir / "events.jsonl")
    debug_events = load_events(run_dir / "debug" / "debug_events.jsonl")
    segment_events = latest_instruction_events(events + debug_events)
    observations = observations_from_events(segment_events) or ([latest_observation] if latest_observation else [])
    messages = list(agent_messages) if agent_messages is not None else agent_messages_from_events(segment_events)
    return {
        "instruction": instruction,
        "run_dir": str(run_dir),
        "completed": state.get("completed"),
        "connected": state.get("connected"),
        "airborne": state.get("airborne"),
        "tools": action_sequence_from_events(segment_events),
        "action_windows": action_windows_from_events(segment_events),
        "initial_pose": initial_pose,
        "final_pose": final_pose,
        "pose_delta": pose_delta(initial_pose, final_pose),
        "phase_execution_summary": progress,
        "observation_summary": observation_summary(observations),
        "artifact_inventory": artifact_inventory_from_observations(observations),
        "agent_messages": messages,
        "shutdown_events": [],
    }


def build_error_summary(
    run_dir: Path,
    instruction: str,
    last_state,
    exc: Exception,
    *,
    agent_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a summary shape compatible with successful instructions."""

    messages = (
        list(agent_messages)
        if agent_messages is not None
        else agent_messages_from_events(latest_instruction_events(load_events(run_dir / "events.jsonl")))
    )
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
        "agent_messages": messages,
        "shutdown_events": [],
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Trim a full instruction summary for terminal printing.

    The full summary keeps every field needed for debugging. The terminal view
    should instead answer: what instruction ran, which user-planned tools ran,
    how far the current phase moved, whether observations/artifacts exist, and
    where to inspect the canonical files.
    """

    compact = {
        "instruction": summary["instruction"],
        "run_dir": summary["run_dir"],
        "completed": summary["completed"],
        "airborne": summary["airborne"],
        "action_count": len(summary.get("tools", [])),
        "tools": summarize_tools(summary.get("tools", [])),
        "action_window": summarize_action_window(latest_action_window(summary.get("action_windows", []))),
        "agent_messages": summarize_agent_messages(summary.get("agent_messages", [])),
        "pose_delta": summary["pose_delta"],
        "phase_progress": summarize_phase_progress(summary.get("phase_execution_summary", {})),
        "observation_summary": summary.get("observation_summary", {}),
        "artifact_counts": summary.get("artifact_inventory", {}).get("counts", {}),
        "inspect_files": {
            "state": str(Path(summary["run_dir"]) / "state" / "latest_state.json"),
            "planner_context": str(Path(summary["run_dir"]) / "contexts" / "latest_context.json"),
            "planner_model_context": str(Path(summary["run_dir"]) / "contexts" / "latest_action_planner_model_context.json"),
            "demo_summary": str(Path(summary["run_dir"]) / "demo" / "summary.json"),
        },
        "shutdown_events": summary["shutdown_events"],
    }
    if "error" in summary:
        compact["error_type"] = summary["error_type"]
        compact["error"] = summary["error"]
    return compact


def summarize_agent_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep terminal output focused on user-readable model replies."""

    return [
        {
            "type": item.get("type"),
            "content": item.get("content"),
            "confidence": item.get("confidence"),
            "requires_user_reply": item.get("requires_user_reply"),
            "source": item.get("source"),
            "phase": item.get("phase"),
        }
        for item in messages
        if isinstance(item, dict) and item.get("content")
    ]


def summarize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep terminal tool output focused on model-planned user actions."""

    return [
        {
            "phase": item.get("phase"),
            "tool": item.get("tool"),
            "args": item.get("args", {}),
            "ok": item.get("ok"),
        }
        for item in tools
    ]


def latest_action_window(action_windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the last action window emitted for the current instruction segment."""

    if not action_windows:
        return {}
    return action_windows[-1] if isinstance(action_windows[-1], dict) else {}


def summarize_action_window(action_window: dict[str, Any]) -> dict[str, Any]:
    """Keep terminal action-window output focused on intent and observation needs."""

    if not isinstance(action_window, dict) or not action_window:
        return {}
    return {
        "intent": action_window.get("intent"),
        "requires_intermediate_observation": action_window.get("requires_intermediate_observation"),
        "expected_effect": action_window.get("expected_effect"),
    }


def summarize_phase_progress(progress: dict[str, Any]) -> dict[str, Any]:
    """Return the phase progress fields most useful while watching the demo."""

    if not isinstance(progress, dict) or not progress:
        return {}
    measurements = progress.get("measurements", {})
    tool_facts = progress.get("tool_facts", {})
    return {
        "phase": progress.get("phase"),
        "phase_key": progress.get("phase_key"),
        "window_count": progress.get("window_count"),
        "max_window_count": progress.get("max_window_count"),
        "measurements": {
            "forward_m": measurements.get("forward_m"),
            "right_m": measurements.get("right_m"),
            "up_m": measurements.get("up_m"),
            "yaw_delta_deg": measurements.get("yaw_delta_deg"),
            "planar_m": measurements.get("planar_m"),
        },
        "tool_facts": {
            "successful_steps": tool_facts.get("successful_steps"),
            "move_duration_by_direction_success": tool_facts.get("move_duration_by_direction_success", {}),
            "turn_yaw_success_sum_deg": tool_facts.get("turn_yaw_success_sum_deg"),
        },
    }


def build_session_summary(run_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the session-level artifacts that best explain the pipeline.

    The latest model/parser/safety payloads are copied from events so someone
    reading demo/summary.json can quickly jump to the relevant full files.
    """

    events = load_events(run_dir / "events.jsonl")
    debug_events = load_events(run_dir / "debug" / "debug_events.jsonl")
    return {
        "run_dir": str(run_dir),
        "instructions": summaries,
        "latest_task_parse_result": latest_event_payload(debug_events, "task_parsed"),
        "latest_model_action_plan": latest_event_payload(debug_events, "model_action_planned"),
        "latest_validated_plan": latest_event_payload(debug_events, "short_horizon_planned"),
        "latest_safety_intervention": latest_event_payload(debug_events, "safety_intervened"),
        "latest_coupled_phase_decision": latest_event_payload(events, "coupled_planner_phase_decided"),
        "model_plan_recorded": latest_event_payload(debug_events, "model_action_planned") is not None,
        "validated_plan_recorded": latest_event_payload(debug_events, "short_horizon_planned") is not None,
        "automatic_observation_recorded": latest_event_payload(debug_events, "automatic_observation_updated") is not None,
        "action_sequence": action_sequence_from_events(debug_events) or action_sequence_from_events(events),
    }


def latest_event_payload(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    """Return the most recent payload for a named event."""

    for event in reversed(events):
        if event.get("event") == event_name:
            payload = event.get("payload")
            return dict(payload) if isinstance(payload, dict) else {}
    return None


def latest_instruction_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return events belonging to the most recent natural-language instruction.

    Continuous runs reuse one run directory, so demo summaries must not count
    earlier instruction actions as part of the current instruction. The runtime
    writes `instruction_started` only to the compact event stream, while detailed
    tool events live in debug events; sorting by timestamp lets us slice both
    streams together.
    """

    ordered = sorted(events, key=lambda item: str(item.get("timestamp", "")))
    start_index = 0
    for index, event in enumerate(ordered):
        if event.get("event") in {"instruction_started", "task_parsed"}:
            start_index = index
    return ordered[start_index:]


def observation_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize which modalities were captured for the current instruction."""

    return {
        "has_pose": any(bool(obs.get("pose")) for obs in observations if isinstance(obs, dict)),
        "has_rgb": any("rgb" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "has_depth": any("depth" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "sources": [str(obs.get("source", "")) for obs in observations if isinstance(obs, dict) and obs.get("source")],
    }


def action_sequence_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract user-planned tool steps from compact and debug event streams.

    Runtime stabilization writes its own brake/settling events. Those are useful
    in debug logs but noisy in the user-facing demo summary. Prefer detailed
    `tool_call_completed` records from current runs, then old `step_completed`
    records, and fall back to compact `tool_step_completed` records only when
    no detailed debug stream is present.
    """

    actions = []
    detailed_events = [event for event in events if event.get("event") == "tool_call_completed"]
    old_step_events = [event for event in events if event.get("event") == "step_completed"]
    source_events = detailed_events or old_step_events or [event for event in events if event.get("event") == "tool_step_completed"]
    for event in source_events:
        if event.get("event") == "tool_call_completed":
            payload = event.get("payload", {})
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            tool = payload.get("tool") if isinstance(payload, dict) else None
            args = payload.get("args", {}) if isinstance(payload, dict) else {}
            ok = result.get("ok") if isinstance(result, dict) else None
        elif event.get("event") == "step_completed":
            payload = event.get("payload", {})
            step = payload.get("step", {}) if isinstance(payload, dict) else {}
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            tool = step.get("tool")
            args = step.get("args", {})
            ok = result.get("ok") if isinstance(result, dict) else None
        else:
            payload = event.get("payload", {})
            step = {}
            tool = payload.get("tool") if isinstance(payload, dict) else None
            args = {}
            ok = payload.get("ok") if isinstance(payload, dict) else None
        if not tool:
            continue
        actions.append(
            {
                "phase": payload.get("phase"),
                "tool": tool,
                "args": args,
                "ok": ok,
            }
        )
    return actions


def action_windows_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract action-window declarations from phase plan/window events."""

    windows = []
    for event in events:
        if event.get("event") not in {"phase_plan_ready", "phase_window_recorded", "coupled_planner_phase_decided"}:
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        window = payload.get("action_window")
        if isinstance(window, dict) and window:
            windows.append(window)
    return _dedupe_action_windows(windows)


def agent_messages_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract model-generated user replies from compact runtime events."""

    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "agent_message":
            continue
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        message = payload.get("message", {}) if isinstance(payload.get("message", {}), dict) else {}
        if message:
            messages.append({"source": payload.get("source"), "phase": payload.get("phase"), **message})
    return messages


def _dedupe_action_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for window in windows:
        signature = json.dumps(window, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(window)
    return deduped


def observations_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract observation payloads written by preflight and automatic sampling."""

    observations = []
    for event in events:
        if event.get("event") not in {"initial_observation_updated", "instruction_start_observation_updated", "automatic_observation_updated"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            observations.append(payload)
    return observations


def artifact_inventory_from_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect RGB/depth artifact paths and existence checks for demo reports."""

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
    """Return shutdown-related event names from both user and debug logs."""

    events = load_events(run_dir / "events.jsonl") + load_events(run_dir / "debug" / "debug_events.jsonl")
    return [
        event["event"]
        for event in events
        if str(event.get("event", "")).startswith("shutdown_")
    ]




def load_events(path: Path) -> list[dict[str, Any]]:
    """Read JSONL events defensively; missing logs mean an empty event stream."""

    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write UTF-8 JSON so Chinese instructions and model reasons remain readable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
