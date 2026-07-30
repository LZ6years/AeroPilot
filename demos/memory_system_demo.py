"""Run a local memory-system demo with two continuous instructions."""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.semantic.task_parser import TaskParser
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK, CONTINUE_CURRENT_PHASE, PhaseTransition
from src.agent.planning.schemas import ActionWindow, CoupledActionPhasePlan, PlannedStep
from src.env_client import SimulatorEnvClient
from src.env_client.schemas import MapPackage
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.task_runtime import TaskRuntime
from src.sim_server.backends.airsim_backend import AirSimBackend
from src.sim_server.backends.registry import BackendRegistry
from src.sim_server.server import SimulationServer


# This demo is the lightest readable entry point for the memory/context layer.
# It intentionally avoids real model calls and real UE/AirSim execution. The
# output is therefore stable: you can rerun it and focus on how records, refs,
# fragments, and latest_context.json are produced.
LAMP_TASK = "前面右转会有一个黑色的路灯，右转面向路灯后停下"
ENTRANCE_TASK = "左转寻找建筑入口，并停在入口前"


class DemoHoverPlanner:
    """Deterministic planner used only to exercise the memory-writing path.

    This is not meant to be a smart planner. It emits one safe ``uav.hover``
    step the first time it sees a phase, then advances on the next window. That
    keeps the demo focused on TaskRuntime -> FlightMemory -> Context Modules ->
    ContextComposer rather than on model quality or UAV control.
    """

    planner_mode = "demo"

    def __init__(self) -> None:
        self._phase_calls: dict[str, int] = {}

    def plan_next_with_phase_decision(self, state, context=None):
        phase = state.current_phase_name
        calls = self._phase_calls.get(phase, 0)
        self._phase_calls[phase] = calls + 1
        if calls > 0:
            decision = COMPLETE_TASK if state.is_final_phase else ADVANCE_TO_NEXT_PHASE
            return CoupledActionPhasePlan(
                phase,
                PhaseTransition(decision, "memory demo phase evidence is sufficient"),
                [],
            )
        return CoupledActionPhasePlan(
            phase,
            PhaseTransition(CONTINUE_CURRENT_PHASE, "memory demo records one safe action window"),
            [PlannedStep.tool_step("uav.hover", {}, "memory demo holds position")],
            action_window=ActionWindow(
                intent="record one predictable memory demo action",
                expected_effect="memory demo writes one safe hover action and refreshed observation",
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic local memory-system demo")
    parser.add_argument("--run-root", default="runs", help="Directory where the demo run folder will be created.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Every run gets an isolated directory so previous memory/context examples
    # are never overwritten while you are comparing outputs.
    run_dir = Path(args.run_root) / ("memory_system_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    # AsyncRunLogger is the same artifact writer used by real runs. Keeping it
    # here makes this local demo produce the same directory shape as the real
    # engine demo.
    recorder = AsyncRunLogger(run_dir)

    # This educational demo deliberately injects an in-memory Backend. The
    # production SimulationServer defaults to managed real-engine execution;
    # keeping the fake explicit prevents a learning demo from silently changing
    # the production process-ownership contract.
    server = SimulationServer(
        [_map_package()],
        run_dir=run_dir,
        backend_registry=_memory_demo_backend_registry(),
    )

    # This is the real orchestration class. Only the parser/planner are swapped
    # for deterministic demo components.
    runtime = TaskRuntime(
        SimulatorEnvClient(server),
        recorder,
        TaskParser(mode="rule_based"),
        planner=DemoHoverPlanner(),
        shutdown_policy="none",
    )

    # Two instructions share one scene so you can verify that memory grows
    # across instructions while latest_state still represents the current run
    # state rather than storing all history.
    first_state = runtime.run_task("m1", LAMP_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)
    second_state = runtime.run_task("m1", ENTRANCE_TASK, scene_policy="reuse_running", stop_scene_on_finish=False)

    # Because the scene was reused, the runtime needs an explicit finish call.
    # This mirrors interactive sessions where the user may send several
    # instructions before closing the scene.
    runtime.finish_reused_scene("m1", second_state)
    recorder.close()

    # Read back the files that matter most when learning the memory system:
    # latest_context.json is the model-facing summary, while flight_memory.json
    # is the manifest of JSONL refs used by context modules.
    context = _read_json(run_dir / "contexts" / "latest_context.json")
    flight_memory = _read_json(run_dir / "state" / "flight_memory.json")
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    debug_events_path = run_dir / "debug" / "debug_events.jsonl"
    if debug_events_path.exists():
        events.extend(json.loads(line) for line in debug_events_path.read_text(encoding="utf-8").splitlines())
    summary = {
        "run_dir": str(run_dir),
        "first_completed": first_state.completed,
        "second_completed": second_state.completed,
        # Quick lifecycle smoke check for this reused local scene.
        "event_counts": {
            "scene_started": _count(events, "scene_started"),
            "scene_reused": _count(events, "scene_reused"),
            "control_connected": _count(events, "control_connected"),
            "preflight_takeoff_completed": _count(events, "preflight_takeoff_completed"),
            "scene_stopped": _count(events, "scene_stopped"),
        },
        "active_instruction": context["active_instruction"],
        # Fragment order should match ContextComposer's priority ordering.
        "fragment_names": [item["name"] for item in context["fragments"]],
        # These refs point into memory/instructions.jsonl. The manifest does not
        # duplicate full historical records.
        "instruction_history_refs": flight_memory["instruction_history"],
        "last_recent_pose_ref": flight_memory["recent_pose_refs"][-1] if flight_memory["recent_pose_refs"] else None,
        "motion_history_summary": _motion_history_summary(context),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _map_package() -> MapPackage:
    # Minimal map definition that satisfies the normal map schema without
    # needing a real packaged UE process for this local memory demo.
    return MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "connection": {"mode": "local"},
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )


def _memory_demo_backend_registry() -> BackendRegistry:
    """Build the explicit in-memory Backend used only by this learning demo."""

    registry = BackendRegistry()
    registry.register(
        "airsim",
        lambda context: AirSimBackend(
            use_real_uav=False,
            action_config=context.action_config,
        ),
    )
    return registry


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _count(events: list[dict], event_name: str) -> int:
    return sum(1 for item in events if item["event"] == event_name)


def _motion_history_summary(context: dict) -> dict:
    for fragment in context["fragments"]:
        if fragment["name"] == "action":
            return fragment["content"].get("motion_history_summary", {})
    return {}


if __name__ == "__main__":
    main()
