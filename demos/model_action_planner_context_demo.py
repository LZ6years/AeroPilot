"""Real-model demo for FlyAgent context composition and action planning.

This demo is intentionally educational: it shows how the main classes are wired
when the planner path is fixed to real model calls. It does not execute UAV
tools; it only builds memory/context, calls ModelActionPlanner, validates the
model plan, and runs SafetySupervisor.

Recommended PowerShell setup for readable Chinese output:

    $env:PYTHONIOENCODING='utf-8'
    python demos/model_action_planner_context_demo.py --env-path .env
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.safety import SafetySupervisor
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.task_parser import TaskParser, _loads_json_object
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.model_config import ModelConfig, load_model_config


DEFAULT_INSTRUCTION = (
    "\u524d\u9762\u53f3\u8f6c\u4f1a\u6709\u4e00\u4e2a\u9ed1\u8272\u7684"
    "\u8def\u706f\uff0c\u53f3\u8f6c\u9762\u5411\u8def\u706f\u540e\u505c\u4e0b"
)
DEFAULT_MAP_CLUE = "\u8d77\u70b9\u53f3\u4fa7\u533a\u57df\u9644\u8fd1\u53ef\u80fd\u6709\u9ed1\u8272\u8def\u706f"
DEFAULT_REGION_HINT = "\u8d77\u70b9\u53f3\u4fa7"

DEMO_OUTPUT_FILES = {
    "state/semantic_plan.json",
    "state/latest_state.json",
    "state/flight_memory.json",
    "contexts/latest_context.json",
    "debug/model_payload.json",
    "debug/raw_model_response.json",
    "debug/raw_model_content.json",
    "debug/validated_plan.json",
    "debug/repair_attempts.json",
    "debug/repair_result.json",
    "debug/safety_decision.json",
    "demo/demo_preview.md",
}


@dataclass(frozen=True)
class DemoArtifacts:
    """All artifacts produced by one educational planner-context run.

    Keeping the intermediate objects together makes the demo easier to read:
    each output file below is written from one of these fields, so you can trace
    how a user instruction becomes semantic plan, memory refs, planner context,
    raw model JSON, validated plan, and safety decision.
    """

    run_dir: Path
    parse_result: TaskParseResult
    state: MissionState
    flight_memory: FlightMemory
    context: dict[str, Any]
    raw_model_response: dict[str, Any]
    raw_model_content: dict[str, Any]
    validated_plan: ShortHorizonPlan
    repair_attempts: list[dict[str, Any]]
    repair_result: dict[str, Any]
    safety_decision: dict[str, Any]
    model_payload: dict[str, Any]


class RecordingModelActionPlanner(ModelActionPlanner):
    """Capture the exact model request and response for demo inspection.

    The production planner only needs the parsed JSON plan. This subclass keeps
    the full HTTP payload and raw response so the demo can answer the practical
    question: "what exactly did we send to Qwen, and what did Qwen send back?"
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self.last_payload: dict[str, Any] = {}
        self.last_raw_response: dict[str, Any] = {}
        self.last_content = ""
        super().__init__(model_config)

    def _call_openai_compatible(self, context: dict[str, Any]) -> dict[str, Any]:
        endpoint = self.model_config.chat_completions_url()
        # The user message is the complete PlannerContext JSON. The system
        # prompt explains how to read that JSON and what action contract the
        # model must obey.
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        self.last_payload = payload

        # Only the request body is saved later. The Authorization header is
        # intentionally kept out of demo artifacts to avoid leaking API keys.
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model action planner request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Model action planner request failed: {exc}") from exc

        self.last_raw_response = data
        self.last_content = data["choices"][0]["message"]["content"]
        return _loads_json_object(self.last_content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-model FlyAgent context and action-planner demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-path", default=".env", help="Path to .env containing FLYAGENT_* model config.")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Instruction to parse and plan for.")
    parser.add_argument("--run-root", default="runs", help="Directory where the demo run folder will be created.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = load_model_config(args.env_path)
    run_dir = make_run_dir(args.run_root)

    artifacts = run_demo(
        instruction=args.instruction,
        model_config=model_config,
        run_dir=run_dir,
    )
    write_demo_outputs(artifacts)

    print(json.dumps(_console_summary(artifacts), indent=2, ensure_ascii=False))


def run_demo(instruction: str, model_config: ModelConfig, run_dir: Path) -> DemoArtifacts:
    # Stage 1: TaskParser uses the configured model to convert user language
    # into SemanticPlan. This is still semantic only: the parser should not emit
    # UAV tools, because tools are decided later by ModelActionPlanner.
    parser = TaskParser(mode="model", model_config=model_config)
    parse_result = parser.parse(instruction)

    # Stage 2: FlightMemory is only a manifest; JSONL records are written by
    # MemoryEventWriter. This split is important: modules read small refs and
    # summaries, not a giant raw history object.
    state, flight_memory, map_memory = build_demo_memory(run_dir, parse_result)

    # Stage 3: ObservationContextBuilder runs the enabled modules, then
    # ContextComposer merges them into the exact PlannerContext sent to the
    # action planner.
    context = ObservationContextBuilder(map_memory=map_memory).build(state, flight_memory)

    # Stage 4: ModelActionPlanner calls the model, then validates schema/tool
    # contract locally. A valid model response is still only a proposal; it has
    # not touched the UAV.
    planner = RecordingModelActionPlanner(model_config)
    validated_plan = planner.plan_next(state, context)

    # Stage 5: SafetySupervisor is the final safety gate before any real UAV
    # tool would run. This demo stops here, so the output is safe to inspect
    # without a simulator.
    safety = SafetySupervisor().review(validated_plan, state)
    safety_decision = {
        "plan": safety.plan.to_dict(),
        "interventions": safety.interventions,
    }

    return DemoArtifacts(
        run_dir=run_dir,
        parse_result=parse_result,
        state=state,
        flight_memory=flight_memory,
        context=context,
        raw_model_response=planner.last_raw_response,
        raw_model_content=_loads_json_object(planner.last_content),
        validated_plan=validated_plan,
        repair_attempts=list(planner.last_repair_attempts),
        repair_result=dict(planner.last_repair_result),
        safety_decision=safety_decision,
        model_payload=redact_model_payload(planner.last_payload, model_config),
    )


def build_demo_memory(
    run_dir: Path,
    parse_result: TaskParseResult,
) -> tuple[MissionState, FlightMemory, MapMemoryStore]:
    # MemoryEventWriter/Reader give modules real refs to read, without running a simulator.
    writer = MemoryEventWriter(run_dir)
    reader = MemoryRecordReader(run_dir)

    # MissionState is the current runtime control state. It is intentionally
    # smaller than memory history; historical records live in JSONL files and
    # are referenced through FlightMemory.
    state = MissionState(str(run_dir.name), parse_result.instruction, parse_result.plan)
    state.connected = True
    state.airborne = True

    # FlightMemory stores refs such as memory/observations.jsonl#1. It lets each
    # context module decide what to read and summarize.
    flight_memory = FlightMemory(
        run_id=str(run_dir.name),
        map_id="m1",
        scene_instance_id="scene_1",
        agent_id="uav_1",
        active_instruction=parse_result.instruction,
        memory_reader=reader,
    )

    instruction_ref = writer.write_instruction(parse_result.instruction, parse_result.plan.to_dict())
    flight_memory.update_instruction_ref(instruction_ref, parse_result.instruction)

    # The initial observation seeds image/pose modules. Mock RGB makes the model plan conservatively.
    observation = {
        "scene_instance_id": "scene_1",
        "agent_id": "uav_1",
        "modalities": ["rgb", "depth", "pose"],
        "pose": {"x": 0.0, "y": 0.0, "z": -2.0, "yaw_deg": 0.0},
        "frames": {
            "rgb": {"path": "artifacts/rgb/start_rgb.png", "mock": True},
            "depth": {"path": "artifacts/depth/start_depth.npy", "mock": True},
        },
        "source": "instruction_start",
    }
    state.record_observation(observation)
    observation_write = writer.write_observation(observation)
    flight_memory.append_observation_ref(observation_write["observation_ref"])
    flight_memory.append_pose_ref(observation_write.get("pose_ref"))
    flight_memory.append_frame_refs(observation_write.get("frame_refs"))

    # The recent hover action gives ActionModule a real but safe action history item.
    hover_result = {
        "ok": True,
        "action": "hover",
        "message": "\u8d77\u59cb\u60ac\u505c\u7a33\u5b9a",
    }
    state.record_action("uav.hover", {}, hover_result)
    action_write = writer.write_action("uav.hover", {}, hover_result)
    flight_memory.append_action_ref(action_write["action_ref"])
    flight_memory.append_pose_ref(action_write.get("pose_ref"))

    # MapMemoryStore contributes a reconfirmable clue; it must enter map inferences, not facts.
    map_memory = MapMemoryStore()
    map_memory.add(
        {
            "map_id": "m1",
            "scene_id": "scene_1",
            "landmark_description": DEFAULT_MAP_CLUE,
            "region_hint": DEFAULT_REGION_HINT,
            "confidence": "medium",
            "requires_reconfirmation": True,
        }
    )
    return state, flight_memory, map_memory


def make_run_dir(run_root: str | Path) -> Path:
    run_dir = Path(run_root) / ("model_action_planner_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def redact_model_payload(payload: dict[str, Any], model_config: ModelConfig) -> dict[str, Any]:
    # The HTTP Authorization header is never stored; this adds only a boolean for config debugging.
    return {
        "api_key_present": bool(model_config.api_key),
        "model_config": model_config.redacted_summary(),
        "payload": dict(payload),
    }


def write_demo_outputs(artifacts: DemoArtifacts) -> None:
    # JSON files are the machine-readable artifacts. Read demo_preview.md first
    # when learning the flow, then open the individual JSON files when you need
    # exact payloads for debugging.
    writes = {
        "state/semantic_plan.json": artifacts.parse_result.to_dict(),
        "state/latest_state.json": artifacts.state.to_dict(),
        "state/flight_memory.json": artifacts.flight_memory.to_dict(),
        "contexts/latest_context.json": artifacts.context,
        "debug/model_payload.json": artifacts.model_payload,
        "debug/raw_model_response.json": artifacts.raw_model_response,
        "debug/raw_model_content.json": artifacts.raw_model_content,
        "debug/validated_plan.json": artifacts.validated_plan.to_dict(),
        "debug/repair_attempts.json": {"attempts": artifacts.repair_attempts},
        "debug/repair_result.json": artifacts.repair_result,
        "debug/safety_decision.json": artifacts.safety_decision,
    }
    for name, data in writes.items():
        write_json(artifacts.run_dir / name, data)
    preview_path = artifacts.run_dir / "demo" / "demo_preview.md"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(build_preview_markdown(artifacts), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_preview_markdown(artifacts: DemoArtifacts) -> str:
    # This Markdown file is intentionally redundant with the JSON artifacts. It
    # gives a single scrollable teaching view of the pipeline, while JSON keeps
    # the data available for tests and diffing.
    context = artifacts.context
    payload = artifacts.model_payload["payload"]
    return "\n".join(
        [
            "# Model Action Planner Demo",
            "",
            "## Class Call Chain",
            "",
            "TaskParser(model) -> MemoryEventWriter/FlightMemory -> ObservationContextBuilder -> "
            "ModelActionPlanner(model) -> SafetySupervisor",
            "",
            "## Model Config (redacted)",
            "```json",
            json.dumps(artifacts.model_payload["model_config"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## TaskParser Model Result",
            "```json",
            json.dumps(artifacts.parse_result.to_dict(), indent=2, ensure_ascii=False),
            "```",
            "",
            "## MissionState Snapshot",
            "",
            "MissionState is runtime execution state; FlightMemory is the manifest read by context modules.",
            "",
            "```json",
            json.dumps(artifacts.state.to_dict(), indent=2, ensure_ascii=False),
            "```",
            "",
            "## PlannerContext Summary",
            "```json",
            json.dumps(context_summary(context), indent=2, ensure_ascii=False),
            "```",
            "",
            "## ModelActionPlanner System Prompt",
            "```text",
            payload["messages"][0]["content"],
            "```",
            "",
            "## Raw Model Action JSON",
            "```json",
            json.dumps(artifacts.raw_model_content, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Validated Planner Output",
            "```json",
            json.dumps(artifacts.validated_plan.to_dict(), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Contract Repair",
            "```json",
            json.dumps(
                {
                    "result": artifacts.repair_result,
                    "attempts": artifacts.repair_attempts,
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## Phase Transition Boundary",
            "",
            "ShortHorizonPlan no longer carries a planner-side phase_transition. "
            "Runtime applies phase changes only from the post-action phase judge.",
            "",
            "```json",
            json.dumps(
                {
                    "planner_side_phase_transition": "removed",
                    "authoritative_events": [
                        "model_phase_transition_judged",
                        "phase_transition_applied",
                        "phase_transition_continued",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
            "## SafetySupervisor Result",
            "```json",
            json.dumps(artifacts.safety_decision, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Full User Content JSON",
            "```json",
            payload["messages"][1]["content"],
            "```",
        ]
    )


def context_summary(context: dict[str, Any]) -> dict[str, Any]:
    # Keep console/Markdown previews compact. The complete PlannerContext is
    # still written to contexts/latest_context.json.
    return {
        "active_instruction": context["active_instruction"],
        "current_phase": context["current_phase"],
        "semantic_goal": context["semantic_goal"],
        "context_schema_keys": list(context["context_schema"].keys()),
        "module_content_fields": list(context["context_schema"]["module_content_fields"].keys()),
        "fragment_names": [fragment["name"] for fragment in context["fragments"]],
        "safety_constraints": context["safety_constraints"],
        "allowed_tools": context["allowed_action_constraints"]["planner_tools"],
        "action_schema_names": list(context["allowed_action_constraints"].get("action_schemas", {})),
    }


def _console_summary(artifacts: DemoArtifacts) -> dict[str, Any]:
    return {
        "run_dir": str(artifacts.run_dir),
        "read_first": str(artifacts.run_dir / "demo" / "demo_preview.md"),
        "validated_plan": artifacts.validated_plan.to_dict(),
        "safety_interventions": artifacts.safety_decision["interventions"],
    }


if __name__ == "__main__":
    main()
