"""Real-model demo for FlyAgent context composition and action planning.
Build one instruction snapshot and request one native AssistantTurn.
This demo never executes tools. Running main explicitly calls the provider.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.agent.transcript import AgentTranscript
from src.llm.audit import snapshot_request
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader, MapMemoryStore
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.model_config import load_model_config
from src.tools.registry import default_registry


def build_demo_memory(run_dir: Path, instruction: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = MemoryEventWriter(run_dir)
    memory = FlightMemory("demo", "demo", "offline", "uav", memory_reader=MemoryRecordReader(run_dir))
    memory.update_instruction_ref(writer.write_instruction(instruction, {}), instruction)
    state = MissionState("demo", instruction)
    return state, memory, MapMemoryStore()


def redact_model_payload(payload, model_config):
    return {
        "api_key_present": bool(model_config.api_key),
        "model_config": model_config.redacted_summary(),
        "payload": snapshot_request(payload)[0],
    }


def run_demo(instruction, model_config, run_dir):
    state, memory, map_memory = build_demo_memory(run_dir, instruction)
    context = ObservationContextBuilder(map_memory=map_memory).build(state, memory)
    transcript = AgentTranscript(MemoryEventWriter(run_dir))
    transcript.start_instruction(instruction, memory.active_instruction_id)
    transcript.append_state(context, None)
    registry = default_registry()
    planner = ModelActionPlanner(model_config)
    turn = planner.plan_turn(transcript, registry)
    effect = validate_native_turn(turn, registry,
        allowed_names={spec.name for spec in registry.model_specs()}, latest_observation={})
    # Validation is not execution, and a plan is not evidence of task success.
    result = {"mode": "model_request_only_no_execution", "context": context,
              "assistant_turn": turn.to_dict(), "effect": effect,
              "exchange": planner.last_exchange}
    path = run_dir / "demo_request.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Explain the next safe action.")
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Use a new output directory")
    print(run_demo(args.instruction, load_model_config(args.env_path), output))


if __name__ == "__main__":
    main()
