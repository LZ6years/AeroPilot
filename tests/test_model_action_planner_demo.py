"""Unit checks for the real-model action planner demo helpers."""
import json
from demos.model_action_planner_context_demo import build_demo_memory, redact_model_payload
from src.agent.mission_state import MissionState
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.registry import ContextModuleRegistry
from src.agent.config import ModelConfig


def test_demo_memory_needs_only_instruction(tmp_path):
    state, memory, map_memory = build_demo_memory(tmp_path, "Turn towards the entrance")
    assert isinstance(state, MissionState) and state.task_plan is None
    builder = ModelContextBuilder(
        registry=ContextModuleRegistry.default(), map_memory=map_memory
    )
    envelope = builder.build_instruction_envelope(state, memory)
    checkpoint, _ = builder.build_checkpoint(
        state, memory, checkpoint_id="checkpoint_0001", reason="instruction_start"
    )
    assert "Turn towards the entrance" not in json.dumps(envelope.to_dict())
    assert checkpoint.task_plan is None
    assert all(fragment.name != "instruction" for fragment in checkpoint.fragments)


def test_demo_redacts_model_payload_api_key():
    config = ModelConfig(api_key="secret-test-key")
    result = redact_model_payload({"authorization": "Bearer secret-test-key"}, config)
    assert result["api_key_present"]
    assert "secret-test-key" not in json.dumps(result)
