"""Unit checks for the real-model action planner demo helpers."""
import json
from demos.model_action_planner_context_demo import build_demo_memory, redact_model_payload
from src.agent.mission_state import MissionState
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.model_config import ModelConfig


def test_demo_memory_needs_only_instruction(tmp_path):
    state, memory, map_memory = build_demo_memory(tmp_path, "Turn towards the entrance")
    assert isinstance(state, MissionState) and state.task_plan is None
    context = ObservationContextBuilder(map_memory=map_memory).build(state, memory)
    assert context["active_instruction"] == "Turn towards the entrance"
    assert "current_phase" not in context and "semantic_plan" not in context
    assert "allowed_action_constraints" not in context


def test_demo_redacts_model_payload_api_key():
    config = ModelConfig(api_key="secret-test-key")
    result = redact_model_payload({"authorization": "Bearer secret-test-key"}, config)
    assert result["api_key_present"]
    assert "secret-test-key" not in json.dumps(result)
