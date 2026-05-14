"""Unit checks for the real-model action planner demo helpers."""

from __future__ import annotations

import json

from demos.demo_model_action_planner_context import (
    DEMO_OUTPUT_FILES,
    DemoArtifacts,
    build_demo_memory,
    redact_model_payload,
    write_demo_outputs,
)
from src.agent.mission_state import MissionState
from src.agent.planning.schemas import ShortHorizonPlan
from src.agent.semantic.schemas import SemanticPlan, TaskParseResult
from src.memory.flight_memory import FlightMemory
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.model_config import ModelConfig


def make_parse_result() -> TaskParseResult:
    plan = SemanticPlan.from_dict(
        {
            "intent": "navigate_by_visual_landmark",
            "goal": {
                "target": {"description": "black street lamp", "attributes": {"color": "black"}},
                "final_state": "uav_facing_target_and_hovering",
            },
            "phases": [
                {
                    "name": "orient_right",
                    "meaning": "turn right and search for the target landmark",
                }
            ],
            "uncertainties": ["target_identity_requires_observation"],
        }
    )
    return TaskParseResult("turn right and find the black street lamp", "model", plan)


def test_demo_declares_expected_output_files():
    assert DEMO_OUTPUT_FILES == {
        "semantic_parse_result.json",
        "mission_state.json",
        "flight_memory.json",
        "latest_context.json",
        "model_payload.json",
        "raw_model_response.json",
        "raw_model_content.json",
        "validated_plan.json",
        "repair_attempts.json",
        "repair_result.json",
        "safety_decision.json",
        "demo_preview.md",
    }


def test_demo_redacts_model_payload_api_key():
    config = ModelConfig(
        model_name="qwen-plus",
        api_key="secret-test-key",
        base_url="https://example.test/v1",
    )
    payload = {
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": "{}"}],
    }

    redacted = redact_model_payload(payload, config)

    assert redacted["api_key_present"] is True
    assert redacted["model_config"]["has_api_key"] is True
    assert "secret-test-key" not in json.dumps(redacted)


def test_demo_memory_builds_context_with_schema_fragments_and_action_contract(tmp_path):
    state, flight_memory, map_memory = build_demo_memory(tmp_path, make_parse_result())

    context = ObservationContextBuilder(map_memory=map_memory).build(state, flight_memory)

    assert isinstance(state, MissionState)
    assert isinstance(flight_memory, FlightMemory)
    assert "context_schema" in context
    assert "fragments" in context
    assert "allowed_action_constraints" in context
    assert "observation_policy" in context
    assert "phase_transition_policy" in context
    assert context["observation_policy"]["mode"] == "automatic"
    assert context["phase_transition_policy"]["mode"] == "model_suggested_runtime_applied"
    assert set(context["context_schema"]["module_content_fields"]) == {"instruction", "image", "pose", "action", "map"}
    assert [fragment["name"] for fragment in context["fragments"]] == ["instruction", "pose", "action", "image", "map"]
    assert context["allowed_action_constraints"]["planner_tools"] == ["uav.hover", "uav.turn_relative", "uav.move_body"]


def test_demo_writes_expected_artifact_files(tmp_path):
    parse_result = make_parse_result()
    state, flight_memory, map_memory = build_demo_memory(tmp_path, parse_result)
    context = ObservationContextBuilder(map_memory=map_memory).build(state, flight_memory)
    plan = ShortHorizonPlan.from_dict(
        {
            "phase": "orient_right",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold position"}],
        }
    )
    artifacts = DemoArtifacts(
        run_dir=tmp_path,
        parse_result=parse_result,
        state=state,
        flight_memory=flight_memory,
        context=context,
        raw_model_response={"choices": [{"message": {"content": "{}"}}]},
        raw_model_content=plan.to_dict(),
        validated_plan=plan,
        repair_attempts=[],
        repair_result={"repair_triggered": False},
        safety_decision={"plan": plan.to_dict(), "interventions": []},
        model_payload={
            "api_key_present": True,
            "model_config": {"model_name": "stub", "has_api_key": True},
            "payload": {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ]
            },
        },
    )

    write_demo_outputs(artifacts)

    assert {path.name for path in tmp_path.iterdir() if path.is_file()} >= DEMO_OUTPUT_FILES
    assert "context_schema" in json.loads((tmp_path / "latest_context.json").read_text(encoding="utf-8"))
    mission_state = json.loads((tmp_path / "mission_state.json").read_text(encoding="utf-8"))
    assert "recent_observations" in mission_state
    assert "recent_actions" in mission_state
    assert "current_phase" in mission_state
    preview = (tmp_path / "demo_preview.md").read_text(encoding="utf-8")
    assert "Model Action Planner Demo" in preview
    assert "Contract Repair" in preview
    assert "模型阶段推进判断" in preview
