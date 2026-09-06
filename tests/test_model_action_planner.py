"""Contract tests for the native tool-call provider boundary."""

from __future__ import annotations

import json

import pytest

from src.agent.planning.model_action_planner import (
    AgentProtocolError,
    ModelActionPlanner,
    ModelActionPlannerConfigError,
)
from src.agent.transcript import AgentTranscript, ToolResult
from src.memory.stores import MemoryEventWriter
from src.runtime.configs.model_config import ModelConfig
from src.tools.registry import default_registry


class StubPlanner(ModelActionPlanner):
    def __init__(self, response: dict, **config_overrides) -> None:
        config = ModelConfig(
            model_name="stub",
            api_key=config_overrides.pop("api_key", "test-key"),
            base_url="https://example.test/v1",
            **config_overrides,
        )
        super().__init__(config)
        self.response = response
        self.payloads = []

    def _request(self, payload):
        self.payloads.append(payload)
        return self.response


def response_with_calls(*calls, content="Inspect and act from the current evidence."):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                        for call_id, name, args in calls
                    ],
                }
            }
        ]
    }


def make_transcript(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("move toward the target", "inst_0001")
    transcript.append_state(
        {
            "current_phase": {"name": "approach", "done_when": ["target is near"]},
            "fragments": [],
            "rules": {"text": "fly safely"},
        },
        "obs_0001",
    )
    return transcript


def test_native_planner_sends_tools_and_parses_canonical_call(tmp_path):
    planner = StubPlanner(
        response_with_calls(("call_1", "uav_hover", {}))
    )
    registry = default_registry()

    turn = planner.plan_turn(make_transcript(tmp_path), registry)

    assert turn.tool_calls[0].tool_name == "uav.hover"
    payload = planner.payloads[0]
    assert "tools" in payload
    assert "response_format" not in payload
    assert payload["messages"][-1]["role"] == "user"
    assert any(
        tool["function"]["name"] == "runtime_commit_phase_turn"
        for tool in payload["tools"]
    )


def test_native_planner_replays_tool_result_in_full_transcript(tmp_path):
    transcript = make_transcript(tmp_path)
    planner = StubPlanner(response_with_calls(("call_1", "uav_hover", {})))
    registry = default_registry()
    turn = planner.plan_turn(transcript, registry)
    transcript.append_assistant(turn)
    transcript.append_tool_result(
        ToolResult("call_1", "uav.hover", "ok", "hovered")
    )
    planner.response = response_with_calls(
        (
            "call_2",
            "runtime_commit_phase_turn",
            {
                "phase": "approach",
                "completion_assessment": {
                    "done_when": [
                        {"index": 0, "status": "supported", "evidence": "stable"}
                    ]
                },
            },
        )
    )

    planner.plan_turn(transcript, registry)

    messages = planner.payloads[-1]["messages"]
    assert [message["role"] for message in messages[-5:]] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[-1]["tool_call_id"] == "call_1"


@pytest.mark.parametrize(
    "response,error",
    [
        (response_with_calls(("call_1", "unknown_tool", {})), "unknown provider tool"),
        (response_with_calls(("same", "uav_hover", {}), ("same", "uav_hover", {})), "unique ids"),
        (response_with_calls(("call_1", "uav_hover", {}), content=None), "public intent"),
        ({"choices": [{"message": {"content": "no tools"}}]}, "must call a tool"),
    ],
)
def test_native_planner_rejects_invalid_provider_turn(tmp_path, response, error):
    with pytest.raises(AgentProtocolError, match=error):
        StubPlanner(response).plan_turn(make_transcript(tmp_path), default_registry())


def test_native_planner_requires_api_key(tmp_path):
    planner = StubPlanner(response_with_calls(("call_1", "uav_hover", {})), api_key="")
    with pytest.raises(ModelActionPlannerConfigError, match="API_KEY"):
        planner.plan_turn(make_transcript(tmp_path), default_registry())


def test_old_repair_planner_configuration_is_rejected():
    with pytest.raises(ModelActionPlannerConfigError, match="separate repair planners"):
        ModelActionPlanner(
            ModelConfig(model_name="stub", api_key="key"),
            max_repair_attempts=1,
        )


def test_model_action_planner_has_no_legacy_json_plan_method():
    planner = StubPlanner(response_with_calls(("call_1", "uav_hover", {})))
    assert not hasattr(planner, "plan_next_with_phase_decision")
    assert not hasattr(planner, "_request_openai_compatible_json")


def test_model_action_planner_uses_configured_native_prompt(tmp_path):
    prompt_path = tmp_path / "native_prompt.txt"
    prompt_path.write_text("custom native tool protocol", encoding="utf-8")

    planner = StubPlanner(
        response_with_calls(("call_1", "uav_hover", {})),
        action_planner_prompt_path=str(prompt_path),
    )

    assert planner.system_prompt == "custom native tool protocol"
