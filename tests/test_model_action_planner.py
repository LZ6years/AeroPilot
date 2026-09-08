"""Contract tests for the native tool-call provider boundary."""

from __future__ import annotations

import base64
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


def test_native_planner_captures_exact_public_exchange_and_hashed_media(tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\nprovider-image"
    image_data = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_media")
    transcript.append_state(
        {
            "current_phase": {"name": "approach", "done_when": ["near"]},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": image_data,
                            "observation_id": "obs_media",
                            "frame_id": "frame_media",
                            "scene_width": 256,
                            "scene_height": 256,
                            "display_width": 256,
                            "display_height": 298,
                        }
                    ],
                }
            ],
        },
        "obs_media",
    )
    response = response_with_calls(("call_1", "uav_hover", {}), content=None)
    response["choices"][0]["finish_reason"] = "tool_calls"
    response["choices"][0]["message"]["reasoning_content"] = "private reasoning"
    response["usage"] = {"prompt_tokens": 12, "completion_tokens": 4}
    planner = StubPlanner(response)

    planner.plan_turn(transcript, default_registry())

    exchange = planner.last_exchange
    request_text = json.dumps(exchange["request"])
    response_text = json.dumps(exchange["response"])
    assert "data:image" not in request_text
    assert "provider_media sha256=" in request_text
    assert exchange["media"][0]["data"] == image_bytes
    assert exchange["media"][0]["request_path"][:2] == ["messages", 2]
    assert exchange["media"][0]["timeline_role"] == "CURRENT"
    assert exchange["media"][0]["observation_id"] == "obs_media"
    assert exchange["media"][0]["frame_id"] == "frame_media"
    assert exchange["media"][0]["scene_size"] == "256x256"
    assert exchange["media"][0]["display_size"] == "256x298"
    assert exchange["media"][0]["scene_bounds_in_display_pixels"] == (
        "x:[0,256) y:[0,256)"
    )
    assert exchange["response"]["message"]["content"] is None
    assert exchange["response"]["message"]["tool_calls"][0]["function"][
        "arguments"
    ] == "{}"
    assert exchange["response"]["usage"]["prompt_tokens"] == 12
    assert "private reasoning" not in response_text


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
        (
            response_with_calls(("call_empty", "", {})),
            "provider tool call call_empty has an empty function name",
        ),
        (response_with_calls(("same", "uav_hover", {}), ("same", "uav_hover", {})), "unique ids"),
        ({"choices": [{"message": {"content": "no tools"}}]}, "must call a tool"),
    ],
)
def test_native_planner_rejects_invalid_provider_turn(tmp_path, response, error):
    with pytest.raises(AgentProtocolError, match=error) as raised:
        StubPlanner(response).plan_turn(make_transcript(tmp_path), default_registry())

    assert raised.value.rejected_turn["error"]
    assert "reasoning_content" not in json.dumps(raised.value.rejected_turn)


def test_native_planner_accepts_null_content_with_read_only_fallback(tmp_path):
    planner = StubPlanner(
        response_with_calls(
            (
                "call_pose",
                "uav_get_pose",
                {"observation_id": "obs_0001"},
            ),
            content=None,
        )
    )

    turn = planner.plan_turn(make_transcript(tmp_path), default_registry())

    assert turn.public_intent_source == "tool_arguments_fallback"
    assert turn.public_intent == "Request uav.get_pose for observation obs_0001."


def test_native_planner_uses_action_window_intent_for_null_content(tmp_path):
    planner = StubPlanner(
        response_with_calls(
            (
                "call_commit",
                "runtime_commit_phase_turn",
                {
                    "phase": "approach",
                    "action_window": {
                        "intent": "move closer to the visible target",
                        "expected_effect": "the target appears larger",
                    },
                },
            ),
            ("call_move", "uav_move_body", {"vx": 1.0}),
            content=None,
        )
    )

    turn = planner.plan_turn(make_transcript(tmp_path), default_registry())

    assert turn.public_intent_source == "tool_arguments_fallback"
    assert turn.public_intent == "move closer to the visible target"


def test_native_planner_uses_phase_commit_fallback_for_null_content(tmp_path):
    planner = StubPlanner(
        response_with_calls(
            (
                "call_commit",
                "runtime_commit_phase_turn",
                {"phase": "land"},
            ),
            content=None,
        )
    )

    turn = planner.plan_turn(make_transcript(tmp_path), default_registry())

    assert turn.public_intent_source == "tool_arguments_fallback"
    assert turn.public_intent == "Commit phase decision for land."


def test_native_planner_rejection_snapshot_keeps_wire_attempt_only(tmp_path):
    response = response_with_calls(("call_1", "unknown_tool", {"value": 1}))
    response["choices"][0]["finish_reason"] = "tool_calls"
    response["choices"][0]["message"]["reasoning_content"] = "private chain"

    with pytest.raises(AgentProtocolError) as raised:
        StubPlanner(response).plan_turn(make_transcript(tmp_path), default_registry())

    snapshot = raised.value.rejected_turn
    assert snapshot["finish_reason"] == "tool_calls"
    assert snapshot["tool_calls"] == [
        {
            "tool_call_id": "call_1",
            "type": "function",
            "wire_name": "unknown_tool",
            "arguments": {"value": 1},
            "arguments_status": "parsed",
        }
    ]
    assert "private chain" not in json.dumps(snapshot)


def test_native_planner_rejection_snapshot_keeps_invalid_argument_text(tmp_path):
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad_json",
                            "type": "function",
                            "function": {
                                "name": "uav_hover",
                                "arguments": "{not-json",
                            },
                        }
                    ],
                },
            }
        ]
    }

    with pytest.raises(AgentProtocolError, match="not valid JSON") as raised:
        StubPlanner(response).plan_turn(make_transcript(tmp_path), default_registry())

    assert raised.value.rejected_turn["tool_calls"][0] == {
        "tool_call_id": "call_bad_json",
        "type": "function",
        "wire_name": "uav_hover",
        "arguments": "{not-json",
        "arguments_status": "invalid",
    }


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
