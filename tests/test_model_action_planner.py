"""Contract tests for the native tool-call provider boundary."""

from __future__ import annotations

import hashlib
import http.client
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.agent.planning.model_action_planner import (
    AgentProtocolError,
    ModelActionPlanner,
    ModelActionPlannerConfigError,
)
from src.agent.transcript import AgentTranscript, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.stores import MemoryEventWriter
from src.agent.config import ModelConfig
from src.agent.context.compression import HistoryCompressionConfig, HistoryProjectionState
from src.agent.context.window_manager import ContextWindowManager
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
    envelope = InstructionEnvelope(
        instruction_id="inst_0001",
        protocol_context="",
        identity={"text": "fly safely"},
        user_preferences={},
        knowledge_catalog=(),
        static_module_guidance=(),
    )
    transcript.start_instruction("move toward the target", "inst_0001", envelope)
    fragment = ContextFragment("pose", {})
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id="checkpoint_0001",
            reason="instruction_start",
            observation_id="obs_0001",
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_0001", None, (), (fragment,)),
        )
    )
    return transcript


def test_native_planner_sends_tools_and_parses_canonical_call(tmp_path):
    planner = StubPlanner(
        response_with_calls(("call_1", "uav_turn_relative", {"yaw_deg": 10}))
    )
    registry = default_registry()

    turn = planner.plan_turn(make_transcript(tmp_path), registry)

    assert turn.tool_calls[0].tool_name == "uav.turn_relative"
    payload = planner.payloads[0]
    assert "tools" in payload
    assert "response_format" not in payload
    assert payload["messages"][-1]["role"] == "user"
    assert any(
        tool["function"]["name"] == "runtime_report_completed"
        for tool in payload["tools"]
    )


def test_planner_records_provider_usage_in_task_local_projection_state(tmp_path):
    response = response_with_calls(
        ("call_1", "uav_turn_relative", {"yaw_deg": 10})
    )
    response["usage"] = {"prompt_tokens": 321}
    planner = StubPlanner(response)
    planner.context_window_manager = ContextWindowManager(
        planner.history_projector,
        HistoryCompressionConfig(),
    )
    state = HistoryProjectionState()

    planner.plan_turn(
        make_transcript(tmp_path),
        default_registry(),
        history_projection_state=state,
    )

    assert state.last_actual_prompt_tokens == 321
    assert planner.last_exchange["history_projection"]["actual_input_tokens"] == 321


def test_planner_boundary_does_not_require_commit_tool(tmp_path):
    from src.tools.registry import ToolRegistry
    from src.tools.builtins import builtin_tools

    registry = ToolRegistry()
    for spec in builtin_tools():
        registry.register(spec)
    planner = StubPlanner(response_with_calls(("plan1", "runtime_update_plan", {
        "explanation": "Simple task", "steps": []}), content=None))
    turn = planner.plan_turn(make_transcript(tmp_path), registry)
    assert turn.tool_calls[0].tool_name == "runtime.update_plan"
    assert turn.public_intent_source == "tool_arguments_fallback"
    assert all(t["function"]["name"] != "runtime_commit_phase_turn" for t in planner.payloads[0]["tools"])


def test_tool_selection_prompt_and_descriptions_do_not_force_a_control_call(tmp_path):
    """The real request builder exposes guidance; direct actions remain valid."""
    planner = StubPlanner(response_with_calls(("direct", "uav_turn_relative", {"yaw_deg": 10}), content=None))
    registry = default_registry()
    transcript = make_transcript(tmp_path)
    history_before = transcript.provider_messages(registry)
    turn = planner.plan_turn(transcript, registry)
    request = planner.payloads[0]
    assert turn.tool_calls[0].tool_name == "uav.turn_relative"
    assert request["tool_choice"] == "auto"
    assert request["temperature"] == 0
    assert "response_format" not in request
    assert request["messages"][-len(history_before):] == history_before
    assert transcript.provider_messages(registry) == history_before
    assert request["tools"] == [spec.to_provider_tool() for spec in registry.model_specs()]
    assert "before the first physical action" in registry.get("runtime.update_plan").description
    system_text = "\n".join(message["content"] for message in request["messages"] if message["role"] == "system")
    assert "call runtime.update_plan before the first" in system_text
    assert "uav_hover" not in {tool["function"]["name"] for tool in request["tools"]}


def test_request_allowlist_cannot_expose_hover_and_includes_generic_prompts(tmp_path):
    planner = StubPlanner(response_with_calls(("turn", "uav_turn_relative", {"yaw_deg": 10})))
    planner.plan_turn(make_transcript(tmp_path), default_registry(),
                      allowed_tool_names={"uav.hover", "uav.turn_relative"})
    payload = planner.payloads[0]
    assert [tool["function"]["name"] for tool in payload["tools"]] == ["uav_turn_relative"]
    systems = [message["content"] for message in payload["messages"] if message["role"] == "system"]
    assert planner.tool_user_prompt in systems
    assert planner.tool_case_studies in systems


@pytest.mark.parametrize("stages", [
    ["Continue the remaining forward displacement using measured self-motion."],
    ["Inspect the obscured entrance so its relation to the road becomes visible."],
    ["move_forward"],  # Prompt quality is not a new structural rejection gate.
])
def test_plan_descriptions_and_updated_prompt_reach_provider_unchanged(tmp_path, stages):
    from src.tools.validation import validate_tool_args

    steps = [{"id": str(i), "description": description, "status": "in_progress" if i == 0 else "pending"}
             for i, description in enumerate(stages)]
    args = {"explanation": "Current evidence", "steps": steps}
    planner = StubPlanner(response_with_calls(
        ("strategy_call", "runtime_update_plan", args)))
    transcript = make_transcript(tmp_path)
    registry = default_registry()
    validate_tool_args(args, registry.get("runtime.update_plan").input_schema)
    turn = planner.plan_turn(transcript, registry)
    assert turn.tool_calls[0].args["steps"] == steps
    expected_prompt = Path("src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(
        encoding="utf-8").strip()
    assert any(message["role"] == "system" and message["content"].strip() == expected_prompt
               for message in planner.payloads[0]["messages"])
    transcript.append_assistant(turn)
    transcript.append_tool_result(ToolResult(
        "strategy_call", "runtime.update_plan", "ok", "Recorded by test fixture."))
    replay = transcript.provider_messages(registry)
    submitted = next(message for message in replay if message["role"] == "assistant")
    assert json.loads(submitted["tool_calls"][0]["function"]["arguments"])["steps"] == steps
    assert replay[-1]["tool_call_id"] == "strategy_call"


def test_native_planner_transport_retry_is_one_model_round_and_not_transcript(
    tmp_path, monkeypatch
):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return json.dumps(
                response_with_calls(("call_1", "uav_hover", {}))
            ).encode("utf-8")

    outcomes = [http.client.RemoteDisconnected("closed"), Response()]
    sleeps = []
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request.data)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(
        "src.llm.transport.urllib.request.urlopen", fake_urlopen
    )
    monkeypatch.setattr("src.llm.transport.time.sleep", sleeps.append)
    planner = ModelActionPlanner(
        ModelConfig(
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )
    transcript = make_transcript(tmp_path)
    registry = default_registry()
    messages_before = transcript.provider_messages(registry)

    turn = planner.plan_turn(transcript, registry, decision_index=6)

    assert turn.tool_calls[0].tool_name == "uav.hover"
    assert requests[0] == requests[1]
    assert json.loads(requests[0])["messages"][-1]["content"] == planner.task_progress_reminder
    assert transcript.provider_messages(registry) == messages_before
    assert sleeps == [2.0]
    assert planner.last_exchange["transport"] == {
        "attempt_count": 2,
        "retry_count": 1,
        "final_status": "success",
        "retries": [
            {
                "attempt": 1,
                "category": "connection",
                "error_type": "RemoteDisconnected",
                "delay_sec": 2.0,
            }
        ],
    }


def test_native_planner_captures_exact_public_exchange_and_hashed_media(tmp_path):
    output = BytesIO()
    Image.new("RGB", (256, 256), (10, 20, 30)).save(output, format="PNG")
    image_bytes = output.getvalue()
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(image_bytes)
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        "inspect",
        "inst_media",
        InstructionEnvelope(
            instruction_id="inst_media",
            protocol_context="",
            identity={"text": "fly safely"},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )
    media_ref = MediaRef(
        ref=str(image_path),
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        mime_type="image/png",
        observation_id="obs_media",
        frame_id="frame_media",
        role="current_observation",
        width=256,
        height=256,
        display_width=256,
        display_height=298,
    )
    fragment = ContextFragment(
        "image", {}, (media_ref.to_model_input(),)
    )
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id="checkpoint_0001",
            reason="instruction_start",
            observation_id="obs_media",
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint("obs_media", None, (), (fragment,)),
        )
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
    assert exchange["media"][0]["request_path"][:2] == ["messages", 5]
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
            "runtime_report_completed",
            {"reason": "stable", "evidence_refs": ["obs_0001"]},
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


@pytest.mark.parametrize("name,args", [
    ("runtime_update_plan", {"explanation": "Route", "steps": []}),
    ("runtime_report_completed", {"reason": "arrived", "evidence_refs": ["obs_0001"]}),
    ("uav_move_body", {"vx": 1, "vy": 0, "vz": 0, "duration": 1}),
])
def test_native_planner_summarizes_canonical_call_for_null_content(tmp_path, name, args):
    planner = StubPlanner(response_with_calls(("call1", name, args), content=None))
    registry = default_registry()
    turn = planner.plan_turn(make_transcript(tmp_path), registry)
    assert turn.public_intent_source == "tool_arguments_fallback"
    assert turn.public_intent == f"Request tool call(s): {registry.from_wire_name(name).name}."


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


@pytest.mark.parametrize('mode', ['', 'disabled', 'enabled'])
def test_thinking_mode_reaches_wire_and_public_option_audit(tmp_path, mode):
    planner = StubPlanner(response_with_calls(('call_1', 'uav_turn_relative', {'yaw_deg': 10})), thinking_mode=mode)
    planner.plan_turn(make_transcript(tmp_path), default_registry())
    payload = planner.payloads[0]
    if mode:
        assert payload['thinking'] == {'type': mode}
        assert planner.last_request['thinking'] == {'type': mode}
    else:
        assert 'thinking' not in payload
    assert 'reasoning_effort' not in payload
    assert payload['temperature'] == 0
