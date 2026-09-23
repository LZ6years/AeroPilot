"""Tests for the canonical chronological Agent transcript."""

from __future__ import annotations

import base64
import json

from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.stores import MemoryEventWriter
from src.tools.registry import default_registry


def test_transcript_projects_state_calls_and_results_in_chronological_order(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("find the person", "inst_0001")
    transcript.append_state(
        {
            "current_phase": {"name": "find_person", "done_when": ["person found"]},
            "allowed_action_constraints": {"planner_tools": ["uav.hover"]},
            "simulator_contract": {"private": "legacy provider field"},
            "runtime_control": {
                "available": ["semantic_plan_update"],
                "contracts": {"semantic_plan_update": "duplicate schema"},
                "status": {"semantic_plan_update": "pending"},
            },
            "fragments": [],
        },
        "obs_0001",
    )
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Confirm the current view.",
            [AgentToolCall("call_detect", "uav.hover", {})],
        )
    )
    transcript.append_tool_result(
        ToolResult("call_detect", "uav.hover", "ok", "stable")
    )

    messages = transcript.provider_messages(default_registry())

    assert [message["role"] for message in messages] == [
        "user",
        "user",
        "assistant",
        "tool",
    ]
    state_json = json.loads(messages[1]["content"][-1]["text"])
    assert "allowed_action_constraints" not in state_json
    assert "simulator_contract" not in state_json
    assert "allowed_action_constraints" in transcript.latest_context()
    assert "simulator_contract" in transcript.latest_context()
    assert state_json["runtime_control"] == {
        "status": {"semantic_plan_update": "pending"}
    }
    assert messages[2]["tool_calls"][0]["id"] == "call_detect"
    assert json.loads(messages[3]["content"])["status"] == "ok"
    assert transcript.latest_observation_id() == "obs_0001"




def test_transcript_keeps_images_in_memory_but_strips_binary_from_jsonl(tmp_path):
    image_data = "data:image/png;base64,AAAA"
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_state(
        {
            "current_phase": {"name": "inspect"},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": image_data,
                            "observation_id": "obs_0001",
                            "frame_id": "frame_0001",
                            "artifact_ref": "memory/frames.jsonl#1",
                        }
                    ],
                }
            ],
        },
        "obs_0001",
    )

    messages = transcript.provider_messages(default_registry())
    transcript_lines = (tmp_path / "memory" / "agent_transcript.jsonl").read_text(
        encoding="utf-8"
    )

    assert any(
        part.get("image_url", {}).get("url") == image_data
        for part in messages[1]["content"]
        if isinstance(part, dict)
    )
    assert image_data not in transcript_lines
    assert "[provider_media_rehydrated_from_artifact]" in transcript_lines
    assert "observation_id=obs_0001" in json.dumps(messages[1], ensure_ascii=False)
    assert "frame_id=frame_0001" in json.dumps(messages[1], ensure_ascii=False)


def test_transcript_projects_each_observation_image_once_in_time_order(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect over time", "inst_0001")

    def context(
        current_id: str,
        current_data: str,
        historical: list[dict[str, str]],
    ):
        return {
            "current_phase": {"name": "inspect"},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        *historical,
                        {
                            "type": "image",
                            "role": "current_observation",
                            "encoding": "data_url",
                            "data": current_data,
                            "observation_id": current_id,
                            "frame_id": f"frame_{current_id}",
                            "artifact_ref": f"memory/frames.jsonl#{current_id}",
                            "scene_width": 256,
                            "scene_height": 256,
                            "display_width": 256,
                            "display_height": 298,
                            "color_bar_height": 42,
                        },
                    ],
                }
            ],
        }

    def image(role: str, observation_id: str, data: str):
        return {
            "type": "image",
            "role": role,
            "encoding": "data_url",
            "data": data,
            "observation_id": observation_id,
            "frame_id": f"frame_{observation_id}",
            "artifact_ref": f"memory/frames.jsonl#{observation_id}",
            "scene_width": 256,
            "scene_height": 256,
            "display_width": 256,
            "display_height": 298,
            "color_bar_height": 42,
        }

    obs1 = "data:image/png;base64,OBS1"
    obs2 = "data:image/png;base64,OBS2"
    obs3 = "data:image/png;base64,OBS3"
    transcript.append_state(context("obs_0001", obs1, []), "obs_0001")
    transcript.append_state(
        context(
            "obs_0002",
            obs2,
            [image("phase_start_before_last_action_reference", "obs_0001", obs1)],
        ),
        "obs_0002",
    )
    transcript.append_state(
        context(
            "obs_0003",
            obs3,
            [
                image("phase_start_reference", "obs_0001", obs1),
                image("before_last_action_reference", "obs_0002", obs2),
            ],
        ),
        "obs_0003",
    )

    rendered = json.dumps(
        transcript.provider_messages(default_registry()), ensure_ascii=False
    )

    assert rendered.count(obs1) == 1
    assert rendered.count(obs2) == 1
    assert rendered.count(obs3) == 1
    assert rendered.index(obs1) < rendered.index(obs2) < rendered.index(obs3)
    assert rendered.count("HISTORY observation_id=obs_0001") == 1
    assert rendered.count("HISTORY observation_id=obs_0002") == 1
    assert rendered.count("CURRENT observation_id=obs_0003") == 1
    assert rendered.count("scene_size=256x256") == 3
    assert rendered.count("display_size=256x298") == 3
    assert rendered.count("bottom color bar is not scene content") == 3
    assert rendered.count("cannot establish an exact decimal distance") == 3


def test_transcript_defers_repeated_observation_image_to_its_last_checkpoint(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("measure", "inst_0001")
    image = {
        "type": "image",
        "role": "current_observation",
        "encoding": "data_url",
        "data": "data:image/png;base64,SAME",
        "observation_id": "obs_0001",
        "frame_id": "frame_0001",
    }
    context = {"fragments": [{"name": "image", "model_inputs": [image]}]}
    transcript.append_state(context, "obs_0001")
    transcript.append_assistant(AssistantTurn("turn_1", "Measure.", []))
    transcript.append_state(context, "obs_0001")

    messages = transcript.provider_messages(default_registry())
    rendered = json.dumps(messages, ensure_ascii=False)

    assert rendered.count(image["data"]) == 1
    assert "CURRENT observation_id=obs_0001" in rendered
    assert image["data"] not in json.dumps(messages[1], ensure_ascii=False)


def test_tool_result_provider_projection_hides_internal_audit_details():
    result = ToolResult(
        "call_depth",
        "skill.metric_depth.sample_region",
        "ok",
        data={
            "observation_id": "obs_0001",
            "distance_m": 2.9,
            "audit_details": {"mad_m": 0.07},
        },
    )

    assert result.to_dict()["data"]["audit_details"] == {"mad_m": 0.07}
    assert result.to_provider_dict()["data"] == {
        "observation_id": "obs_0001",
        "distance_m": 2.9,
    }


def test_tool_result_projects_path_based_visual_evidence_after_tool_role(tmp_path):
    evidence_path = tmp_path / "grounding_candidates.png"
    evidence_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    evidence_path.write_bytes(evidence_bytes)
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("measure", "inst_depth")
    transcript.append_assistant(
        AssistantTurn(
            "turn_depth",
            "Measure the selected surface.",
            [
                AgentToolCall(
                    "call_depth",
                    "skill.visual_grounding.locate",
                    {
                        "observation_id": "obs_0001",
                        "query": "the visible person",
                        "include_distance": True,
                    },
                )
            ],
        )
    )
    result = ToolResult(
        "call_depth",
        "skill.visual_grounding.locate",
        "ok",
        data={"observation_id": "obs_0001", "distance_m": 2.9},
        artifact_refs=[str(evidence_path)],
        model_inputs=[
            {
                "type": "image",
                "role": "visual_grounding_candidates",
                "observation_id": "obs_0001",
                "path": str(evidence_path),
                "mime_type": "image/png",
                "usage": "Confirm candidate_01 against the instruction.",
            }
        ],
    )
    transcript.append_tool_result(result)

    messages = transcript.provider_messages(default_registry())

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert json.loads(messages[2]["content"])["data"]["distance_m"] == 2.9
    assert "model_inputs" not in json.loads(messages[2]["content"])
    assert "TOOL EVIDENCE tool_call_id=call_depth" in messages[3]["content"][0]["text"]
    assert messages[3]["content"][1]["image_url"]["url"].endswith(
        base64.b64encode(evidence_bytes).decode("ascii")
    )
    assert result.to_dict()["model_inputs"][0]["path"] == str(evidence_path)


def test_transcript_terminal_event_is_audit_only_and_prior_refs_exclude_current_turn(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_state({"fragments": []}, "obs_0001")
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Need more evidence.",
            [AgentToolCall("call_prior", "uav.hover", {})],
        )
    )
    transcript.append_tool_result(
        ToolResult("call_prior", "uav.hover", "ok", result_ref="memory/actions.jsonl#1")
    )
    transcript.append_assistant(
        AssistantTurn(
            "assistant_2",
            "Cannot continue safely.",
            [AgentToolCall("call_terminal", "runtime.report_blocked", {})],
        )
    )
    prior_refs = transcript.prior_evidence_reference_ids()
    transcript.append_task_terminal({"stopped": True})

    assert prior_refs == {
        "obs_0001",
        "call_prior",
        "memory/actions.jsonl#1",
    }
    assert "task_terminal" not in json.dumps(
        transcript.provider_messages(default_registry()), ensure_ascii=False
    )


def test_fallback_intent_is_audited_but_not_fabricated_for_provider(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_assistant(
        AssistantTurn(
            "assistant_1",
            "Request uav.get_pose for observation obs_0001.",
            [AgentToolCall("call_pose", "uav.get_pose", {})],
            public_intent_source="tool_arguments_fallback",
        )
    )
    transcript.append_tool_result(
        ToolResult("call_pose", "uav.get_pose", "ok", "pose read")
    )

    messages = transcript.provider_messages(default_registry())
    assistant = next(item for item in messages if item["role"] == "assistant")

    assert assistant["content"] is None
    assert transcript.to_dict()["items"][1]["payload"] == {
        "turn_id": "assistant_1",
        "public_intent": "Request uav.get_pose for observation obs_0001.",
        "public_intent_source": "tool_arguments_fallback",
        "tool_calls": [
            {"tool_call_id": "call_pose", "tool_name": "uav.get_pose", "args": {}}
        ],
    }


def test_provider_rejection_is_persisted_but_omitted_from_provider_projection(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("inspect", "inst_0001")
    transcript.append_provider_rejection(
        {
            "error": "unknown provider tool",
            "finish_reason": "tool_calls",
            "content_present": False,
            "content": None,
            "tool_calls": [
                {
                    "tool_call_id": "bad_call",
                    "wire_name": "unknown_tool",
                    "arguments": {},
                    "arguments_status": "parsed",
                }
            ],
        }
    )
    transcript.append_protocol_feedback("Return a corrected ToolCall.")

    messages = transcript.provider_messages(default_registry())
    records = transcript.to_dict()["items"]

    assert [item["role"] for item in messages] == ["user", "user"]
    assert messages[-1]["content"] == "Return a corrected ToolCall."
    assert records[1]["kind"] == "provider_turn_rejected"
