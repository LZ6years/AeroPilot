"""Test model-backed short-horizon action planning."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from dataclasses import replace
from pathlib import Path
import urllib.error

import pytest
from PIL import Image

from src.agent.mission_state import MissionState
from src.agent.planning.model_action_planner import (
    ModelActionPlanner,
    ModelActionPlannerConfigError,
)
from src.agent.planning.model_inputs import build_model_user_content
from src.agent.planning.model_repair_planner import ModelRepairPlannerError
from src.agent.planning.tool_contract_validator import (
    ToolContractValidationError,
    ToolContractValidator,
)
from src.agent.phase import ADVANCE_TO_NEXT_PHASE, COMPLETE_TASK
from src.agent.planning.schemas import CoupledActionPhasePlan, RuntimeControlAction, ShortHorizonPlan
from src.agent.semantic.schemas import SemanticPhase
from src.agent.semantic.task_parser import TaskParser
from src.runtime.configs.model_config import ModelConfig


def test_tool_contract_explicit_empty_planner_tools_forbids_all_actions():
    plan = ShortHorizonPlan.from_dict(
        {
            "phase": "phase_a",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.hover",
                    "args": {},
                    "reason": "hold position",
                }
            ],
        }
    )

    with pytest.raises(ToolContractValidationError) as exc_info:
        ToolContractValidator().validate(
            plan,
            "phase_a",
            {"allowed_action_constraints": {"planner_tools": []}},
        )

    assert exc_info.value.violation.code == "tool_not_allowed"
    assert exc_info.value.violation.expected == []


class StubModelActionPlanner(ModelActionPlanner):
    def __init__(
        self,
        response: dict,
        model_config: ModelConfig | None = None,
        *,
        max_repair_attempts: int = 0,
        repair_planner=None,
    ) -> None:
        self.response = dict(response)
        if not self.response.pop("_omit_action_window", False):
            self.response.setdefault("action_window", make_action_window())
        if not self.response.pop("_omit_agent_messages", False):
            self.response.setdefault("agent_messages", make_agent_messages())
        super().__init__(
            model_config
            or ModelConfig(
                task_parser_mode="model",
                model_name="stub",
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            max_repair_attempts=max_repair_attempts,
            repair_planner=repair_planner,
        )

    def _call_coupled_action_phase_openai_compatible(self, context: dict):
        response = dict(self.response)
        response.setdefault(
            "completion_assessment",
            {
                "done_when": [
                    {
                        "index": 0,
                        "status": "unknown",
                        "evidence": "the stub requires more evidence",
                    }
                ]
            },
        )
        return response


class StubRepairPlanner:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.contexts = []

    def repair(self, repair_context):
        self.contexts.append(repair_context)
        if not self.responses:
            raise AssertionError("unexpected repair call")
        response = dict(self.responses.pop(0))
        if not response.pop("_omit_action_window", False):
            response.setdefault("action_window", make_action_window(intent="repair current phase contract"))
        if not response.pop("_omit_agent_messages", False):
            response.setdefault("agent_messages", make_agent_messages(content="我会改用符合约束的动作继续。"))
        return response


def make_action_window(**overrides) -> dict:
    window = {
        "intent": "make current phase progress",
        "scope": "current_phase_only",
        "requires_intermediate_observation": False,
        "expected_effect": "the current phase should have measurable progress",
    }
    window.update(overrides)
    return window


def make_agent_messages(message_type: str = "progress", content: str = "我准备执行当前阶段的安全动作。") -> list[dict]:
    return [
        {
            "type": message_type,
            "content": content,
            "evidence_refs": [],
            "confidence": "medium",
            "requires_user_reply": False,
        }
    ]


def make_state() -> MissionState:
    result = TaskParser(mode="rule_based").parse("左转寻找建筑入口并停在入口前")
    result.plan.phases[0] = SemanticPhase(
        index=1,
        total=len(result.plan.phases),
        name="orient_leftward",
        goal="orient toward the left-side entrance search area",
        done_when=["left-side entrance search area is visible"],
        continue_when=["left-side area is not yet visible"],
        metrics=[],
    )
    state = MissionState("run_test", result.instruction, result.plan)
    state.record_observation({"source": "preflight", "modalities": ["rgb", "depth", "pose"]})
    return state


def make_context() -> dict:
    return {
        "current_phase": {
            "name": "orient_leftward",
            "goal": "orient toward the left-side entrance search area",
            "done_when": ["left-side entrance search area is visible"],
            "continue_when": ["left-side area is not yet visible"],
            "metrics": [],
        },
        "observation_policy": {
            "mode": "automatic",
            "modalities": ["rgb", "depth", "pose"],
        },
        "allowed_action_constraints": {
            "planner_tools": ["uav.hover", "uav.turn_relative", "uav.move_body"],
            "action_schemas": {
                "uav.hover": {"args": {}},
                "uav.turn_relative": {
                    "args": {"yaw_deg": "number"},
                    "limits": {"yaw_deg_min": -30.0, "yaw_deg_max": 30.0},
                },
                "uav.move_body": {
                    "args_one_of": [
                        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"},
                    ],
                    "limits": {
                        "abs_vx_vy_max": 2.0,
                        "abs_vz_max": 1.0,
                        "duration_max": 2.0,
                    },
                },
            },
        },
    }


def make_multimodal_context() -> dict:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(buffer, format="PNG")
    image_data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    context = make_context()
    context["fragments"] = [
        {
            "name": "image",
            "model_inputs": [
                {
                    "type": "image",
                    "role": "phase_start_reference",
                    "usage": "PAST | PHASE START. Comparison only.",
                    "encoding": "data_url",
                    "data": image_data,
                },
                {
                    "type": "image",
                    "role": "current_observation",
                    "usage": "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions.",
                    "encoding": "data_url",
                    "data": image_data,
                },
                {
                    "type": "text",
                    "role": "temporal_evidence_rule",
                    "text": (
                        "Current-state claims must use NOW only. "
                        "PAST is only for comparing change."
                    ),
                },
            ],
        }
    ]
    return context


class FakeHTTPResponse:
    def __init__(self, content: str = "{}") -> None:
        self.body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self.body


def make_http_error(status: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test/v1/chat/completions",
        status,
        "Bad Request",
        {},
        BytesIO(body.encode("utf-8")),
    )


def make_tool_plan(
    planner: ModelActionPlanner,
    state: MissionState | None = None,
    context: dict | None = None,
) -> ShortHorizonPlan:
    coupled_plan = planner.plan_next_with_phase_decision(state or make_state(), context or make_context())
    return coupled_plan.to_short_horizon_plan()


def test_model_user_content_stays_text_without_model_inputs():
    context = make_context()
    context["safety_constraints"] = {"yaw_scan_remaining_deg": 10.0}
    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    assert text_context["current_phase"]["name"] == "orient_leftward"
    assert "safety_constraints" not in text_context
    assert "observation_policy" not in text_context


def test_model_action_request_success_does_not_retry(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        return FakeHTTPResponse()

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    assert planner._call_openai_compatible_json("system", make_multimodal_context()) == {}
    assert len(calls) == 1
    assert planner.last_model_request_recovery["status"] == "not_needed"
    assert planner.last_model_request_recovery["attempt_count"] == 1


@pytest.mark.parametrize(
    "first_error",
    [
        make_http_error(429, '{"error":{"code":"rate_limit"}}'),
        make_http_error(500, '{"error":{"code":"internal_error"}}'),
        urllib.error.URLError("temporary network failure"),
        TimeoutError("request timed out"),
    ],
)
def test_model_action_request_retries_transient_error_once(monkeypatch, first_error):
    responses = [
        first_error,
        FakeHTTPResponse(),
    ]

    def fake_urlopen(request, timeout):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    assert planner._call_openai_compatible_json("system", make_multimodal_context()) == {}
    assert responses == []
    recovery = planner.last_model_request_recovery
    assert recovery["status"] == "recovered"
    assert recovery["attempt_count"] == 2
    assert recovery["degraded_to_now_only"] is False


def test_model_action_request_stops_after_one_transient_retry(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise make_http_error(503, '{"error":{"code":"service_unavailable"}}')

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        planner._call_openai_compatible_json("system", make_multimodal_context())

    assert len(calls) == 2
    assert planner.last_model_request_recovery["status"] == "failed"
    assert planner.last_model_request_recovery["attempt_count"] == 2


def test_model_action_request_degrades_image_url_error_to_now_only(monkeypatch):
    provider_body = json.dumps(
        {
            "error": {
                "message": "The provided URL does not appear to be valid. Ensure it is correctly formatted.",
                "code": "invalid_parameter_error",
            }
        }
    )
    requests = []
    responses = [
        make_http_error(400, provider_body),
        make_http_error(400, provider_body),
        FakeHTTPResponse(),
    ]

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    assert planner._call_openai_compatible_json("system", make_multimodal_context()) == {}
    assert responses == []
    assert sum(
        part["type"] == "image_url"
        for part in requests[0]["messages"][1]["content"]
    ) == 2
    final_content = requests[-1]["messages"][1]["content"]
    assert sum(part["type"] == "image_url" for part in final_content) == 1
    final_text = "\n".join(
        part["text"] for part in final_content if part["type"] == "text"
    )
    assert "PAST | PHASE START" not in final_text
    assert "PAST is only for comparing change" not in final_text
    recovery = planner.last_model_request_recovery
    assert recovery["status"] == "recovered"
    assert recovery["attempt_count"] == 3
    assert recovery["degraded_to_now_only"] is True
    assert "base64" not in json.dumps(recovery)


def test_model_action_request_does_not_retry_image_url_error_with_invalid_local_data(monkeypatch):
    context = make_multimodal_context()
    context["fragments"][0]["model_inputs"][0]["data"] = "data:image/png;base64,not-valid"
    calls = []
    provider_body = (
        '{"error":{"message":"The provided URL does not appear to be valid.",'
        '"code":"invalid_parameter_error"}}'
    )

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise make_http_error(400, provider_body)

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    with pytest.raises(RuntimeError, match="HTTP 400"):
        planner._call_openai_compatible_json("system", context)

    assert len(calls) == 1
    assert planner.last_model_request_recovery["attempt_count"] == 1


def test_model_action_request_does_not_retry_unrelated_http_400(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise make_http_error(400, '{"error":{"code":"invalid_request"}}')

    monkeypatch.setattr(
        "src.agent.planning.model_action_planner.urllib.request.urlopen",
        fake_urlopen,
    )
    planner = ModelActionPlanner(
        ModelConfig(
            task_parser_mode="model",
            model_name="stub",
            api_key="test-key",
            base_url="https://example.test/v1",
        )
    )

    with pytest.raises(RuntimeError, match="HTTP 400"):
        planner._call_openai_compatible_json("system", make_multimodal_context())

    assert len(calls) == 1
    assert planner.last_model_request_recovery["status"] == "failed"
    assert planner.last_model_request_recovery["attempt_count"] == 1


def test_model_user_content_strips_legacy_move_direction_shorthand():
    context = make_context()
    move_schema = context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]
    move_schema["args_one_of"].append(
        {"direction": "legacy", "duration": "number"}
    )
    move_schema["argument_semantics"] = {
        "vx": "Positive moves forward; negative moves backward.",
        "vy": "Positive moves right; negative moves left.",
        "vz": "Positive moves up; negative moves down.",
    }
    context["phase_execution_summary"] = {
        "tool_facts": {
            "move_duration_by_direction_success": {"forward": 1.0},
            "move_duration_success_sum_sec": 1.0,
        }
    }
    context["fragments"] = [
        {
            "name": "action",
            "content": {
                "recent_actions": [{"tool": "uav.hover", "args": {}, "ok": True}],
                "action_trend": {"motion_pattern": {"type": "repeated_move"}},
                "motion_history_summary": {
                    "window_count": 1,
                    "phase_start_relative_uav_motion": {},
                    "recent_execution_order": "oldest_to_newest",
                    "recent_execution": [
                        {
                            "tool": "uav.move_body",
                            "args": {"direction": "forward", "duration": 1.0},
                        }
                    ],
                }
            },
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    text = json.dumps(text_context, ensure_ascii=False)
    assert '"direction"' not in text
    assert "move_duration_by_direction_success" not in text
    assert text_context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["args_one_of"] == [
        {"vx": "number", "vy": "number", "vz": "number", "duration": "number"}
    ]
    assert text_context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["argument_semantics"] == {
        "vx": "Positive moves forward; negative moves backward.",
        "vy": "Positive moves right; negative moves left.",
        "vz": "Positive moves up; negative moves down.",
    }
    assert "phase_execution_summary" not in text_context
    assert "recent_actions" not in text_context["fragments"][0]["content"]
    assert "action_trend" not in text_context["fragments"][0]["content"]
    assert text_context["fragments"][0]["content"]["motion_history_summary"]["recent_execution"][0]["args"] == {
        "duration": 1.0,
        "legacy_named_motion": True,
    }


def test_model_user_content_puts_flyagent_rules_before_task_fragments():
    context = make_context()
    context.update(
        {
            "rules": {"text": "FlyAgent baseline flight rules.", "source": "memory/FLYAGENT.md"},
            "active_instruction": "Turn toward the visual target.",
            "semantic_goal": {"target": "visual target"},
            "runtime_control": {
                "available": ["semantic_plan_update"],
                "contracts": {"semantic_plan_update": "Review the semantic contract when confidence is low."},
            },
            "fragments": [
                {
                    "name": "skills",
                    "content": {
                        "selected_skills": [
                            {
                                "name": "visual_target_task_reasoning",
                                "summary": "Use visual evidence carefully.",
                                "skill_text": "Full selected skill body.",
                            }
                        ]
                    },
                    "metadata": {"debug_only": True},
                }
            ],
            "phase_execution_summary": {"window_count": 1},
            "phase_progress_evidence": {"motion_progress": {"yaw_only_so_far": False}},
            "user_preferences": {"text": "Be concise."},
            "context_schema": {"planner_context_fields": {"rules": "Global FlyAgent behavior guidance."}},
            "metadata": {"run_id": "run_test"},
        }
    )

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    assert list(text_context)[:8] == [
        "rules",
        "active_instruction",
        "current_phase",
        "semantic_goal",
        "runtime_control",
        "fragments",
        "allowed_action_constraints",
        "user_preferences",
    ]
    assert "phase_progress_evidence" not in text_context
    assert "phase_execution_summary" not in text_context
    assert text_context["rules"]["text"] == "FlyAgent baseline flight rules."
    assert text_context["fragments"][0]["content"]["selected_skills"][0]["name"] == "visual_target_task_reasoning"
    assert "summary" not in text_context["fragments"][0]["content"]["selected_skills"][0]
    assert text_context["fragments"][0]["content"]["selected_skills"][0]["skill_text"] == "Full selected skill body."
    assert "metadata" not in text_context["fragments"][0]
    assert "context_schema" not in text_context
    assert "metadata" not in text_context


def test_model_user_content_strips_phase_judge_audit_fields():
    context = {
        "active_instruction": "Turn toward the person.",
        "semantic_goal": {"target": "person"},
        "current_phase": {
            "name": "orient_toward_person",
            "goal": "face the person",
            "done_when": ["person is visible", "view is roughly aligned"],
            "continue_when": ["person is not confirmed"],
            "metrics": [],
        },
        "phase_metrics": [],
        "phase_completion_evaluation": {"model_judged_count": 1},
        "phase_execution_summary": {"window_count": 1},
        "measurements": {"yaw_delta_deg": -5.0},
        "user_preferences": {"style": "concise"},
        "fragments": [{"name": "image", "model_inputs": []}],
        "planner_context_summary": {"fragment_names": ["image"], "safety_constraints": {"yaw_scan_remaining_deg": 340}},
        "action_window": {"intent": "scan", "expected_effect": "target centered"},
        "action_plan": {"steps": [{"tool": "uav.turn_relative", "args": {"yaw_deg": 5}}]},
        "safety_intervention_summary": {"reasons": ["yaw_scan_budget_exceeded"]},
        "yaw_scan_remaining_deg": 340,
        "visual_phase_budget_exhausted": False,
        "recent_executed_tools": ["uav.turn_relative"],
        "recent_safety_reasons": [],
        "safety_interventions": [],
        "recent_history": {"recent_action_refs": [{"ref": "memory/actions.jsonl#1"}]},
        "observation_before": {"pose": {"x": 0, "velocity": {"x": 1}}},
        "phase_window_count": 1,
        "max_phase_window_count": 100,
        "observation_after": {
            "source": "automatic_phase_window",
            "timestamp": "t1",
            "modalities": ["rgb", "depth", "pose"],
            "pose": {"x": 1},
            "frames": {"rgb": {"path": "frame.png"}, "depth": {"path": "depth.png"}},
        },
        "executed_steps": [
            {
                "phase": "orient_toward_person",
                "step_index": 0,
                "step": {"tool": "uav.turn_relative", "args": {"yaw_deg": -5}},
                "result": {"ok": True, "message": "turned", "raw": {"debug": True}},
                "stabilization": {"elapsed_sec": 1.0},
            }
        ],
    }

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    for debug_key in [
        "planner_context_summary",
        "action_window",
        "action_plan",
        "safety_intervention_summary",
        "yaw_scan_remaining_deg",
        "visual_phase_budget_exhausted",
        "recent_executed_tools",
        "recent_safety_reasons",
        "safety_interventions",
        "recent_history",
        "observation_before",
        "phase_window_count",
        "max_phase_window_count",
    ]:
        assert debug_key not in text_context
    assert text_context["active_instruction"] == "Turn toward the person."
    assert "continue_when" in context["current_phase"]
    assert text_context["current_phase"]["continue_when"] == ["person is not confirmed"]
    assert text_context["executed_steps"] == [
        {
            "phase": "orient_toward_person",
            "step_index": 0,
            "tool": "uav.turn_relative",
            "args": {"yaw_deg": -5},
            "ok": True,
            "message": "turned",
        }
    ]
    assert text_context["observation_after"] == {
        "source": "automatic_phase_window",
        "timestamp": "t1",
        "modalities": ["rgb", "depth", "pose"],
        "pose": {"x": 1},
        "has_rgb": True,
        "has_depth": True,
    }


def test_model_user_content_slims_fragments_and_observation_debug_fields():
    context = make_context()
    context.update(
        {
            "context_schema": {"debug": True},
            "metadata": {"run_id": "run_test"},
            "disabled_fragments": [],
            "phase_metrics": [],
            "phase_completion_evaluation": {
                "metrics_count": 0,
                "measurable_count": 0,
                "all_within_tolerance": False,
                "evaluations": [{"status": "requires_model_judgment"}],
            },
            "phase_execution_summary": {
                "phase": "landmark_phase",
                "window_count": 7,
                "measurements": {"forward_m": 10.0, "right_m": 2.0},
                "tool_facts": {"tools_executed": {"uav.move_body": 3}},
            },
            "latest_observation_summary": {
                "source": "automatic_phase_window",
                "timestamp": "t1",
                "modalities": ["rgb", "pose"],
                "pose": {"x": 1.0},
                "frames": {
                    "rgb": {
                        "path": "runs/example/frame.png",
                        "artifact_ref": "runs/example/frame.png",
                        "case_id": "case-debug",
                        "source": "uav_flow_benchmark",
                    }
                },
            },
            "fragments": [
                {
                    "name": "image",
                    "enabled": True,
                    "source": "DefaultImageContextModule",
                    "content": {"selected_frames": [{"role": "current_observation"}]},
                    "model_inputs": [],
                    "facts": [],
                    "inferences": [],
                    "goals": [],
                    "risks": [],
                    "artifact_refs": ["runs/example/frame.png"],
                    "budget": {"tokens_estimate": 300},
                    "confidence": "medium",
                    "warnings": [],
                    "metadata": {"debug_only": True},
                }
            ],
        }
    )

    content = build_model_user_content(context)

    assert isinstance(content, str)
    text_context = json.loads(content)
    text = json.dumps(text_context, ensure_ascii=False)
    assert "context_schema" not in text_context
    assert "metadata" not in text_context
    assert "disabled_fragments" not in text_context
    assert "phase_metrics" not in text_context
    assert "phase_completion_evaluation" not in text_context
    assert "phase_execution_summary" not in text_context
    assert "phase_progress_evidence" not in text_context
    assert "latest_observation_summary" not in text_context
    assert "fragments" not in text_context
    for forbidden in ["artifact_refs", "budget", "confidence", "source_path", "runs/example", "case-debug"]:
        assert forbidden not in text


def test_model_user_content_attaches_encoded_image_from_any_fragment():
    image_data = "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
    context = make_context()
    context["fragments"] = [
        {
            "name": "custom_visual",
            "content": {
                "visual_processing": {"strategy": "select_latest_rgb"},
                "visual_evidence": {"status": "debug_only"},
                "selected_frames": [{"role": "latest_observation"}],
                "interpretation_status": "image_attached_not_interpreted",
                "attached_image_count": 1,
            },
            "model_inputs": [
                {
                    "type": "image",
                    "role": "latest_observation",
                    "mime_type": "image/png",
                    "encoding": "data_url",
                    "data": image_data,
                    "metadata": {"source_path": "does-not-need-to-exist.png", "modality": "rgb"},
                }
            ],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0] == {"type": "text", "text": "Image role: latest_observation."}
    assert content[1] == {"type": "image_url", "image_url": {"url": image_data}}
    text_context = json.loads(content[2]["text"])
    assert text_context["current_phase"]["name"] == "orient_leftward"
    assert image_data not in content[2]["text"]
    assert text_context["fragments"] == [
        {
            "name": "custom_visual",
            "content": {"visual_processing": {"strategy": "select_latest_rgb"}},
        }
    ]


def test_model_user_content_inserts_label_before_semantic_image_input():
    image_data = "data:image/png;base64," + base64.b64encode(b"fake-png").decode("ascii")
    context = make_context()
    context["fragments"] = [
        {
            "name": "image",
            "model_inputs": [
                {
                    "type": "image",
                    "input_id": "image_1_current_observation",
                    "role": "current_observation",
                    "usage": "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions.",
                    "mime_type": "image/png",
                    "encoding": "data_url",
                    "data": image_data,
                }
            ],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert len(content) == 3
    assert content[0] == {
        "type": "text",
        "text": "NOW | CURRENT OBSERVATION. Use for current visibility and action decisions.",
    }
    assert content[1] == {"type": "image_url", "image_url": {"url": image_data}}
    text_context = json.loads(content[2]["text"])
    assert "fragments" not in text_context


def test_model_user_content_ignores_unencoded_image_path():
    context = make_context()
    context["fragments"] = [
        {
            "name": "image",
            "model_inputs": [{"type": "image", "path": "missing.png", "metadata": {"modality": "rgb"}}],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, str)
    assert json.loads(content)["current_phase"]["name"] == "orient_leftward"


def test_model_user_content_adds_text_model_inputs():
    context = make_context()
    context["fragments"] = [
        {
            "name": "visual_caption",
            "model_inputs": [{"type": "text", "role": "visual_caption", "text": "画面中疑似入口位于右前方。"}],
        }
    ]

    content = build_model_user_content(context)

    assert isinstance(content, list)
    assert json.loads(content[0]["text"])["current_phase"]["name"] == "orient_leftward"
    assert content[1] == {"type": "text", "text": "画面中疑似入口位于右前方。"}


def test_model_action_planner_rejects_legacy_action_prompt_override():
    with pytest.raises(ModelActionPlannerConfigError, match="coupled action/phase prompt only"):
        ModelActionPlanner(ModelConfig(model_name="stub", api_key="test-key"), prompt_path="legacy.txt")


def test_model_action_planner_combines_coupled_prompt_fragments_in_order():
    planner = ModelActionPlanner(ModelConfig(model_name="stub", api_key="test-key"))
    prompt = planner.coupled_action_phase_system_prompt

    overview_index = prompt.index("experimental coupled action-and-phase planner")
    phase_index = prompt.index("--- Phase Decision Contract ---")
    action_index = prompt.index("--- Action Planning Contract ---")

    assert overview_index < phase_index < action_index
    assert "Completion Contract" in prompt
    assert "Phase Decision Guide" not in prompt
    assert "Candidate Tools For Incomplete Assessments" in prompt
    assert (
        "The only tools available in this call are listed in "
        "allowed_action_constraints.planner_tools"
    ) in prompt
    assert "coupled_action_phase_planner_system.txt" not in prompt


def test_model_action_planner_requires_api_key():
    planner = ModelActionPlanner(ModelConfig(model_name="stub", api_key=""))

    with pytest.raises(ModelActionPlannerConfigError, match="API_KEY"):
        planner.plan_next_with_phase_decision(make_state(), {"current_phase": {"name": "orient_leftward"}})


def test_model_action_planner_accepts_open_phase_name():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.turn_relative",
                    "args": {"yaw_deg": -15},
                    "reason": "small left scan",
                },
            ],
        }
    )

    plan = make_tool_plan(planner, state, make_context())

    assert plan.phase == "orient_leftward"
    assert plan.steps[0].tool == "uav.turn_relative"
    assert plan.steps[0].args["yaw_deg"] == -15
    assert len(plan.steps) == 1
    assert "phase_transition" not in plan.to_dict()
    assert plan.agent_messages[0].type == "progress"


def test_coupled_action_phase_plan_round_trips_control_actions():
    plan = CoupledActionPhasePlan.from_dict(
        {
            "phase": "orient_leftward",
            "phase_decision": {"decision": "continue_current_phase", "reason": "target not aligned yet"},
            "action_window": make_action_window(),
            "agent_messages": make_agent_messages(),
            "control_actions": [
                {
                    "name": "semantic_plan_update",
                    "reason": "current phase contract is too strict for observed evidence",
                    "payload": {"reason": "completion evidence should use available observations"},
                }
            ],
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        },
        require_agent_messages=True,
    )

    assert plan.control_actions == [
        RuntimeControlAction(
            name="semantic_plan_update",
            reason="current phase contract is too strict for observed evidence",
            payload={"reason": "completion evidence should use available observations"},
        )
    ]
    assert plan.to_dict()["control_actions"][0]["name"] == "semantic_plan_update"
    assert plan.to_short_horizon_plan().to_dict().get("control_actions") is None


def test_short_horizon_plan_direct_constructor_has_no_planner_side_transition():
    plan = ShortHorizonPlan("orient_leftward", [])

    assert "phase_transition" not in plan.to_dict()
    assert plan.action_window.scope == "current_phase_only"
    assert plan.to_dict()["action_window"]["intent"]


def test_model_action_planner_missing_action_window_falls_back_to_hover():
    planner = StubModelActionPlanner(
        {
            "_omit_action_window": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.steps[0].tool == "uav.hover"
    assert planner.last_completion_result["fallback_reasons"] == ["continue_action_window_missing"]


def test_model_action_planner_accepts_missing_agent_messages_as_empty():
    planner = StubModelActionPlanner(
        {
            "_omit_agent_messages": True,
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.agent_messages == []
    assert planner.last_completion_result["fallback"] is False


def test_model_action_planner_rejects_invalid_action_window_scope():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "action_window": make_action_window(scope="whole_task"),
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="current_phase_only"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_multi_step_window_that_needs_observation():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "action_window": make_action_window(
                requires_intermediate_observation=True,
                expected_effect="need a fresh observation before the second step",
            ),
            "steps": [
                {"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": -10}, "reason": "scan"},
                {"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"},
            ],
        }
    )

    with pytest.raises(ValueError, match="exactly one step"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_unknown_tool():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.fake", "args": {}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_env_observe_tool_step():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "env.observe",
                    "args": {"modalities": ["rgb", "depth", "pose"]},
                    "reason": "bad observe",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_phase_mismatch():
    planner = StubModelActionPlanner(
        {
            "phase": "different_phase",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_yaw_above_context_limit():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.turn_relative",
                    "args": {"yaw_deg": -45.0},
                    "reason": "too large",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="below limit"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_observe_step():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "observe",
                    "modalities": ["rgb", "depth", "pose"],
                    "reason": "model should not request observation",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="runtime perception policy"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_hover_with_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {"duration": 1.0}, "reason": "bad"}],
        }
    )

    with pytest.raises(ValueError, match="uav.hover args must be"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_move_body_distance_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"x_m": 4.0, "y_m": -2.0, "duration_sec": 2.0},
                    "reason": "move using distance args",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="uav.move_body args must be exactly"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_model_facing_move_body_direction_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "backward", "duration": 1.5},
                    "reason": "legacy named motion",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_accepts_numeric_move_for_5m_nominal_scale():
    state = make_state()
    context = make_context()
    context["allowed_action_constraints"]["action_schemas"]["uav.move_body"]["limits"]["duration_max"] = 8.0
    context["allowed_action_constraints"]["nominal_action_scale"] = {
        "linear_velocity_mps": 2.0,
        "vertical_velocity_mps": 1.0,
    }
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"vx": 2.0, "vy": 0.0, "vz": 0.0, "duration": 2.5},
                    "reason": "translate about 5 meters using nominal 2 m/s scale",
                }
            ],
        }
    )

    plan = make_tool_plan(planner, state, context)

    assert plan.steps[0].args == {"vx": 2.0, "vy": 0.0, "vz": 0.0, "duration": 2.5}


def test_model_action_planner_accepts_contract_move_body_velocity_args_without_normalization():
    state = make_state()
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"vx": 1.0, "vy": -0.5, "vz": 0.25, "duration": 1.2},
                    "reason": "move with body-frame velocity",
                }
            ],
        }
    )

    plan = make_tool_plan(planner, state, make_context())

    assert plan.steps[0].args == {"vx": 1.0, "vy": -0.5, "vz": 0.25, "duration": 1.2}


def test_model_action_planner_rejects_legacy_move_body_alias_args():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "front", "duration": 1.0},
                    "reason": "legacy alias should be repaired, not normalized",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_legacy_move_body_duration_alias():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "forward", "duration_sec": 1.0},
                    "reason": "legacy duration alias should be repaired, not normalized",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_rejects_unknown_move_body_direction():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.move_body",
                    "args": {"direction": "diagonal", "duration": 1.0},
                    "reason": "bad direction",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="direction shorthand is not allowed"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_repairs_contract_violation_on_first_attempt():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "repaired hold"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    plan = make_tool_plan(planner, make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert repair_planner.contexts[0].attempt == 1
    assert planner.last_repair_result["status"] == "repaired"
    assert planner.last_repair_result["successful_attempt"] == 1


def test_model_action_planner_repairs_contract_violation_on_third_attempt():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 1.0}, "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "observe", "modalities": ["rgb"], "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "finally safe"}],
            },
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    plan = make_tool_plan(planner, make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert len(planner.last_repair_attempts) == 3
    assert planner.last_repair_result["successful_attempt"] == 3


def test_model_action_planner_repair_failure_exposes_structured_error():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 1.0}, "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 2.0}, "reason": "bad"}],
            },
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.move_body", "args": {"x_m": 3.0}, "reason": "bad"}],
            },
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError, match="exhausted") as exc_info:
        planner.plan_next_with_phase_decision(make_state(), make_context())

    data = exc_info.value.to_dict()
    assert data["reason"] == "max_repair_attempts_exhausted"
    assert data["max_repair_attempts"] == 3
    assert len(data["repair_attempts"]) == 3


def test_model_action_planner_repair_repeated_same_invalid_plan_fails_early():
    bad_plan = {
        "phase": "orient_leftward",
        "action_window": make_action_window(),
        "agent_messages": make_agent_messages(),
        "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
    }
    repair_planner = StubRepairPlanner([bad_plan, bad_plan])
    planner = StubModelActionPlanner(
        bad_plan,
        max_repair_attempts=3,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError, match="repeated"):
        planner.plan_next_with_phase_decision(make_state(), make_context())


def test_model_action_planner_repair_output_missing_action_window_fails():
    repair_planner = StubRepairPlanner(
        [
            {
                "_omit_action_window": True,
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "missing window"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError) as exc_info:
        planner.plan_next_with_phase_decision(make_state(), make_context())

    assert exc_info.value.violations[-1]["code"] == "repair_plan_parse_error"
    assert "requires action_window" in exc_info.value.violations[-1]["message"]


def test_model_action_planner_repair_output_missing_agent_messages_fails():
    repair_planner = StubRepairPlanner(
        [
            {
                "_omit_agent_messages": True,
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "missing messages"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "env.observe", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )

    with pytest.raises(ModelRepairPlannerError) as exc_info:
        planner.plan_next_with_phase_decision(make_state(), make_context())

    assert exc_info.value.violations[-1]["code"] == "repair_plan_parse_error"
    assert "requires agent_messages" in exc_info.value.violations[-1]["message"]


def test_model_action_planner_has_no_independent_phase_judge_method():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        }
    )

    assert not hasattr(planner, "judge_phase_transition")


def test_coupled_action_phase_plan_accepts_continue_with_tool_steps():
    plan = CoupledActionPhasePlan.from_dict(
        {
            "phase": "orient_leftward",
            "phase_decision": {"decision": "continue_current_phase", "reason": "target not aligned yet"},
            "action_window": make_action_window(),
            "agent_messages": make_agent_messages(),
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
        },
        require_agent_messages=True,
    )

    short_plan = plan.to_short_horizon_plan()

    assert plan.phase_decision.decision == "continue_current_phase"
    assert short_plan.steps[0].tool == "uav.hover"
    assert "phase_decision" in plan.to_dict()
    assert "phase_transition" not in plan.to_dict()


def test_coupled_action_phase_plan_accepts_advance_with_empty_steps():
    plan = CoupledActionPhasePlan.from_dict(
        {
            "phase": "orient_leftward",
            "phase_decision": {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "phase already complete"},
            "agent_messages": make_agent_messages(),
            "steps": [],
        }
    )

    assert plan.phase_decision.decision == ADVANCE_TO_NEXT_PHASE
    assert plan.steps == []


def test_coupled_action_phase_plan_rejects_transition_with_steps():
    with pytest.raises(ValueError, match="empty steps"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": COMPLETE_TASK, "reason": "done"},
                "agent_messages": make_agent_messages(),
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "bad"}],
            }
        )


def test_coupled_action_phase_plan_rejects_continue_without_steps():
    with pytest.raises(ValueError, match="1-3 tool steps"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": "continue_current_phase", "reason": "not done"},
                "action_window": make_action_window(),
                "agent_messages": make_agent_messages(),
                "steps": [],
            }
        )


def test_coupled_action_phase_plan_rejects_invalid_decision():
    with pytest.raises(ValueError, match="Unsupported phase_decision decision"):
        CoupledActionPhasePlan.from_dict(
            {
                "phase": "orient_leftward",
                "phase_decision": {"decision": "invalid_transition_request", "reason": "old design"},
                "agent_messages": make_agent_messages(),
                "steps": [],
            }
        )


def test_model_action_planner_coupled_continue_uses_tool_contract_validator():
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        }
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "phase_decision": {"decision": "continue_current_phase", "reason": "need one more action"},
        "action_window": make_action_window(),
        "agent_messages": make_agent_messages(),
        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "hold"}],
    }

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.steps[0].tool == "uav.hover"


def test_model_action_planner_coupled_continue_repairs_tool_contract_violation():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "action_window": make_action_window(
                    intent="repair to one safe observation window",
                    requires_intermediate_observation=True,
                    expected_effect="collect one new observation after holding",
                ),
                "agent_messages": make_agent_messages(content="我会改用一个安全的短动作继续确认。"),
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "single repaired step"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "completion_assessment": {
            "done_when": [
                {"index": 0, "status": "unknown", "evidence": "fresh evidence is still needed"}
            ]
        },
        "action_window": make_action_window(
            requires_intermediate_observation=True,
            expected_effect="bad multi-step plan asks for a mid-window observation",
        ),
        "agent_messages": make_agent_messages(),
        "steps": [
            {"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": -5}, "reason": "first"},
            {"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "second"},
        ],
    }

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.phase_decision.reason == "completion evidence still requires progress: 0=unknown"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "uav.hover"
    assert plan.action_window.requires_intermediate_observation is True
    assert planner.last_repair_result["status"] == "repaired"
    assert repair_planner.contexts[0].violation["code"] == "action_window_intermediate_observation_requires_single_step"


def test_model_action_planner_preserves_control_actions_after_tool_repair():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "repaired hold"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "control_actions": [
                {
                    "name": "semantic_plan_update",
                    "reason": "phase contract is blocking progress",
                    "payload": {"reason": "contract should be revisited"},
                }
            ],
            "steps": [{"kind": "tool", "tool": "uav.fake", "args": {}, "reason": "bad"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.steps[0].tool == "uav.hover"
    assert plan.control_actions[0].name == "semantic_plan_update"
    assert plan.control_actions[0].payload == {"reason": "contract should be revisited"}


def test_model_action_planner_supported_completion_discards_stale_steps_without_tool_repair():
    repair_planner = StubRepairPlanner(
        [
            {
                "phase": "orient_leftward",
                "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "should not be used"}],
            }
        ]
    )
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        },
        max_repair_attempts=1,
        repair_planner=repair_planner,
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "orient_leftward",
        "completion_assessment": {
            "done_when": [
                {"index": 0, "status": "supported", "evidence": "the phase goal is visibly satisfied"}
            ]
        },
        "control_actions": [
            {
                "name": "semantic_plan_update",
                "reason": "stale request",
                "payload": {"reason": "stale request"},
            }
        ],
        "agent_messages": make_agent_messages(),
        "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "bad transition action"}],
    }

    state = make_state()
    state.semantic_plan = replace(state.semantic_plan, phases=state.semantic_plan.phases[:1])
    plan = planner.plan_next_with_phase_decision(state, make_context())

    assert plan.phase_decision.decision == COMPLETE_TASK
    assert plan.steps == []
    assert plan.agent_messages == []
    assert plan.control_actions == []
    assert planner.last_completion_result["fallback_reasons"] == [
        "transition_steps_discarded",
        "transition_control_actions_discarded",
    ]
    assert repair_planner.contexts == []


def test_model_action_planner_supported_completion_advances_non_final_phase():
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "phase_decision": {
                "decision": "continue_current_phase",
                "reason": "the model is not allowed to choose this transition",
            },
            "completion_assessment": {
                "done_when": [
                    {
                        "index": 0,
                        "status": "supported",
                        "evidence": "the required visual condition is satisfied",
                    }
                ]
            },
            "steps": [],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == ADVANCE_TO_NEXT_PHASE
    assert plan.steps == []
    assert planner.last_completion_result["fallback_reasons"] == ["model_phase_decision_ignored"]


@pytest.mark.parametrize("status", ["unknown", "contradicted"])
def test_model_action_planner_incomplete_semantic_assessment_continues(status: str):
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "completion_assessment": {
                "done_when": [
                    {
                        "index": 0,
                        "status": status,
                        "evidence": "free text is explanatory only",
                    }
                ]
            },
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "refresh evidence"}],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.steps[0].tool == "uav.hover"
    assert planner.last_completion_result["semantic_complete"] is False


def test_model_action_planner_reports_the_incomplete_atomic_done_when_index():
    context = make_context()
    context["current_phase"]["done_when"] = [
        "the person is visually confirmed",
        "visual cues are broadly consistent with the requested distance",
    ]
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "completion_assessment": {
                "done_when": [
                    {
                        "index": 0,
                        "status": "supported",
                        "evidence": "the person is clear in NOW",
                    },
                    {
                        "index": 1,
                        "status": "unknown",
                        "evidence": "distance cues remain inconclusive",
                    },
                ]
            },
            "steps": [
                {
                    "kind": "tool",
                    "tool": "uav.hover",
                    "args": {},
                    "reason": "refresh distance evidence",
                }
            ],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), context)

    assert plan.phase_decision.decision == "continue_current_phase"
    assert plan.phase_decision.reason == "completion evidence still requires progress: 1=unknown"
    assert planner.last_completion_result["semantic_complete"] is False


def test_model_action_planner_completes_when_all_atomic_done_when_items_are_supported():
    context = make_context()
    context["current_phase"]["done_when"] = [
        "the person is visually confirmed",
        "visual cues are broadly consistent with the requested distance",
    ]
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "completion_assessment": {
                "done_when": [
                    {
                        "index": 0,
                        "status": "supported",
                        "evidence": "the person is clear in NOW",
                    },
                    {
                        "index": 1,
                        "status": "supported",
                        "evidence": "scale and scene cues support the requested distance",
                    },
                ]
            },
            "steps": [],
        }
    )
    state = make_state()
    state.semantic_plan = replace(state.semantic_plan, phases=state.semantic_plan.phases[:1])

    plan = planner.plan_next_with_phase_decision(state, context)

    assert plan.phase_decision.decision == COMPLETE_TASK
    assert planner.last_completion_result["semantic_complete"] is True


def test_model_action_planner_metric_gate_blocks_supported_semantic_completion():
    context = make_context()
    context["current_phase"]["metrics"] = [
        {
            "name": "yaw_delta",
            "target": -30.0,
            "unit": "deg",
            "tolerance": 2.0,
            "correction_allowed": True,
        }
    ]
    context["phase_completion_evaluation"] = {
        "metrics_count": 1,
        "measurable_count": 1,
        "all_within_tolerance": False,
        "evaluations": [{"status": "below_target"}],
    }
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "completion_assessment": {
                "done_when": [
                    {
                        "index": 0,
                        "status": "supported",
                        "evidence": "the semantic condition is satisfied",
                    }
                ]
            },
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "metric remains pending"}],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), context)

    assert plan.phase_decision.decision == "continue_current_phase"
    assert planner.last_completion_result["semantic_complete"] is True
    assert planner.last_completion_result["metric_complete"] is False


@pytest.mark.parametrize(
    "items",
    [
        [],
        [
            {"index": 0, "status": "supported", "evidence": "first"},
            {"index": 0, "status": "supported", "evidence": "duplicate"},
        ],
        [{"index": 1, "status": "supported", "evidence": "out of range"}],
        [{"index": 0, "status": "maybe", "evidence": "invalid status"}],
    ],
)
def test_model_action_planner_invalid_completion_assessment_falls_back_to_hover(items: list[dict]):
    planner = StubModelActionPlanner(
        {
            "phase": "orient_leftward",
            "completion_assessment": {"done_when": items},
            "steps": [{"kind": "tool", "tool": "uav.turn_relative", "args": {"yaw_deg": -10}, "reason": "stale"}],
        }
    )

    plan = planner.plan_next_with_phase_decision(make_state(), make_context())

    assert plan.phase_decision.decision == "continue_current_phase"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "uav.hover"
    assert planner.last_completion_result["status"] == "fallback"
    assert planner.last_completion_result["fallback_reasons"][0].startswith(
        "invalid_completion_assessment:"
    )


def test_model_action_planner_coupled_rejects_phase_mismatch_without_tool_steps():
    planner = StubModelActionPlanner(
        {
            "phase": "unused",
            "steps": [{"kind": "tool", "tool": "uav.hover", "args": {}, "reason": "unused"}],
        }
    )
    planner._call_coupled_action_phase_openai_compatible = lambda context: {
        "phase": "wrong_phase",
        "phase_decision": {"decision": ADVANCE_TO_NEXT_PHASE, "reason": "bad phase"},
        "agent_messages": make_agent_messages(),
        "steps": [],
    }

    with pytest.raises(ValueError, match="phase mismatch"):
        planner.plan_next_with_phase_decision(make_state(), make_context())
