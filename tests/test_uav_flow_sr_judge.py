"""Tests for UAV-Flow semantic SR judge metric guardrails."""

from __future__ import annotations

import json

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.sr_judge import judge_success, metric_success_assessment, _payload
from src.agent.config import ModelConfig


def make_case(class_name: str = "Move") -> UavFlowCase:
    return UavFlowCase(
        case_id="case",
        file_name="case.json",
        source_path="case.json",
        class_name=class_name,
        instruction="Move to a position 3.5 meters from the person",
        instruction_unified="Move to a position 3.5 meters from the person",
        selected_instruction="Move to a position 3.5 meters from the person",
        initial_pos=[0, 0, 100, 0, 0, 0],
        target_pos=[350, 0, 0, 0, 0, 0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0], [350, 0, 0, 0, 0, 0]],
    )


def test_metric_success_assessment_marks_very_low_move_ndtw_as_failure():
    result = metric_success_assessment(
        make_case("Move"),
        {"point_count": 2},
        {
            "status": "ok",
            "ndtw": 0.000001,
            "metric_interpretation": {"mode": "position_and_rotation_sequence"},
        },
    )

    assert result["success"] is False
    assert result["failure_type"] == "low_ndtw"
    assert result["thresholds"]["low_ndtw_failure"] == 0.05


def test_sr_judge_low_ndtw_overrides_llm_success(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "success": True,
                                    "confidence": 0.95,
                                    "reason": "incorrectly says low nDTW confirms alignment",
                                    "failure_type": None,
                                    "needs_human_review": False,
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    result = judge_success(
        make_case("Move"),
        {"point_count": 2},
        {
            "status": "ok",
            "ndtw": 8.6e-11,
            "metric_interpretation": {"mode": "position_and_rotation_sequence"},
        },
        enabled=True,
        model_config=ModelConfig(model_name="fake-vl", api_key="key", base_url="https://example.test/v1"),
    )

    assert result["status"] == "conflict"
    assert result["success"] is False
    assert result["llm_success"] is True
    assert result["failure_type"] == "metric_gate_conflict"
    assert result["needs_human_review"] is True
    assert result["confidence"] <= 0.2


def test_sr_payload_explains_ndtw_direction_and_turn_metric_mode():
    payload = _payload(
        make_case("Turn"),
        {"point_count": 2},
        {
            "status": "ok",
            "ndtw": 0.02,
            "metric_interpretation": {"mode": "rotation_sequence"},
        },
        ModelConfig(model_name="fake-vl", api_key="key"),
    )

    system_prompt = payload["messages"][0]["content"]
    user = json.loads(payload["messages"][1]["content"])
    assert "1 is best and 0 is worst" in system_prompt
    assert "Treat metric_success_assessment as a hard constraint" in system_prompt
    assert user["metric_success_assessment"]["success"] is False
    assert user["metric_success_assessment"]["metric_mode"] == "rotation_sequence"


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")
