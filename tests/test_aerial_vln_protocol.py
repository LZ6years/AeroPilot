"""Contract tests for AerialVLN native-action v2 termination semantics."""

from __future__ import annotations

from evaluation.benchmarks.aerial_vln.protocol import (
    AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
    AERIAL_VLN_PROTOCOL,
)
from src.agent.semantic.task_parser import TaskParser
from src.runtime.configs.model_config import ModelConfig


def test_aerial_vln_v2_protocol_defines_episode_stop_without_physical_land():
    context = AERIAL_VLN_PARSER_PROTOCOL_CONTEXT

    assert AERIAL_VLN_PROTOCOL == "aerial_vln.uav_agent_native_actions.v2"
    assert '"land on X"' in context
    assert '"land in X"' in context
    assert '"land by X"' in context
    assert "reaching the target region" in context
    assert "ending the episode" in context
    assert "In parsed task_goal and phase name/goal text" in context
    assert "Do not carry land, landing, landed, touchdown, or descent wording" in context
    assert "original user instruction remains unchanged" in context
    assert "position relation requested by the instruction" in context
    assert "does not require physical ground contact" in context
    assert "AirSim landed state" in context
    assert "calling uav.land" in context
    assert "In completion conditions" in context
    assert "Do not require or claim a landed, landing, touchdown" in context
    assert "landing-readiness, or suitable-for-landing state" in context
    assert "Do not infer descent, low altitude, surface proximity" in context
    assert "only when the instruction requests it separately" in context


def test_aerial_vln_protocol_context_is_absent_from_default_parser_messages():
    parser = TaskParser(
        mode="model",
        model_config=ModelConfig(
            task_parser_mode="model",
            model_name="test-model",
            api_key="test-key",
        ),
    )

    messages = parser._build_structure_messages("Land by the red building")

    assert AERIAL_VLN_PARSER_PROTOCOL_CONTEXT not in [
        message["content"] for message in messages
    ]
