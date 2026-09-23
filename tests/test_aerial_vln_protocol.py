"""Contract tests for AerialVLN native-action v2 termination semantics."""

from __future__ import annotations

from evaluation.benchmarks.aerial_vln.protocol import (
    AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
    AERIAL_VLN_PROTOCOL,
)
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
    assert "Do not carry land, landing, landed, or touchdown wording from that terminal phrase" in context
    assert "Preserve separately requested ascent, descent, approach" in context
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


def test_aerial_vln_protocol_is_caller_owned_not_default_mission_state():
    from src.agent.mission_state import MissionState
    state = MissionState("test", "Land by the red building")
    assert state.protocol_context == ""
    configured = MissionState("test", state.user_instruction,
                              protocol_context=AERIAL_VLN_PARSER_PROTOCOL_CONTEXT)
    assert configured.protocol_context == AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
