"""Caller protocol and intact instruction contracts for native-action v3."""

from __future__ import annotations

from evaluation.benchmarks.aerial_vln.protocol import (
    AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
    AERIAL_VLN_PROTOCOL,
)
import pytest


def test_aerial_vln_v3_requires_the_whole_route_without_landing_special_cases():
    context = AERIAL_VLN_PARSER_PROTOCOL_CONTEXT

    assert AERIAL_VLN_PROTOCOL == "aerial_vln.uav_agent_native_actions.v3"
    assert "fulfilling the whole instructed route and its final position relation" in context
    assert "intermediate arrival or stop does not complete the episode" in context
    assert "explicit subsequent requirements remain" in context
    assert "parsed task_goal" not in context
    assert "phase name/goal" not in context
    assert "Seeing a landmark is different from reaching or passing it" in context
    assert "historical passage requirements" in context
    for removed in ('"land on X"', '"land in X"', '"land by X"',
                    "ground contact", "AirSim landed state", "uav.land",
                    "landing-readiness", "Do not infer descent", "episode-level landing"):
        assert removed not in context
    assert "separately requested ascent, descent, approach and intermediate checkpoints" in context


@pytest.mark.parametrize("instruction", [
    "Fly to the bend and land on the curve. Pass four chimneys and fly down. Turn right.",
    "Land on the street. Approach the theatre. Stop and land. Turn toward the statue and return.",
    "Rise above the roofs, descend facing the building, and land beside the entrance.",
    "Follow the road, turn left at the truck, and stop.",
])
def test_adapter_preserves_middle_stops_final_stops_and_vertical_requirements(instruction):
    """No substring-based truncation or semantic rewrite enters Runtime input.

    This checks the wire boundary, not the model's interpretation. Saved-real-
    context decisions and actual navigation cover that separate behavior.
    """
    from evaluation.benchmarks.aerial_vln.adapter import AerialVlnCaseAdapter
    from evaluation.benchmarks.aerial_vln.cases import AerialVlnCase
    from src.env_client import ObservationProfile

    case = AerialVlnCase("episode", "trajectory", "1", instruction, (0, 0, -10),
                        (1, 0, 0, 0), (10, 10, -10), ((0, 0, -10), (10, 10, -10)), (1, 0), {})
    adapter = AerialVlnCaseAdapter(scene_map={"1": "scene"},
                                  observation_profile=ObservationProfile())
    prepared = adapter.prepare_case(case)
    assert prepared.instruction == instruction
    assert prepared.protocol_context == AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
    assert prepared.protocol_variant == AERIAL_VLN_PROTOCOL


def test_aerial_vln_protocol_is_caller_owned_not_default_mission_state():
    from src.agent.mission_state import MissionState
    state = MissionState("test", "Land by the red building")
    assert state.protocol_context == ""
    configured = MissionState("test", state.user_instruction,
                              protocol_context=AERIAL_VLN_PARSER_PROTOCOL_CONTEXT)
    assert configured.protocol_context == AERIAL_VLN_PARSER_PROTOCOL_CONTEXT


def test_arrival_aim_is_aerial_vln_guidance_without_changing_scoring_or_generic_knowledge():
    from pathlib import Path
    from evaluation.benchmarks.aerial_vln.metrics import SUCCESS_DISTANCE
    from src.plugins import PluginRegistry

    context = AERIAL_VLN_PARSER_PROTOCOL_CONTEXT
    assert SUCCESS_DISTANCE == 20.0
    assert "at most 20 metres" in context and "within roughly 10 metres" in context
    assert "outer tolerance" in context
    assert "unavailable, invalid or unsuitable for horizontal separation" in context
    assert "do not invent a claim" in context
    assert "does not override the instructed route" in context
    assert "final position relations" in context
    # Caller-specific guidance must not leak through reusable skills or standing
    # prompts; changing this protocol cannot affect a non-AerialVLN mission.
    catalog = PluginRegistry.discover({"plugins": {
        "roots": ["plugins/knowledge"], "enabled": ["knowledge.uav_flight"]}}).skill_catalog
    for skill in catalog.skills:
        for text in (skill.body, skill.description):
            assert "AerialVLN" not in text and "10 metres" not in text and "20 metres" not in text
    prompt = Path("src/agent/planning/prompts/agent_tool_loop_system.txt").read_text(encoding="utf-8")
    assert "AerialVLN" not in prompt and "10 metres" not in prompt and "20 metres" not in prompt
