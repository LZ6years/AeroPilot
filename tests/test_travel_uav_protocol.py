"""TravelUAV protocol identity and semantic-boundary tests."""

from evaluation.benchmarks.travel_uav.protocol import (
    TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT,
    TRAVEL_UAV_PROTOCOL,
)


def test_protocol_is_explicitly_front_view_no_assistant_transfer():
    assert TRAVEL_UAV_PROTOCOL == "travel_uav.flyagent_native_actions.front_view.no_assistant.v1"
    context = TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT.lower()
    assert "visually confirmed" in context
    assert "physical ground contact" in context
    assert "20" not in context
    assert "reference path" not in context
    assert "target coordinate" not in context
