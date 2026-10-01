"""TravelUAV protocol identity and semantic-boundary tests."""

from evaluation.benchmarks.travel_uav.protocol import (
    TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT,
    TRAVEL_UAV_PROTOCOL,
)


def test_protocol_is_explicitly_front_view_no_assistant_transfer():
    assert TRAVEL_UAV_PROTOCOL == "travel_uav.flyagent_native_actions.front_view.no_assistant.v2"
    context = TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT.lower()
    assert "visually confirmed" in context
    assert "physical ground contact" in context
    assert "approximately 20 m" in context
    assert "stop unsupported large back-and-forth rotations" in context
    assert "strong, persistent search-bearing prior" in context
    assert "persistent search-bearing prior" in context
    assert "do not accumulate reverse turns that cancel the initial yaw" in context
    assert "same-category detector match" in context
    assert "bearing conflicts with the initial direction" in context
    assert "reference path" not in context
    assert "target coordinate" not in context
    assert "anchored to the uav orientation at instruction start" in context
    assert "neither target distance nor guaranteed immediate visibility" in context
    assert "not automatically an ordered route" in context
