"""TravelUAV protocol identity and semantic-boundary tests."""

from evaluation.benchmarks.travel_uav.protocol import (
    TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT,
    TRAVEL_UAV_PROTOCOL,
)


def test_protocol_is_explicitly_front_view_no_assistant_transfer():
    assert TRAVEL_UAV_PROTOCOL == "travel_uav.flyagent_native_actions.front_view.no_assistant.v3"
    context = TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT.lower()
    assert "visually confirmed" in context
    assert "physical ground contact" in context
    assert "no more than 20 m" in context
    assert "accurate target bearing" in context
    assert "measured from the uav heading at instruction start" in context
    assert "world-referenced ray from the instruction-start pose" in context
    assert "does not provide target distance" in context
    assert "prescribe that the uav continuously move" in context
    assert "same-category detector match" in context
    assert "reference path" not in context
    assert "target coordinate" not in context
    assert "neither target distance nor guaranteed immediate visibility" in context
    assert "not automatically an ordered route" in context
    for action_policy in (
        "preserve that world-relative search corridor",
        "do not accumulate reverse turns",
        "deviate only",
        "stop unsupported large back-and-forth rotations",
    ):
        assert action_policy not in context
