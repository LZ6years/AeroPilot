"""Admission and measurement checks for the strictly offline assembly probe."""
import copy

import pytest

from tests.manual.context_assembly_probe import alternating, distribution, prefix_boundary, diagnostic


def test_unique_boundary_excludes_response_and_future_events():
    turn = {"turn_id": "t1", "tool_calls": []}
    records = [{"instruction_id": "i", "kind": "state_checkpoint", "payload": {}},
               {"instruction_id": "i", "kind": "assistant_turn", "payload": turn},
               {"instruction_id": "i", "kind": "tool_result", "payload": {}}]
    assert prefix_boundary(records, turn, "i") == 1
    with pytest.raises(ValueError, match="ambiguous"):
        prefix_boundary(records + [records[1]], turn, "i")
    with pytest.raises(ValueError, match="contents"):
        prefix_boundary(records, {**turn, "tool_calls": ["changed"]}, "i")
    with pytest.raises(ValueError, match="different instruction"):
        prefix_boundary([{**records[0], "instruction_id": "other"}, *records[1:]], turn, "i")
    with pytest.raises(ValueError, match="identity"):
        prefix_boundary(records, {}, "i")


def test_alternating_keeps_first_separate_and_detects_nested_differences():
    order = []
    def run(name):
        order.append(name)
        return {"messages": [{"image": "exact bytes"}]}
    result = alternating({name: lambda name=name: run(name) for name in ("baseline", "candidate")}, repeats=2)
    assert order == ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]
    assert result["stats"]["baseline"]["n"] == 2
    assert sum(r["first"] for r in result["rows"]) == 2
    with pytest.raises(ValueError, match="differs"):
        alternating({"baseline": lambda: {"image": "a"}, "candidate": lambda: {"image": "b"}}, repeats=1)


def test_percentile_uses_fraction_and_memory_pass_is_separate():
    assert distribution([.001, .002, .003])["p95_ms"] == pytest.approx(2.9)
    calls = []
    def run():
        calls.append(1)
        return copy.deepcopy({"nested": [1, 2]})
    result = diagnostic(run)
    assert len(calls) == 2
    assert result["copy_calls"] == 1
    assert result["python_peak_bytes"] > 0


def test_correctness_only_has_no_warm_timing_claim():
    result = alternating({"baseline": lambda: {"bytes": b"same"},
                          "candidate": lambda: {"bytes": b"same"}}, repeats=0)
    assert result["equal"]
    assert result["stats"]["baseline"] == {"p50_ms": None, "p95_ms": None, "n": 0}
