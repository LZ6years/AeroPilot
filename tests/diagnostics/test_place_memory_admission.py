"""Provider-chain and temporal-audit checks without external model calls."""
import json
from types import SimpleNamespace

import pytest

from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.manual.place_memory_admission import run_place_admission
from tests.manual.aerial_vln_pixel_place_audit import audit_place_timeline
from tests.manual.aerial_vln_plan_context_probe import write
from tests.test_pixel_place_memory import flight


@pytest.mark.parametrize("bad", [False, True])
def test_three_provider_chains_and_first_failure_preserved(tmp_path, bad):
    sandbox = tmp_path / "capability_admission"
    sandbox.mkdir()
    flight(sandbox, [(0, 0, 0), (1, 0, 0)])
    tools = [t.to_provider_tool() for t in default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json")).model_specs()]
    calls = []

    def request(value):
        phase = len(calls) % 6
        calls.append(value)
        name = ["list_places", "save_place", "list_places", "inspect_place", "save_place", "inspect_place"][phase]
        args = {"offset": 0, "limit": 10} if name == "list_places" else {"place_id": "P001"}
        if name == "save_place":
            args.update(summary="Possibly red" if phase == 1 else "Needs confirmation", landmarks=[],
                        evidence_refs=["obs_0001"], photo_observation_ids=["obs_0001"] if phase == 1 else [])
        if bad:
            args["offset"] = -1
        return SimpleNamespace(data={"model": "test"}, text="", usage={}, transport_trace={},
            tool_calls=[{"id": str(len(calls)), "type": "function", "function": {
                "name": "skill_place_memory_" + name, "arguments": json.dumps(args)}}])

    model = SimpleNamespace(model_name="test", generation_parameters=lambda: {})
    if bad:
        with pytest.raises(RuntimeError, match="not started"):
            run_place_admission(tmp_path, SimpleNamespace(request=request), model, tools, "test", write)
        assert len(calls) == 1
        assert json.loads((tmp_path / "place_memory_admission.json").read_text())["status"] == "failed"
    else:
        result = run_place_admission(tmp_path, SimpleNamespace(request=request), model, tools, "test", write)
        assert result["status"] == "passed" and len(calls) == 18
        assert result["groups_completed"] == 3


def test_directory_audit_uses_prior_saves_not_future_graph():
    events = []

    def append(identity, name, args, data):
        tool = "skill.place_memory." + name
        events.extend([{"kind": "assistant_turn", "payload": {"tool_calls": [
            {"tool_call_id": identity, "tool_name": tool, "args": args}]}},
            {"kind": "tool_result", "payload": {"tool_call_id": identity, "tool_name": tool, "status": "ok", "data": data}}])

    args = {"offset": 0, "limit": 10}
    append("before", "list_places", args, {"places": [], "total": 0})
    append("save", "save_place", {"place_id": "P001", "summary": "Uncertain", "landmarks": [], "evidence_refs": ["o1"]}, {})
    append("after", "list_places", args, {"places": [{"place_id": "P001", "summary": "Uncertain"}], "total": 1})
    report = audit_place_timeline(events)
    assert not report["failures"] and report["calls"][-1]["repeated_after_save"]
    events[-1]["payload"]["data"]["places"][0]["summary"] = "Future revision"
    assert audit_place_timeline(events)["failures"] == ["directory_differs_from_prior_saves"]
