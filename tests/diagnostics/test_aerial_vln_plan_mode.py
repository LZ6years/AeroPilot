"""Frozen experiment exposure, no-flight admission and nullable-state checks."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.planning.tool_contract_validator import NativeTurnValidationError, validate_native_turn
from src.agent.transcript import AgentToolCall, AssistantTurn
from src.runtime.task.turn_executor import TurnExecutor
from src.tools.registry import default_registry
from tests.manual.aerial_vln_plan_context_admission import run_no_plan_admission
from tests.manual.aerial_vln_plan_context_acceptance import request_checks
from tests.manual.aerial_vln_plan_context_probe import digest, freeze_source, token_summary, write
from tests.manual.aerial_vln_plan_mode import admission_passed, freeze_plan_mode


@pytest.fixture
def snapshot(tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    relative = ["src/tools/builtins/planning.py", *[
        "src/agent/planning/prompts/" + name for name in (
            "agent_tool_loop_system.txt", "task_progress_reminder.txt", "tool_user_system.txt")]]
    hashes = {}
    for name in relative:
        target = tmp_path / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((workspace / name).read_bytes())
        hashes[name] = digest(target)
    return tmp_path, hashes


def off_registry(root):
    module_spec = importlib.util.spec_from_file_location("frozen_plan_definition", root / "source/src/tools/builtins/planning.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    registry = default_registry()
    registry.register(module.planning_tool())
    return registry


def test_off_overlay_retains_storage_contract_and_excludes_provider_and_execution(snapshot):
    root, hashes = snapshot
    original = dict(hashes)
    receipts = freeze_plan_mode(root, hashes, "off", digest)
    assert len(receipts) == 4
    assert all(hashes[p] != original[p] for p in hashes)
    assert all(r["after_sha256"] == hashes[r["source"]] for r in receipts)
    registry = off_registry(root)
    plan = registry.get("runtime.update_plan")
    assert not plan.runtime.model_exposed and plan.strict is True
    assert plan.input_schema == default_registry().get(plan.name).input_schema
    allowed = TurnExecutor.allowed_tools(SimpleNamespace(capabilities=None), registry)
    assert "runtime.update_plan" not in allowed
    assert {"uav.move_body", "uav.turn_relative", "knowledge.load"} <= allowed
    assert "runtime_update_plan" not in {t.wire_name for t in registry.model_specs()}
    turn = AssistantTurn("attempt", "", [AgentToolCall("call_off", plan.name, {
        "operation": "replace", "explanation": None, "steps": []})])
    with pytest.raises(NativeTurnValidationError) as rejected:
        validate_native_turn(turn, registry, allowed_names=allowed, latest_observation=None)
    assert rejected.value.diagnostic["execution"] == "not_started"


def test_optional_overlay_does_not_mutate_snapshot(snapshot):
    root, hashes = snapshot
    before = dict(hashes)
    assert freeze_plan_mode(root, hashes, "optional", digest) == []
    assert hashes == before


def test_overlay_prevalidation_avoids_partial_writes(snapshot):
    root, hashes = snapshot
    last = root / "source/src/agent/planning/prompts/tool_user_system.txt"
    last.write_text("unrecognized prompt", encoding="utf-8")
    before = {name: digest(root / "source" / name) for name in hashes}
    with pytest.raises(ValueError, match="expected one match"):
        freeze_plan_mode(root, hashes, "off", digest)
    assert before == {name: digest(root / "source" / name) for name in hashes}


def test_off_request_requires_missing_tool_empty_plan_and_clean_prompts(snapshot):
    root, hashes = snapshot
    freeze_plan_mode(root, hashes, "off", digest)
    registry = off_registry(root)
    state = {"task_plan": None, "loaded_knowledge": []}
    prompt = root / "source/src/agent/planning/prompts"
    turn = {"current_task_state": state, "request": {
        "tools": [t.to_provider_tool() for t in registry.model_specs()], "messages": [
            *[{"role": "system", "content": (prompt / name).read_text(encoding="utf-8")}
              for name in ("agent_tool_loop_system.txt", "task_progress_reminder.txt", "tool_user_system.txt")],
            {"role": "system", "content": "CURRENT TASK STATE\n" + json.dumps(state)}]}}
    assert request_checks(turn, plan_mode="off")[0] == []
    turn["request"]["tools"].append(default_registry().get("runtime.update_plan").to_provider_tool())
    assert "off_mode_plan_tool_exposed" in request_checks(turn, plan_mode="off")[0]
    state["task_plan"] = {"revision": 1}
    assert "off_mode_saved_plan_not_empty" in request_checks(turn, plan_mode="off")[0]


class NativeClient:
    """A recording client proves admission validates without any dispatcher."""
    def __init__(self, bad=False):
        self.requests = []
        self.bad = bad

    def request(self, request):
        self.requests.append(request)
        name, args = [
            ("uav_turn_relative", {"yaw_deg": 999 if self.bad else 10}),
            ("uav_move_body", {"vx": 1, "vy": 0, "vz": 0, "duration": 1}),
            ("knowledge_load", {"name": "basic_flight_control"})][len(self.requests) - 1]
        return SimpleNamespace(tool_calls=[{"id": f"call_{len(self.requests)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}], data={"model": "test"},
            usage={"input_tokens": 10, "output_tokens": 4}, transport_trace={})


def test_off_admission_three_first_attempt_native_calls_and_no_plan_lifecycle(snapshot):
    root, hashes = snapshot
    freeze_plan_mode(root, hashes, "off", digest)
    tools = [t.to_provider_tool() for t in off_registry(root).model_specs()]
    client = NativeClient()
    model = SimpleNamespace(model_name="test", generation_parameters=lambda: {}, redacted_summary=lambda: {})
    result = run_no_plan_admission(root, client, model, tools, "data:image/png;base64,saved", "saved.png", "hash", write)
    assert admission_passed(result, "off")
    assert result["plan_probe_status"] == "not_applicable"
    assert len(client.requests) == 3
    assert all(r["execution"] == "not_started" for r in result["receipts"])
    assert not admission_passed(result, "optional")


def test_off_admission_failure_stops_without_repair(snapshot):
    root, hashes = snapshot
    freeze_plan_mode(root, hashes, "off", digest)
    tools = [t.to_provider_tool() for t in off_registry(root).model_specs()]
    client = NativeClient(bad=True)
    model = SimpleNamespace(model_name="test", generation_parameters=lambda: {}, redacted_summary=lambda: {})
    with pytest.raises(RuntimeError, match="navigation was not started"):
        run_no_plan_admission(root, client, model, tools, "image", "saved.png", "hash", write)
    assert len(client.requests) == 1
    result = json.loads((root / "admission.json").read_text(encoding="utf-8"))
    assert not admission_passed(result, "off")
    assert result["receipts"][0]["diagnostic"]["execution"] == "not_started"


def test_actual_output_usage_including_zero_and_missing_is_independent_of_input():
    summary = token_summary([None, {"completion_tokens": 0}, {"input_tokens": 8, "output_tokens": 4}])
    assert summary["requests_with_usage"] == 1
    assert summary["requests_with_output_usage"] == 2
    assert summary["output_tokens_sum"] == 4 and summary["output_tokens_mean"] == 2
    assert token_summary([{}])["output_tokens_sum"] is None


def test_isolated_process_uses_frozen_plan_definition(tmp_path):
    workspace = Path(__file__).resolve().parents[2]
    hashes = freeze_source(workspace, tmp_path / "source")
    freeze_plan_mode(tmp_path, hashes, "off", digest)
    code = ("import json; from src.tools.registry import default_registry; "
            "import src.tools.builtins.planning as definition; "
            "r=default_registry(); print(json.dumps({'definition':definition.__file__,"
            "'tools':[s.wire_name for s in r.model_specs()]}))")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path / "source",
                            check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    assert Path(value["definition"]).is_relative_to(tmp_path / "source")
    assert "runtime_update_plan" not in value["tools"]
    assert "uav_move_body" in value["tools"] and "knowledge_load" in value["tools"]
