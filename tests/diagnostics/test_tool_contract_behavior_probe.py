"""Offline regression for the reusable tool-contract experiment probe."""

import copy
import json

import pytest

from src.agent.transcript import AgentToolCall, AssistantTurn
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from tests.support.environment import tool_contract_behavior_probe as probe


def test_structural_comparison_does_not_hide_constraints():
    tools = probe.tool_snapshot(default_registry())
    changed = copy.deepcopy(tools)
    changed[-1]["schema"]["properties"]["yaw_deg"]["maximum"] = 1
    assert probe.structural_contracts(changed) != probe.structural_contracts(tools)
    changed = copy.deepcopy(tools)
    changed[-1]["schema"]["properties"]["yaw_deg"]["description"] = "Reworded."
    assert probe.structural_contracts(changed) == probe.structural_contracts(tools)
    changed = copy.deepcopy(tools)
    plan = next(tool for tool in changed if tool["name"] == "runtime.update_plan")
    plan["schema"]["properties"]["steps"]["items"]["properties"]["description"]["minLength"] = 20
    assert probe.structural_contracts(changed) != probe.structural_contracts(tools)
    extension = {"x-flyagent-policy": {"description": "Semantic extension payload"}}
    assert probe.without_descriptions(extension) == extension



@pytest.mark.parametrize("index", [0, 6, 7, 8, 9])
def test_probe_uses_default_memory_fixed_cases_and_no_backend(tmp_path, index):
    cases = probe.corpus()
    assert len(cases) == 10 and probe.digest(cases) == probe.digest(probe.corpus())
    transcript, observation = probe.build_transcript(cases[index], tmp_path, PluginRegistry.from_config())
    envelope = transcript.instruction_envelope()
    checkpoint = transcript.latest_checkpoint()
    assert checkpoint is not None
    assert "Identity And View" in envelope["identity"]["text"]
    assert "Precision And Judgment" in envelope["user_preferences"]["text"]
    assert "Synthetic" in json.dumps(checkpoint.to_dict())
    assert observation["observation_id"] in transcript.prior_evidence_reference_ids()
    if index == 8:
        assert "synthetic_prior_call" in transcript.prior_evidence_reference_ids()
    if index == 7:
        media_ref = checkpoint.fragments[-1].media_refs[0]["media_ref"]
        assert media_ref["ref"].endswith("frame_000001.png")
        assert len(media_ref["sha256"]) == 64



def test_probe_distinguishes_missing_call_extra_call_and_wrong_toggle():
    case = probe.corpus()[7]
    turn = AssistantTurn("t", "Measure", [AgentToolCall("c", case["expected"],
                         {"observation_id": "synthetic_obs_1", "include_distance": False})])
    metrics = probe.selection_metrics(case, turn)
    assert not metrics["omitted_expected_call"] and not metrics["expected_arguments_match"]
    assert not metrics["matched"]
    turn = AssistantTurn("t", "Plan", [AgentToolCall("c", "runtime.update_plan", {"operation": "replace", "explanation": None, })])
    metrics = probe.selection_metrics(case, turn)
    assert metrics["omitted_expected_call"]
    assert metrics["unnecessary_calls"] == ["runtime.update_plan"]

