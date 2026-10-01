"""Description-only migration invariants and bounded offline probe contracts.

The fixture captures the pre-cleanup configured contracts. It deliberately
checks structure and hidden-tool definitions rather than snapshotting new prose.
No real model requests or backend calls are made by these tests.
"""

import copy
import json
from pathlib import Path

import pytest

from src.agent.transcript import AgentToolCall, AssistantTurn
from src.memory.context.modules.image import CURRENT_IMAGE_USAGE, CURRENT_RGB_DEPTH_OVERLAY_USAGE
from src.plugins import PluginRegistry
from src.uav.config import load_uav_safety_config
from src.tools.registry import default_registry
from tests.support.environment import tool_contract_behavior_probe as probe


def test_description_cleanup_preserves_contracts_except_retired_completion_gate():
    baseline = json.loads(Path("tests/fixtures/tool_description_contract_baseline.json").read_text(encoding="utf-8"))
    registry = default_registry(
        PluginRegistry.from_config(),
        safety_config=load_uav_safety_config(),
    )
    current = probe.tool_snapshot(registry)
    # Keep the historical fixture intact. The later grounding migration removes
    # only this task-global policy; schemas, limits and hidden tools stay fixed.
    for tool in baseline:
        if tool["name"] == "skill.visual_grounding.measure_visible_depth":
            tool["schema"].pop("x-flyagent-completion-evidence", None)
            tool["schema"]["properties"].pop("include_distance")
            tool["schema"]["required"].remove("include_distance")
    assert probe.structural_contracts(current) == probe.structural_contracts(baseline)
    assert sum(tool["runtime"]["model_exposed"] for tool in current) == 7
    for before, after in zip(baseline, current):
        if not before["runtime"]["model_exposed"]:
            assert before["description"] == after["description"]
            assert before["schema"] == after["schema"]
        if before["name"] != "runtime.update_plan":
            assert before["input_examples"] == after["input_examples"]
    assert registry.get("runtime.update_plan").input_examples == ()


def test_arguments_explain_units_and_effects_not_selection_order():
    registry = default_registry(PluginRegistry.from_config())
    movement = registry.get("uav.move_body").input_schema["properties"]
    for name in ("vx", "vy", "vz"):
        assert "m/s" in movement[name]["description"]
    measurement = registry.get("skill.visual_grounding.measure_visible_depth").input_schema["properties"]
    assert "CURRENT" in measurement["observation_id"]["description"]
    assert set(measurement) == {"observation_id", "query"}


def test_provider_prose_removes_repeated_global_rules_but_keeps_local_policies():
    registry = default_registry(PluginRegistry.from_config())
    for spec in registry.model_specs():
        text = spec.to_provider_tool()["function"]["description"]
        assert "effect=" in text and "result_timing=" in text
        assert "Do not mix with runtime-control or physical calls" not in text
        assert "Reuse identical current-observation results" not in text
    grounding = registry.get("skill.visual_grounding.measure_visible_depth").to_provider_tool()["function"]["description"]
    assert "At most 2 calls per observation" in grounding
    assert "Call only when the image itself supplies direct evidence" in grounding
    user = Path("src/agent/planning/prompts/tool_user_system.txt").read_text(encoding="utf-8")
    assert "these groups do not mix" in user
    assert "reuse an applicable read result" in user


def test_image_labels_and_active_knowledge_do_not_prescribe_tool_order_or_hover():
    for usage in (CURRENT_IMAGE_USAGE, CURRENT_RGB_DEPTH_OVERLAY_USAGE):
        assert "grounding" not in usage and "request" not in usage
    assert "meters" in CURRENT_RGB_DEPTH_OVERLAY_USAGE
    assert "qualitative" in CURRENT_RGB_DEPTH_OVERLAY_USAGE
    plugins = PluginRegistry.from_config()
    knowledge = [skill for skill in plugins.skill_catalog.skills if skill.capability is None]
    assert len(knowledge) == 5
    for skill in knowledge:
        texts = [skill.body, *(path.read_text(encoding="utf-8") for path in skill.reference_paths)]
        for text in texts:
            assert "hover" not in text.lower()
            for retired in ("Interpret images first", "grounded candidate distance only",
                            "visual grounding with candidate distance enabled", "repeat precise grounding"):
                assert retired not in text
    identity = Path("memory/FLYAGENT.md").read_text(encoding="utf-8")
    assert "hold position" not in identity
    assert not Path("src/agent/planning/prompts/tool_use_case_studies.txt").exists()


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
    turn = AssistantTurn("t", "Plan", [AgentToolCall("c", "runtime.update_plan", {})])
    metrics = probe.selection_metrics(case, turn)
    assert metrics["omitted_expected_call"]
    assert metrics["unnecessary_calls"] == ["runtime.update_plan"]
