"""Lock caller-owned message order and generation options across communication migration."""
from __future__ import annotations

import json

import pytest

from src.llm import ModelResponse
from src.llm.providers.openai_compatible import encode_request
from src.agent.semantic.task_parser import TaskParser, _configured_evidence_message
from src.agent.semantic.parser_contracts import SemanticPlanStructure
from src.agent.semantic.plan_updater import SemanticPlanUpdater
from src.plugins.skills.selection import ModelSkillSelector
from src.runtime.configs.model_config import ModelConfig


class CaptureClient:
    """Never performs I/O; freezes the exact outgoing wire dictionary."""
    def __init__(self, config):
        self.config = config
        self.payloads = []

    def request(self, request, *, error_prefix, transport_trace=None):
        self.payloads.append(encode_request(self.config.to_provider_settings(), request))
        return ModelResponse({"choices": [{"message": {"content": "{}"}}]})


def system(text):
    return {"role": "system", "content": text}


@pytest.mark.parametrize("with_optional", [False, True])
def test_parser_two_stage_wire_equivalence(with_optional):
    config = ModelConfig(model_name="test", api_key="fake")
    client = CaptureClient(config)
    parser = TaskParser(mode="model", model_config=config, client=client,
                        phase_skill_resolver=object(),
                        protocol_context="protocol" if with_optional else "")
    parser.agent_rules, parser.user_preferences = "identity", "preferences"
    parser.structure_prompt, parser.criteria_prompt = "structure", "criteria"
    parser.done_when_rules = "done rules"
    parser.configured_evidence = ["RGB"] if with_optional else []
    instruction = "Go to the destination."
    structure = SemanticPlanStructure("navigate", {"target": "destination"}, [])
    phase = {"index": 1, "name": "travel", "goal": instruction}
    guidance = [{"name": "guidance", "content": "knowledge"}] if with_optional else []

    structure_messages = parser._build_structure_messages(instruction)
    criteria_messages = parser._build_phase_criteria_messages(instruction, structure, phase, guidance)
    parser._request_model_content(structure_messages)
    parser._request_model_content(criteria_messages)

    prefix = [system("identity"), system("preferences")]
    expected_structure = prefix + ([system("protocol")] if with_optional else []) + [
        system("structure"), {"role": "user", "content": instruction}]
    expected_criteria = list(prefix)
    if with_optional:
        expected_criteria += [system(
            "Selected knowledge for this immutable semantic phase follows. "
            "Treat it as domain guidance, not observation, tools, or output schema.\n\n"
            + json.dumps(guidance, ensure_ascii=False)),
            system(_configured_evidence_message(["RGB"])), system("protocol")]
    expected_criteria += [system("done rules"), system("criteria"),
        {"role": "user", "content": json.dumps({
            "instruction": instruction, "task_goal": structure.task_goal, "phase": phase},
            ensure_ascii=False)}]
    for payload, messages in zip(client.payloads, [expected_structure, expected_criteria]):
        assert payload == {"model": "test", "messages": messages,
                           "temperature": 0, "response_format": {"type": "json_object"}}


def test_updater_and_selector_do_not_inherit_other_callers_context():
    config = ModelConfig(model_name="test", api_key="fake")
    client = CaptureClient(config)
    updater = SemanticPlanUpdater(config, client=client)
    updater.agent_rules, updater.user_preferences = "identity", "preferences"
    updater.done_when_rules, updater.system_prompt = "done rules", "updater"
    context = {"user_instruction": "Move", "selected_skill_guidance": [],
               "configured_evidence": []}
    assert updater._call_openai_compatible(context) == {}
    assert client.payloads[-1] == {
        "model": "test", "temperature": 0, "response_format": {"type": "json_object"},
        "messages": [system("identity"), system("preferences"), system("done rules"),
                     system("updater"),
                     {"role": "user", "content": '{"user_instruction": "Move"}'}]}
    selector = ModelSkillSelector(config, system_prompt="selector", client=client)
    selector._request_content({"available_skills": []})
    assert client.payloads[-1] == {
        "model": "test", "temperature": 0, "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "messages": [system("selector"),
                     {"role": "user", "content": '{"available_skills": []}'}]}


def test_planner_injected_client_matches_legacy_request_hook(tmp_path):
    from tests.test_model_action_planner import StubPlanner, make_transcript, response_with_calls
    from src.agent.planning.model_action_planner import ModelActionPlanner
    from src.tools.registry import default_registry

    response = response_with_calls(("call_1", "uav_hover", {}), content=None)
    legacy_hook = StubPlanner(response)
    config = legacy_hook.model_config
    client = CaptureClient(config)
    captured_request = client.request

    def request(model_request, **kwargs):
        captured_request(model_request, **kwargs)
        return ModelResponse(response)

    client.request = request
    transcript = make_transcript(tmp_path)
    before = transcript.snapshot()
    registry = default_registry()
    old_turn = legacy_hook.plan_turn(transcript, registry)
    new = ModelActionPlanner(config, client=client)
    new_turn = new.plan_turn(transcript, registry)
    assert client.payloads == legacy_hook.payloads
    assert new_turn.tool_calls == old_turn.tool_calls
    assert new_turn.public_intent_source == "tool_arguments_fallback"
    assert transcript.snapshot() == before
    assert new.last_exchange["request"] == legacy_hook.last_exchange["request"]
