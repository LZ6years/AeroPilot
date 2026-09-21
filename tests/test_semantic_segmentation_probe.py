"""Parser-only experiment wiring, not semantic correctness assertions."""
import argparse
import json
from pathlib import Path

import pytest

from tests.manual import aerial_vln_semantic_segmentation_probe as probe
from src.runtime.configs.model_config import ModelConfig


def test_manifest_order_and_no_truth_fields(tmp_path):
    path = tmp_path / "inputs.json"
    rows = [{"id": "b", "instruction": "Move", "protocol": "generic"},
            {"id": "a", "instruction": "Turn", "protocol": "aerial_vln"}]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert probe.load_instruction_list(path) == rows
    rows[0]["reference_path"] = []
    path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        probe.load_instruction_list(path)


def test_recording_parser_loads_snapshots_and_redacts(monkeypatch):
    monkeypatch.setattr(probe.TaskParser, "_default_phase_skill_resolver", lambda self: object())
    config = ModelConfig(api_key="test-secret")
    prompts = {slot: "snapshot " + slot for slot in probe.PROMPT_FILES}
    parser = probe.RecordingTaskParser(config, "Move", protocol_context="", prompts=prompts)
    assert parser.protocol_context == ""
    assert all(getattr(parser, slot) == text for slot, text in prompts.items())
    monkeypatch.setattr(probe.TaskParser, "_request_model_content", lambda *a: "public response")
    assert parser._request_model_content([{"role": "user", "content": "test-secret"}]) == "public response"
    assert "test-secret" not in json.dumps(probe._redacted(parser.exchanges, config.api_key))


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("structure_only", [False, True])
def test_comparison_runs_once_in_order_without_dataset_or_engine(tmp_path, monkeypatch, fail, structure_only):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"id": "one", "instruction": "<Move>", "protocol": "generic"}]))
    before = tmp_path / "before"
    before.mkdir()
    for slot, name in probe.PROMPT_FILES.items():
        text = ((probe.REPO_ROOT / "src/agent/semantic/prompts" / name).read_text(encoding="utf-8")
                if structure_only and slot != "structure_prompt" else "before snapshot")
        (before / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(probe, "load_model_config", lambda _: ModelConfig(api_key="secret", model_name="fake"))
    monkeypatch.setattr(probe, "load_aerial_vln_dataset", lambda *a, **k: pytest.fail("Dataset must not be loaded"))
    calls = []

    class FakeParser:
        def __init__(self, config, instruction, *, prompts, protocol_context):
            calls.append((prompts["structure_prompt"], protocol_context))
            self.exchanges, self.contract_calls, self.last_parse_trace = [], [], {}

        def parse(self, instruction):
            assert not structure_only, "Full parse must not run in structure-only mode"
            if fail:
                raise RuntimeError("provider exhausted secret")
            class Result:
                class plan:
                    @staticmethod
                    def to_dict():
                        return {"phases": [{"name": "move", "goal": instruction, "done_when": ["done"]}]}
            return Result()

        def parse_structure(self, instruction):
            assert structure_only
            if fail:
                raise RuntimeError("provider exhausted secret")
            return probe.SemanticPlanStructure(
                intent="move", task_goal={"target": "<target>"},
                phases=[{"index": 1, "total": 1, "name": "move", "goal": instruction}],
            )

    monkeypatch.setattr(probe, "RecordingTaskParser", FakeParser)
    args = argparse.Namespace(instructions_file=inputs, before_prompts=before,
                              output_dir=str(tmp_path / "output"), env_path="unused",
                              structure_only=structure_only)
    assert probe.run_comparison(args) == (1 if fail else 0)
    assert len(calls) == (1 if fail else 2) and calls[0] == ("before snapshot", "")
    page = (Path(args.output_dir) / "review.html").read_text(encoding="utf-8")
    assert "&lt;Move&gt;" in page and "heuristic" not in page
    assert "secret" not in (Path(args.output_dir) / "summary.json").read_text(encoding="utf-8")
    summary = json.loads((Path(args.output_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == ("structure_only" if structure_only else "full_parse")
    if structure_only:
        assert "Phase structure only" in page
        assert all("plan" not in item for item in summary["results"])
        if not fail:
            assert "&lt;target&gt;" in page
            assert all("done_when" not in item["structure"]["phases"][0] for item in summary["results"])
    snapshots = json.loads((Path(args.output_dir) / "prompts.json").read_text(encoding="utf-8"))
    for version in ("before", "after"):
        for slot in probe.PROMPT_FILES:
            saved = snapshots[version][slot]
            assert saved["sha256"] == probe._sha256_text(saved["text"])
    with pytest.raises(ValueError):
        probe.run_comparison(args)


def test_comparison_path_has_no_scene_runtime_or_ssh_calls():
    """Static guard complements the fake-parser execution test above."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(probe.run_comparison))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"TaskRuntime", "SimulationServer", "EnvClient", "subprocess", "paramiko", "load_aerial_vln_dataset"}


@pytest.mark.parametrize("protocol", ["", probe.AERIAL_VLN_PARSER_PROTOCOL_CONTEXT])
def test_structure_only_uses_real_contract_without_skills_or_criteria(monkeypatch, protocol):
    """Stub only provider text; retain production message and repair machinery."""
    monkeypatch.setattr(probe.TaskParser, "_default_phase_skill_resolver", lambda self: object())
    def forbidden(*args, **kwargs):
        pytest.fail("Structure-only probe entered a later parsing stage")
    for method in ("parse", "_parse_with_model", "_resolve_phase_guidance", "_build_phase_criteria_messages"):
        monkeypatch.setattr(probe.TaskParser, method, forbidden)
    valid = {"intent": "move", "task_goal": {"target": "ahead", "target_attributes": {}, "final_state": "displaced"},
             "phases": [{"index": 1, "total": 1, "name": "move", "goal": "Move forward 4 meters"}]}
    replies = iter(["{}", json.dumps(valid)])
    monkeypatch.setattr(probe.TaskParser, "_request_model_content", lambda *args: next(replies))
    parser = probe.RecordingTaskParser(ModelConfig(api_key="fake"), "Move forward 4 meters", protocol_context=protocol)
    result = parser.parse_structure("Move forward 4 meters")
    assert result.to_dict() == valid
    assert len(parser.exchanges) == 2  # Existing structural repair is not bypassed.
    assert [item["stage"] for item in parser.contract_calls] == ["single_pass_structure"]
    messages = parser.exchanges[0]["messages"]
    assert messages[-2]["content"] == parser.structure_prompt
    assert messages[-1]["content"] == "Move forward 4 meters"
    assert sum(message["content"] == protocol for message in messages) == bool(protocol)
    assert all(message["content"] not in (parser.done_when_rules, parser.criteria_prompt) for message in messages)
    assert parser.last_parse_trace == {}


def test_structure_only_rejects_missing_list_before_external_calls(monkeypatch):
    monkeypatch.setattr(probe, "load_model_config", lambda *args: pytest.fail("No external setup"))
    with pytest.raises(ValueError, match="--structure-only requires"):
        probe.main(["--structure-only"])


def test_structure_only_requires_fixed_criteria_snapshots(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([{"id": "one", "instruction": "Move", "protocol": "generic"}]), encoding="utf-8")
    before = tmp_path / "before"
    before.mkdir()
    for filename in probe.PROMPT_FILES.values():
        (before / filename).write_text("different", encoding="utf-8")
    monkeypatch.setattr(probe, "load_model_config", lambda _: ModelConfig(api_key="fake", model_name="fake"))
    monkeypatch.setattr(probe, "RecordingTaskParser", lambda *a, **k: pytest.fail("Must reject before requests"))
    args = argparse.Namespace(instructions_file=inputs, before_prompts=before,
                              output_dir=str(tmp_path / "output"), env_path="unused", structure_only=True)
    with pytest.raises(ValueError, match="unchanged done_when_rules"):
        probe.run_comparison(args)
    assert not Path(args.output_dir).exists()
