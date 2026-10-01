"""Teaching demo regressions: no network by default and no hidden history mutation."""
import json

import pytest

from demos import llm_context_demo as demo
from src.agent.transcript import AgentTranscript
from src.llm import ProviderRequestError, ProviderSettings
from src.memory.stores import MemoryEventWriter
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.agent.config import ModelConfig


def test_mock_two_rounds_persist_and_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "load_model_config", lambda *a: pytest.fail("mock must not read .env"))
    monkeypatch.setattr("src.llm.transport.urllib.request.urlopen",
                        lambda *a, **k: pytest.fail("mock must not use HTTP"))
    output = tmp_path / "demo"
    assert demo.main(["--output-dir", str(output)]) == 0
    first = json.loads((output / "round_01_request.json").read_text(encoding="utf-8"))
    second = json.loads((output / "round_02_request.json").read_text(encoding="utf-8"))
    assert first["mock"] and second["mock"]
    assert [m["role"] for m in first["request"]["messages"]] == ["system", "user", "user"]
    assert [m["role"] for m in second["request"]["messages"]] == [
        "system", "user", "user", "assistant", "user"]
    assert first["request"]["messages"][0] == second["request"]["messages"][0]
    assert "临时消息" not in json.dumps(second["request"], ensure_ascii=False)
    records = [json.loads(line) for line in
               (output / "memory/agent_transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in records] == [
        "instruction_envelope", "user_instruction", "state_checkpoint", "assistant_turn",
        "state_checkpoint", "assistant_turn"]
    assert all(record["schema_version"] == "agent_transcript_v3" for record in records)
    answer = records[3]["payload"]["public_intent"]
    assert second["request"]["messages"][3]["content"] == answer
    original = (output / "final_history.json").read_bytes()
    assert demo.main(["--output-dir", str(output)]) == 1
    assert (output / "final_history.json").read_bytes() == original


def test_builder_does_not_mutate_transcript(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        "test",
        "instruction",
        InstructionEnvelope("instruction", "", {}, {}, (), ()),
    )
    fragment = ContextFragment("demo", {"note": "test"})
    transcript.append_state(
        StateCheckpoint(
            "checkpoint_0001",
            "instruction_start",
            "obs_1",
            (fragment,),
            checkpoint_fingerprint("obs_1", (fragment,)),
        )
    )
    before = transcript.to_dict()
    request, audit = demo.build_request(transcript)
    request.messages[1]["content"] = "changed request copy"
    assert transcript.to_dict() == before
    assert audit["builder_example"]["removed_before_sending"]


def test_failure_keeps_records_stops_and_does_not_log_exception_text(tmp_path):
    class FailureClient:
        calls = 0

        def request(self, request, *, error_prefix="Model request failed", transport_trace=None):
            self.calls += 1
            transport_trace.update(final_status="failed", attempt_count=2)
            raise ProviderRequestError("secret echoed by server", category="timeout",
                                       attempts=2, retry_exhausted=True)

    client = FailureClient()
    output = tmp_path / "failed"
    with pytest.raises(ProviderRequestError):
        demo.run_demo(output, client=client, settings=ProviderSettings(), mode="mock")
    assert client.calls == 1
    assert (output / "round_01_request.json").exists()
    assert not (output / "round_02_request.json").exists()
    failure = (output / "round_01_failure.json").read_text(encoding="utf-8")
    assert "secret" not in failure
    assert json.loads(failure)["transport"]["attempt_count"] == 2


def test_real_mode_requires_credentials_before_creating_run(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "load_model_config", lambda *a: ModelConfig())
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as failure:
        demo.main(["--real", "--output-dir", str(output)])
    assert failure.value.code == 2
    assert not output.exists()


def test_response_audit_excludes_private_reasoning(tmp_path):
    from src.llm import ModelResponse

    class TextClient:
        def request(self, request, **kwargs):
            return ModelResponse({"choices": [{"message": {
                "content": "公开回答", "reasoning_content": "private reasoning"}}]})

    output = tmp_path / "public"
    demo.run_demo(output, client=TextClient(), settings=ProviderSettings(), mode="mock")
    for path in output.rglob("*.json*"):
        assert "private reasoning" not in path.read_text(encoding="utf-8")
