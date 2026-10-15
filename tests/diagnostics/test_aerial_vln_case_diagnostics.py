"""Public intent provenance and complete multi-round diagnostic rendering."""
import json

from tests.manual.aerial_vln_case_diagnostics import captured_turns, render_turns


def test_all_rounds_and_receipts_are_escaped_and_keep_provenance(tmp_path):
    path = tmp_path / "debug/tool_loop/task/decision_01.json"
    path.parent.mkdir(parents=True)
    calls = [{"tool_call_id": key, "tool_name": "inspect", "args": {"place_id": key}} for key in ("a", "b")]
    path.write_text(json.dumps({"decision_index": 1, "checkpoint_observation_id": "obs1", "model_rounds": [
        {"round_index": 0, "response": {"message": {"content": "<script>correct me</script>"}}},
        {"round_index": 1, "canonical_turn": {"public_intent": "完整意图" * 80,
            "public_intent_source": "provider_content", "tool_calls": calls}},
        {"round_index": 2, "canonical_turn": {"public_intent": "inspect a", "public_intent_source": "tool_arguments_fallback"}},
        {"round_index": 3}]}), encoding="utf-8")
    transcript = tmp_path / "memory/agent_transcript.jsonl"
    transcript.parent.mkdir()
    transcript.write_text(json.dumps({"kind": "tool_result", "payload": {"tool_call_id": "b", "status": "ok"}}), encoding="utf-8")
    turns = captured_turns(tmp_path)
    assert len(turns) == 4
    assert turns[1]["calls"][0]["results"] == []
    assert turns[1]["calls"][1]["results"][0]["status"] == "ok"
    page = render_turns(turns, tmp_path, lambda root, path: 'source"&.json')
    assert "完整意图" * 80 in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "参数生成的动作说明（非模型原文）" in page
    assert "未捕获响应" in page and 'source&quot;&amp;.json' in page


def test_missing_capture_does_not_invent_intent(tmp_path):
    assert captured_turns(tmp_path) == []
