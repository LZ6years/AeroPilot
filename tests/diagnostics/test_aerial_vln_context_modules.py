"""Request-module audit sizes describe actual fields without token attribution."""
import json

from tests.manual.aerial_vln_context_module_audit import module_counts, preview


def test_compressed_history_can_omit_but_never_rewrite_or_reorder_calls():
    from tests.manual.aerial_vln_context_module_audit import historical_calls_match
    original = [{"id": "b", "args": {"x": 1}}, {"id": "c"}]
    full = [{"id": "a"}, *original]
    assert historical_calls_match(original, full, compressed=True)
    assert not historical_calls_match(original, full, compressed=False)
    assert not historical_calls_match(original[::-1], full, compressed=True)
    assert not historical_calls_match([{"id": "b", "args": {"x": 2}}], full, compressed=True)


def test_failed_provider_request_is_counted_without_inventing_projection(tmp_path, monkeypatch):
    from tests.manual import aerial_vln_context_module_audit as audit
    from tests.manual import aerial_vln_plan_context_acceptance as acceptance
    from tests.manual.aerial_vln_plan_context_probe import write, digest, read
    old, output, run = tmp_path / "old", tmp_path / "new", tmp_path / "run"
    write(old / "progress.json", {"cases": [{"task": {"episode_id": "one"},
        "attempts": [{"result": {"run_dir": str(run)}}]}]})
    transcript = run / "memory/agent_transcript.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('', encoding="utf-8")
    source = run / "debug/tool_loop/one/decision_01.json"
    messages = [{"role": "user", "content": "original input"}]
    write(source, {"decision_id": "one", "model_rounds": [{"response": None,
        "request": {"messages": messages}}]})
    original_hash = digest(source)
    monkeypatch.setattr(audit, "offline_audit_registry", lambda *args: object())
    checked = []
    monkeypatch.setattr(acceptance, "request_checks", lambda item: (checked.append(item) or [], 0))
    result = audit.replay_history(output, old, write, digest, expected_requests=1)
    assert result["passed"] and result["requests"] == 1 and len(checked) == 1
    entry = read(result["records"][0]["preview"])
    assert entry["boundary"] is None and entry["candidate_messages"] == []
    assert entry["original_messages"] == messages and digest(source) == original_hash


def test_actual_fields_split_static_dynamic_and_tool_feedback_without_images():
    messages = [
        {"role": "user", "content": "INSTRUCTION CONTEXT\n"+json.dumps({
            "static_module_guidance": [{"name": "motion", "summary": "Coordinates"}], "protocol_context": "guide"})},
        {"role": "user", "content": [{"type": "text", "text": json.dumps({
            "fragments": [{"name": "motion", "content": {"distance": 2}}, {"name": "pose", "content": {"yaw": 3}}]})},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
        {"role": "tool", "content": '{"status":"ok"}'},
    ]
    counts = module_counts(messages)
    assert all(counts[key] > 0 for key in ("static.motion", "dynamic.motion", "dynamic.pose", "tool_result"))
    assert not any("image" in key for key in counts)
    projected = preview(messages)
    assert projected[1]["content"][1]["image_url"]["data_url_sha256"]
    assert messages[1]["content"][1]["image_url"]["url"].startswith("data:image")


def test_history_report_uses_recorded_request_count_without_a_fixed_baseline(tmp_path):
    from tests.manual.aerial_vln_context_module_audit import write_module_report
    write_module_report(tmp_path, [], {"passed": True, "requests": 216, "records": []},
                        lambda *_: None)
    text = (tmp_path / "context_modules.html").read_text(encoding="utf-8")
    assert "216 份历史请求" in text and "220 份历史请求" not in text
    write_module_report(tmp_path, [], None, lambda *_: None)
    text = (tmp_path / "context_modules.html").read_text(encoding="utf-8")
    assert "未知数量" in text and "未运行" in text
