"""Acceptance catches delivered-context regressions without live services."""
import json

from tests.manual.aerial_vln_plan_context_acceptance import accept, request_checks, report_link_checks
from src.tools.builtins.planning import planning_tool


def test_delivered_request_checks_state_time_roles_and_argument_duplication():
    state = {"task_plan": None, "loaded_knowledge": []}
    request = {"current_task_state": state, "request": {"tools": [planning_tool().to_provider_tool()],
        "messages": [{"role": "system", "content": "CURRENT TASK STATE\n"+json.dumps(state)},
            {"role": "user", "content": [{"type": "text", "text": "STATE CHECKPOINT CURRENT observation_id=obs"}]},
            {"role": "assistant", "tool_calls": [{"id": "read"}]},
            {"role": "tool", "tool_call_id": "read", "content": '{"status":"ok"}'}]}}
    assert request_checks(request) == ([], 0)
    request["request"]["messages"][-1]["content"] = '{"status":"ok","requested_args":{}}'
    assert "duplicate_requested_args" in request_checks(request)[0]
    request["request"]["messages"].append(request["request"]["messages"][0])
    assert "latest_task_state_not_unique" in request_checks(request)[0]


def test_report_links_preserve_encoded_paths_and_find_missing_evidence(tmp_path):
    evidence = tmp_path / "pixel marker.png"
    evidence.write_bytes(b"captured image")
    (tmp_path / "comparison.html").write_text(
        '<a href="pixel%20marker.png#sample">pixel</a><a href="#case1">case</a>'
        '<a href="https://example.test/report">external</a>', encoding="utf-8")
    (tmp_path / "context_modules.html").write_text('<a href="' + evidence.as_uri() + '">original</a>', encoding="utf-8")
    (tmp_path / "case_diagnostics_01.html").write_text('<a href="missing.png">missing</a>', encoding="utf-8")
    result = report_link_checks(tmp_path)
    assert result["checked"] == 3
    assert len(result["failures"]) == 1
    assert result["failures"][0]["target"] == "missing.png"


def test_terminal_model_error_without_physical_windows_keeps_unknown_metric(tmp_path, monkeypatch):
    from tests.manual.aerial_vln_plan_context_probe import write
    run = tmp_path / 'run'
    (run / 'memory').mkdir(parents=True)
    (run / 'memory/agent_transcript.jsonl').write_text('', encoding='utf-8')
    directory = tmp_path / 'attempt_01'
    write(directory / 'cleanup.json', {'verified': True})
    write(directory / 'environment.json', {'sys_executable': 'F:/conda_env/flyagent/python.exe', 'cuda_available': True})
    write(tmp_path / 'manifest.json', {'source_hashes': {}, 'tasks': [{}],
        'plugins': {'plugins': {'enabled': ['capabilities.place_memory']}}})
    write(tmp_path / 'progress.json', {'status': 'complete', 'cases': [{'status': 'model_error', 'attempts': [
        {'number': 1, 'directory': str(directory), 'result': {'run_dir': str(run), 'error': 'provider failed before first action'}}]}]})
    write(tmp_path / 'admission.json', {'status':'passed', 'plan_probe_status':'passed', 'groups_completed': 3,
        'receipts': [{'status': 'ok', 'attempt': 1}] * 15})
    write(tmp_path / 'historical_request_replay.json', {'passed': True})
    write(tmp_path / 'pixel_place_admission.json', {'passed': True})
    for filename in ('comparison.md', 'comparison.html', 'context_modules.html', 'case_diagnostics_01.html'):
        (tmp_path / filename).write_text('captured model error', encoding='utf-8')
    stats = {'capability_integrity': {'failures': []}, 'max_calls_in_observed_window': None}
    monkeypatch.setattr('tests.manual.aerial_vln_plan_context_probe.run_stats', lambda result: stats)
    result = accept(tmp_path)
    assert result['passed']
    assert result['requests'] == 0
    assert stats['max_calls_in_observed_window'] is None
