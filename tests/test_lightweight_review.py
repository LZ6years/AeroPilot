"""Compact reporting and trajectory contracts, independent of model behavior."""
import copy
from pathlib import Path

import pytest

from evaluation.benchmarks.core import BenchmarkAuditLog, BenchmarkCaseReview, BenchmarkReviewRenderer
from evaluation.benchmarks.core.trajectory_plots import ned_display_points, write_ned_comparison


def test_ned_display_conversion_is_copy_only():
    source = [[1, 2, -3], [4, 5, 0]]
    original = copy.deepcopy(source)
    assert ned_display_points(source) == [[1., 2., 3.], [4., 5., 0.]]
    assert source == original
    with pytest.raises(ValueError):
        ned_display_points([[0, float('nan'), 1]])


def test_comparison_plots_handle_missing_and_single_point(tmp_path):
    assert write_ned_comparison(tmp_path, [], [])['reason'] == 'empty_actual_trajectory'
    result = write_ned_comparison(tmp_path, [[1, 2, -3]], [])
    assert result['status'] == 'ok' and not result['has_reference']
    assert Path(result['path_2d']).is_file() and Path(result['path_3d']).is_file()


def test_default_review_omits_large_events_but_detail_preserves_them(tmp_path):
    audit = BenchmarkAuditLog(run_dir=str(tmp_path), started_at=None, ended_at=None,
        task_initial_observation_ref=None, observations=(), frames=(), poses=(), actions=(),
        events=({'event': 'private_debug', 'payload': 'UNIQUE_RAW_DEBUG' * 1000},),
        debug_events=(), planner_debug=())
    case = BenchmarkCaseReview('case1', 'Test', 'v1', '<script>instruction</script>', {}, {}, audit, ())
    compact = BenchmarkReviewRenderer().write_case(tmp_path, case)
    page = Path(compact['case_html']).read_text(encoding='utf-8')
    assert 'UNIQUE_RAW_DEBUG' not in page and 'Observation Gallery' not in page
    assert '&lt;script&gt;' in page
    detailed = BenchmarkReviewRenderer(detailed_audit=True).write_case(tmp_path, case)
    detail = Path(detailed['case_html']).read_text(encoding='utf-8')
    assert Path(detailed['case_html']).name == 'audit_review.html'
    assert 'UNIQUE_RAW_DEBUG' in detail
    assert len(page) < len(detail) / 2
    assert Path(compact['case_html']).read_text(encoding='utf-8') == page


def test_decision_plan_uses_captured_state_and_never_later_decisions(tmp_path):
    from evaluation.benchmarks.core.review import _compact_decision, _provider_round
    plan = {"revision": 2, "explanation": "Route", "steps": [
        {"id": "gate", "description": "PRIOR_GATE", "status": "in_progress"}]}
    future = {"revision": 99, "steps": [{"id": "roof", "description": "FUTURE_ROOF", "status": "completed"}]}
    record = {"loop_schema": "unified_tool_loop_v2", "decision_id": "d1", "decision_index": 1,
              "current_task_state": {"task_plan": plan}, "model_context": {}, "model_rounds": []}
    audit = BenchmarkAuditLog(run_dir=str(tmp_path), started_at=None, ended_at=None,
        task_initial_observation_ref=None, observations=(), frames=(), poses=(), actions=(),
        events=(), debug_events=(), planner_debug=(record, {"current_task_state": {"task_plan": future}}))
    html = _compact_decision(record, audit, (), tmp_path, [], {})
    assert "Plan revision before decision: 2" in html and "PRIOR_GATE" in html
    assert "FUTURE_ROOF" not in html
    missing = {**record, "current_task_state": {"task_plan": None}}
    html = _compact_decision(missing, audit, (), tmp_path, [], {})
    assert "No plan created" in html and "FUTURE_ROOF" not in html
    round_html = _provider_round({"round_index": 1, "current_task_state": {"task_plan": plan}}, tmp_path, [])
    assert "Captured task state before this round" in round_html and "PRIOR_GATE" in round_html
