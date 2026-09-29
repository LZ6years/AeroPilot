"""Portable review bundles preserve evidence and enforce browser-safe paths."""

from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image
import pytest

from evaluation.benchmarks.core import BenchmarkAuditLog
from tests.support.review import (
    checked,
    export_batch,
    review_exporter,
    validate_bundle,
    write_json,
)


def fixture_batch(tmp_path):
    root = tmp_path / 'batch'
    source = root / 'cases/01_aerial_original_long_id/evaluation_output/evaluation/cases/original_long_id'
    source.mkdir(parents=True)
    image = root / 'frame.png'
    Image.new('RGB', (32, 32), (40, 100, 200)).save(image)
    audit = BenchmarkAuditLog(run_dir=str(root), started_at=None, ended_at=None,
        task_initial_observation_ref=None, observations=(),
        frames=({'id': 'frame_1', 'modality': 'rgb', 'metadata': {'artifact_path': str(image)}},),
        poses=(), actions=(), events=({'event': 'preserved_detail'},),
        debug_events=(), planner_debug=())
    write_json(source / 'audit.json', asdict(audit))
    write_json(source / 'case_result.json', {'episode_id': 'original_long_id', 'status': 'ok', 'metrics': {'success': True}})
    write_json(root / 'manifest.json', {'cases': [{'case_id': 'original_long_id', 'benchmark': 'aerial', 'difficulty': 'low'}]})
    return root, source


def test_export_has_short_paths_full_audit_and_unchanged_sources(tmp_path):
    root, source = fixture_batch(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / 'review'
    result = export_batch(root, output)
    assert result['validation']['image_references'] > 0
    assert result['validation']['max_path_length'] <= 240
    detailed = (output / 'cases/01/audit_review.html').read_text(encoding='utf-8')
    assert 'original_long_id' in detailed
    assert 'preserved_detail' in detailed
    assert 'Observation Gallery' in detailed
    assert (output / 'cases/01/review.html').is_file()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert export_batch(root, output)['validation']['image_references'] > 0


def test_recovery_keeps_failed_attempt_and_latest_result(tmp_path):
    root, source = fixture_batch(tmp_path)
    write_json(source / 'case_result.json', {'episode_id': 'original_long_id', 'status': 'error', 'infrastructure_error': {'error': 'timeout'}})
    output = tmp_path / 'review'
    export_batch(root, output)
    assert '无最终评分' in (output / 'cases/01/review.html').read_text(encoding='utf-8')
    recovery = root / 'cases/01_aerial_original_long_id_recovery_01/evaluation_output/evaluation/cases/original_long_id'
    recovery.mkdir(parents=True)
    (recovery / 'audit.json').write_bytes((source / 'audit.json').read_bytes())
    write_json(recovery / 'case_result.json', {'episode_id': 'original_long_id', 'status': 'ok'})
    result = export_batch(root, output)
    assert len(result['cases'][0]['attempts']) == 2
    assert result['cases'][0]['attempts'][0]['incomplete'] is True
    assert result['cases'][0]['attempts'][1]['incomplete'] is False


def test_missing_image_and_long_path_are_rejected(tmp_path):
    (tmp_path / 'page.html').write_text('<img src="missing.png">', encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid review resource'):
        validate_bundle(tmp_path)
    with pytest.raises(ValueError, match='exceeds 240'):
        checked(tmp_path / ('x' * 200) / 'image.png')


def test_native_exit_partial_page_has_no_score(tmp_path):
    root, source = fixture_batch(tmp_path)
    (source / 'audit.json').unlink()
    attempt = source.parents[3]
    write_json(attempt / 'process.json', {'exit_code': -11})
    output = tmp_path / 'review'
    result = export_batch(root, output)
    assert result['cases'][0]['attempts'][0]['incomplete']
    assert '无最终评分' in (output / 'cases/01/audit_review.html').read_text(encoding='utf-8')


def test_export_failure_is_recorded_without_rewriting_navigation(tmp_path, monkeypatch):
    from tests.support.review import refresh_reviews
    root, source = fixture_batch(tmp_path)
    original = (source / 'case_result.json').read_bytes()
    def fail(*_args):
        raise ValueError('missing image')
    monkeypatch.setattr(review_exporter, 'export_batch', fail)
    with pytest.raises(ValueError, match='missing image'):
        refresh_reviews(root, tmp_path / 'review')
    assert json.loads((root / 'review_failure.json').read_text(encoding='utf-8'))['error'] == 'missing image'
    assert (source / 'case_result.json').read_bytes() == original
