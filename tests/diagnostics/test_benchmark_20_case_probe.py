"""Regression checks for frozen selection and supervised 10+10 progress."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from tests.manual import benchmark_20_case_probe as batch


def cases():
    aerial = [{'benchmark': 'aerial', 'case_id': f'a{i}', 'difficulty': level,
               'scene': str(i), 'historical': True}
              for i, level in enumerate(['low'] * 4 + ['medium'] * 3 + ['high'] * 3)]
    travel = [{'benchmark': 'travel', 'case_id': case_id, 'difficulty': 'easy',
               'scene': 'map', 'historical': True}
              for case_id in batch.TRAVEL_HISTORICAL]
    travel.extend({'benchmark': 'travel', 'case_id': f'hard{i}', 'difficulty': 'hard',
                   'scene': 'map', 'historical': False} for i in range(3))
    return aerial + travel


def test_frozen_case_contract_rejects_duplicates_and_distribution():
    selected = cases()
    batch.validate_manifest_cases(selected)
    duplicated = [dict(case) for case in selected]
    duplicated[-1]['case_id'] = duplicated[0]['case_id']
    with pytest.raises(ValueError, match='20 distinct'):
        batch.validate_manifest_cases(duplicated)
    wrong = [dict(case) for case in selected]
    wrong[10]['difficulty'] = 'hard'
    with pytest.raises(ValueError, match='difficulty order'):
        batch.validate_manifest_cases(wrong)


def test_historical_identity_and_modern_city_error_recorded():
    selected = cases()
    selected[12]['historical'] = False
    with pytest.raises(ValueError, match='historical identity'):
        batch.validate_manifest_cases(selected)
    assert batch.TRAVEL_HISTORICAL[2].startswith('ModernCityMap__')


def test_native_crash_keeps_partial_html_and_cleanup_evidence(tmp_path):
    item = cases()[0]

    def supervisor(_command, directory):
        directory.mkdir(parents=True)
        (directory / 'stdout.log').write_text('started\n', encoding='utf-8')
        (directory / 'stderr.log').write_text('native crash\n', encoding='utf-8')
        return {'exit_code': -11, 'elapsed_seconds': 2.0, 'descendants': []}

    def cleaner(*_args):
        return [{'scene': 's', 'remote_stop': {'cleanup_verified': True},
                 'endpoint': {'status': 'unreachable'}, 'tunnels': []}]

    status = batch.run_one(tmp_path, 0, item, supervisor=supervisor, cleaner=cleaner)
    directory = batch.case_dir(tmp_path, 0, item)
    assert status['status'] == 'incomplete'
    assert status['cleanup_verified']
    assert 'no final benchmark score' in (directory / 'review.html').read_text(encoding='utf-8')
    assert json.loads((directory / 'supervision.json').read_text(encoding='utf-8'))['exit_code'] == -11


def test_audit_metrics_accepts_missing_response_after_provider_timeout(tmp_path):
    item = cases()[0]
    audit = tmp_path / 'evaluation_output/evaluation/cases' / item['case_id'] / 'audit.json'
    audit.parent.mkdir(parents=True)
    batch.write_json(audit, {'agent_transcript': [], 'planner_debug': [
        {'model_rounds': [{'response': {'usage': {'total_tokens': 12}}},
                          {'response': None}]}]})
    metrics = batch.audit_metrics(tmp_path, item)
    assert metrics['token_usage']['total_tokens'] == 12
    assert metrics['plan_calls'] == 0


def test_infrastructure_error_pauses_and_resumes_same_id_in_new_directory(tmp_path, monkeypatch):
    selected = cases()
    root = tmp_path / 'batch'
    root.mkdir()
    batch.write_json(root / 'manifest.json', {'model': 'qwen3.8-flash',
                     'dataset_sha256': {'aerial': 'a', 'travel': 'b'}, 'cases': selected})
    monkeypatch.setattr(batch, 'load_model_config', lambda _path: SimpleNamespace(model_name='qwen3.8-flash'))
    monkeypatch.setattr(batch, 'preflight', lambda _root, _manifest: {})
    monkeypatch.setattr(batch, 'refresh_reviews', lambda *_args: None)
    attempts = []

    def fake_run_one(root, index, item, *, directory):
        directory.mkdir(parents=True)
        status = {'status': 'infrastructure_error' if index == 0 and not attempts else 'completed',
                  'cleanup_verified': True, 'metrics': {'success': False},
                  'elapsed_seconds': 1, 'exit_code': 0}
        attempts.append((index, directory))
        batch.write_json(directory / 'status.json', status)
        review = directory / 'evaluation_output/evaluation/cases' / item['case_id'] / 'review.html'
        review.parent.mkdir(parents=True)
        review.write_text('review', encoding='utf-8')
        return status

    monkeypatch.setattr(batch, 'run_one', fake_run_one)
    assert batch.execute(root)['counts'] == {'infrastructure_error': 1, 'pending': 19}
    assert len(attempts) == 1
    batch.execute(root)
    assert len(attempts) == 1  # Plain rerun cannot silently replace failed evidence.
    assert batch.execute(root, resume=True)['counts'] == {'completed': 20}
    assert attempts[1][0] == 0
    assert attempts[1][1].name.endswith('_recovery_01')
    assert (batch.case_dir(root, 0, selected[0]) / 'status.json').exists()
    assert (root / 'aerial_summary.html').exists()
    assert (root / 'travel_summary.html').exists()
    assert (root / 'index.html').exists()


def test_model_error_continues_but_is_excluded_from_navigation_score(tmp_path, monkeypatch):
    selected = cases()
    root = tmp_path / 'batch'
    root.mkdir()
    batch.write_json(root / 'manifest.json', {'model': 'qwen3.8-flash',
                     'dataset_sha256': {'aerial': 'a', 'travel': 'b'}, 'cases': selected})
    monkeypatch.setattr(batch, 'load_model_config', lambda _path: SimpleNamespace(model_name='qwen3.8-flash'))
    monkeypatch.setattr(batch, 'preflight', lambda _root, _manifest: {})
    monkeypatch.setattr(batch, 'refresh_reviews', lambda *_args: None)

    def fake_run_one(root, index, item, *, directory):
        directory.mkdir(parents=True)
        status = {'status': 'model_error' if index == 2 else 'completed',
                  'cleanup_verified': True, 'metrics': {'success': False},
                  'elapsed_seconds': 1, 'exit_code': 0}
        batch.write_json(directory / 'status.json', status)
        review = directory / 'evaluation_output/evaluation/cases' / item['case_id'] / 'review.html'
        review.parent.mkdir(parents=True)
        review.write_text('review', encoding='utf-8')
        return status

    monkeypatch.setattr(batch, 'run_one', fake_run_one)
    summary = batch.execute(root)
    assert summary['counts'] == {'completed': 19, 'model_error': 1}
    assert '&quot;scored&quot;: 3' in (root / 'aerial_summary.html').read_text(encoding='utf-8')


def test_aerial_only_manifest_and_report(tmp_path):
    selected = cases()[:10]
    batch.validate_manifest_cases(selected)
    with pytest.raises(ValueError):
        batch.validate_manifest_cases(selected[:-1])
    result = batch.report(tmp_path, {'model': 'deepseek-flash', 'dataset_sha256': {}, 'cases': selected})
    assert len(result['cases']) == 10
    assert not (tmp_path / 'travel_summary.html').exists()
    assert 'travel_summary.html' not in (tmp_path / 'index.html').read_text(encoding='utf-8')


def test_failed_admission_stops_before_scene_preflight(tmp_path, monkeypatch):
    from src.agent.config import ModelConfig
    config = ModelConfig(model_name='deepseek-flash', thinking_mode='disabled')
    manifest = {'model': config.model_name, 'model_config': config.redacted_summary(),
                'requires_admission': True, 'cases': cases()[:10]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(batch, 'load_model_config', lambda *_: config)
    def forbidden(*_):
        pytest.fail('Scene preflight must not run after failed admission')
    monkeypatch.setattr(batch, 'preflight', forbidden)
    with pytest.raises(ValueError, match='successful API admission'):
        batch.execute(tmp_path)
