"""Boundary and isolation checks for operator-requested terminal rescoring."""
import copy
import pytest
from tests.manual.aerial_vln_rescore_terminal import rescore


@pytest.mark.parametrize('distance,sr,sr25', [(19.9, True, True), (20, True, True), (24.9, False, True), (25, False, False)])
def test_blocked_uses_final_distance_without_mutating_source(distance, sr, sr25):
    result = dict(status='stopped', completed=False, stopped=True, stop_source='runtime.report_blocked',
                  metrics=dict(distance_to_goal=distance, success=False, success_25m=False, ndtw=.4, sdtw=0))
    original = copy.deepcopy(result)
    updated = rescore(result)
    assert result == original
    assert updated['metrics']['success'] == sr
    assert updated['metrics']['success_25m'] == sr25
    assert updated['metrics']['sdtw'] == .4 * sr


@pytest.mark.parametrize('extra', [{'error': 'API failure'}, {'infrastructure_error': {'stage': 'provider'}}, {'stop_source': 'safety'}])
def test_technical_or_unrecognized_stops_are_not_promoted(extra):
    result = dict(status='stopped', completed=False, stop_source='runtime.report_blocked',
                  metrics=dict(distance_to_goal=1, success=False, ndtw=.4, sdtw=0))
    result.update(extra)
    assert rescore(result) == result
