"""Timeline accounting excludes nested CPU waits from GPU activity."""
import json
import pytest
import sys
from types import SimpleNamespace
from tests.manual.cuda_timeline import summarize_trace


def test_overlapping_kernels_and_host_wait_are_not_added(tmp_path):
    path = tmp_path / 'trace.json'
    path.write_text(json.dumps({'traceEvents': [
        {'ph': 'X', 'cat': 'kernel', 'name': 'a', 'ts': 0, 'dur': 10},
        {'ph': 'X', 'cat': 'kernel', 'name': 'b', 'ts': 5, 'dur': 10},
        {'ph': 'X', 'cat': 'cuda_runtime', 'name': 'cudaDeviceSynchronize', 'ts': 0, 'dur': 40},
        {'ph': 'X', 'cat': 'gpu_memcpy', 'name': 'HtoD', 'ts': 20, 'dur': 5},
    ]}))
    result = summarize_trace(path)
    assert result['device_active_union_sec'] == 20 / 1e6
    assert result['device_gap_sec'] == pytest.approx(5 / 1e6)
    assert result['host_sync']['sum_sec'] == 40 / 1e6


def test_cpu_only_trace_rejected(tmp_path):
    path = tmp_path / 'trace.json'
    path.write_text('{"traceEvents": []}')
    with pytest.raises(ValueError, match='No CUDA kernel'):
        summarize_trace(path)


def test_profiled_batch_is_not_normal_warm_measurement(tmp_path):
    from tests.manual.read_only_concurrency_probe import condition_summary
    (tmp_path / 'results.json').write_text(json.dumps([
        {'cold': True, 'wall_sec': 10}, {'cold': False, 'wall_sec': 2},
        {'cold': False, 'wall_sec': 99, 'profiled': True},
    ]))
    summary = condition_summary(tmp_path)
    assert summary['completed_batches'] == 3
    assert summary['warm_p50_sec'] == 2
    assert summary['warm_p95_sec'] == 2


def test_cuda_profiler_capability_rejected_before_inference(tmp_path, monkeypatch):
    from tests.manual.cuda_timeline import require_cuda_profiler
    torch = SimpleNamespace(__version__='test', cuda=SimpleNamespace(is_available=lambda: True),
        profiler=SimpleNamespace(supported_activities=lambda: {'CPU'}, kineto_available=lambda: True,
                                 ProfilerActivity=SimpleNamespace(CUDA='CUDA')))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    with pytest.raises(ValueError, match='cannot capture CUDA'):
        require_cuda_profiler(tmp_path)
    assert json.loads((tmp_path / 'profiler_environment.json').read_text())['cuda_available']


def test_partial_run_counts_persisted_batches(tmp_path):
    from tests.manual.read_only_concurrency_probe import condition_summary
    for i in (2, 1):
        target = tmp_path / f'r{i}'
        target.mkdir()
        (target / 'results.json').write_text(json.dumps({'repeat': i, 'cold': i == 1, 'wall_sec': i}))
    summary = condition_summary(tmp_path)
    assert summary['completed_batches'] == 2
    assert summary['cold_sec'] == 1
    assert summary['warm_p50_sec'] == 2
