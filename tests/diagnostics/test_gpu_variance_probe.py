"""GPU variance joins must preserve missing samples and avoid cold/warm mixing."""
import json

from tests.manual.gpu_variance_probe import batch_resources


def test_batch_resource_windows_and_external_process_identity(tmp_path, monkeypatch):
    import tests.manual.gpu_variance_probe as probe
    spans = [dict(name='experiment.batch', time=10, ended=12, decision=1, inclusive_sec=2),
             dict(name='vision.sam.forward', time=10.1, ended=11.1, decision=1, inclusive_sec=1),
             dict(name='experiment.batch', time=15, ended=16, decision=2, inclusive_sec=1)]
    monkeypatch.setattr(probe, 'summarize', lambda path: {'spans': spans})
    (tmp_path / 'process.json').write_text(json.dumps({'pid': 123}))
    sample = {'monotonic_sec': 11, 'devices': [{'used_mib': 5555}],
              'performance': {'rows': [{'sm_mhz': 1500, 'power_w': None}]},
              'compute_processes': {'rows': [{'pid': 123}, {'pid': 456}]}}
    (tmp_path / 'gpu_resources.jsonl').write_text(json.dumps(sample))
    rows = batch_resources(tmp_path)
    assert rows[0]['sam_sec'] == 1
    assert rows[0]['ranges']['sm_mhz'] == [1500, 1500]
    assert rows[0]['visible_external_compute_pids'] == [456]
    assert rows[1]['sample_count'] == 0
    assert rows[1]['ranges']['used_mib'] is None
