"""Keep adapter counters distinct, reject PID prefixes and unknown samples."""
import json
from tests.manual.windows_gpu_memory import summarize_process_memory


def test_counter_identity_unknowns_and_partial_final_line(tmp_path):
    rows = [{'utc': '2026-09-27T00:00:00Z', 'counters': [
        {'instance': 'pid_123_luid_a', 'path': r'\GPU Process Memory\Dedicated Usage', 'bytes': 100, 'status': 0},
        {'instance': 'pid_123_luid_a', 'path': r'\GPU Process Memory\Shared Usage', 'bytes': None, 'status': 1},
        {'instance': 'pid_1234_luid_a', 'path': r'\GPU Process Memory\Dedicated Usage', 'bytes': 999, 'status': 0},
        {'instance': 'pid_123_luid_b', 'path': r'\GPU Process Memory\Dedicated Usage', 'bytes': 200, 'status': 0},
    ]}]
    (tmp_path/'windows_gpu_memory.jsonl').write_text(json.dumps(rows[0])+'\n{"partial":')
    result = summarize_process_memory(tmp_path, 123)
    assert result['samples'] == 1
    assert result['invalid_rows'] == 1
    assert result['invalid_counters'] == 1
    assert len(result['instances']) == 2
    assert result['instances']['pid_123_luid_a']['dedicated usage']['max_bytes'] == 100
    assert 'shared usage' not in result['instances']['pid_123_luid_a']


def test_missing_counter_file_is_not_zero_memory(tmp_path):
    assert summarize_process_memory(tmp_path, 123)['instances'] == {}
