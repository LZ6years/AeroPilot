"""Real GPU events must exist; missing data is not a zero-time kernel run."""
import sqlite3
import pytest
from tests.manual.nsight_stage_probe import summarize_sqlite
from tests.manual.nsight_stage_probe import script_target
from tests.manual.nsight_stage_probe import should_capture


def test_capture_trigger_excludes_cold_and_never_repeats():
    assert not should_capture(2, 10, 9, False, True)
    assert should_capture(3, 10, 4, False, True)
    assert not should_capture(4, 10, 4, True, True)
    assert not should_capture(3, 10, 4, False, False)
    assert should_capture(10, 10, 1, False, True)


def test_nsight_target_has_no_nested_inline_python_source():
    original = ['python.exe', '-u', '-X', 'faulthandler', '-c', 'run_name="__main__"',
                'F:/snapshot', 'F:/snapshot/tests/manual/read_only_concurrency_probe.py',
                'child', '--root', 'F:/output', '--sample', '1', '--workers', '1']
    target = script_target(original)
    assert '-c' not in target
    assert target[4].endswith('nsight_stage_probe.py')
    assert target[5:] == original[8:]


def test_export_summary_requires_kernel_table(tmp_path):
    path = tmp_path / 'trace.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)')
    with pytest.raises(ValueError, match='no actual CUDA'):
        summarize_sqlite(path)
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,1000000010)')
    result = summarize_sqlite(path)
    assert result['CUPTI_ACTIVITY_KIND_KERNEL'] == {'count': 1, 'sum_sec': 1.0}
    assert result['CUPTI_ACTIVITY_KIND_MEMCPY'] is None
