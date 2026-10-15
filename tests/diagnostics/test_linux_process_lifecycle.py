"""Real Linux process-group lifecycle tests, isolated from UE and other jobs."""
import os
import platform
import signal
import sys
import time
from pathlib import Path

import pytest

from src.sim_server.process_manager import ProcessManager

pytestmark = pytest.mark.skipif(platform.system() != 'Linux', reason='Real Linux process groups required')


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for test process')
        time.sleep(.03)


@pytest.mark.parametrize('early_parent,ignore_term', [(False, False), (True, False), (True, True)])
def test_descendants_cleaned_without_touching_other_group(tmp_path, early_parent, ignore_term):
    manager = ProcessManager(terminate_timeout_sec=.2, kill_timeout_sec=3)
    child_file = tmp_path / 'child.pid'
    child = ("import os,signal,time,pathlib; "
             + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else '')
             + f"pathlib.Path({str(child_file)!r}).write_text(str(os.getpid())); time.sleep(60)")
    script = tmp_path / 'parent.py'
    script.write_text('import subprocess,sys,time\n'
                      + f'subprocess.Popen([sys.executable,"-c",{child!r}])\n'
                      + ('time.sleep(.1)\n' if early_parent else 'time.sleep(60)\n'), encoding='utf-8')
    ids = []
    try:
        own = manager.start('own', sys.executable, [str(script)])
        ids.append(own)
        other = manager.start('other', sys.executable, ['-c', 'import time;time.sleep(60)'])
        ids.append(other)
        wait_for(child_file.exists)
        if early_parent:
            wait_for(lambda: manager._processes['own'].poll() is not None)
        manager.stop('own')
        assert not manager._group_alive(own)
        assert manager._group_alive(other)
        assert 'own' not in manager._processes
        manager.stop('own')  # Repeated cleanup is harmless.
        manager.stop('other')
        assert not manager._group_alive(other)
    finally:
        # Emergency cleanup only for the test's explicitly recorded group IDs.
        for group in ids:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in manager._processes.values():
            proc.wait(timeout=5)


def test_clean_parent_exit(tmp_path):
    manager = ProcessManager()
    manager.start('short', sys.executable, ['-c', 'pass'])
    manager._processes['short'].wait(timeout=5)
    manager.stop('short')
    assert not manager._processes and not manager._process_groups
