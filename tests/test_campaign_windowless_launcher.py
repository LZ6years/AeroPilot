"""Exercise the actual Windows GUI launcher, console ownership and pipe draining."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows process creation contract")
def test_windowless_launcher_and_completion(tmp_path):
    root = Path(__file__).resolve().parents[1]
    compiler = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    executable = tmp_path / "launcher.exe"
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/reference:System.Web.Extensions.dll",
                    f"/out:{executable}", str(root / "tests/manual/campaign_windowless_launcher.cs")],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    probe = tmp_path / "probe.py"
    probe.write_text("import ctypes, sys\nprint('console=' + str(ctypes.windll.kernel32.GetConsoleWindow()))\n"
                     "print(sys.executable)\nprint('error capture', file=sys.stderr)\nraise SystemExit(7)\n", encoding="utf-8")
    progress = tmp_path / "progress.json"
    config = tmp_path / "launcher.json"
    config.write_text(json.dumps(dict(interpreter=sys.executable, arguments=f'-u "{probe}"',
        working_directory=str(tmp_path), log_directory=str(tmp_path / "logs"),
        progress=str(progress), completion_status="custom_complete")), encoding="utf-8")
    child = subprocess.run([str(executable), str(config)], timeout=20)
    assert child.returncode == 7
    stdout = next((tmp_path / "logs").glob("*.stdout.log")).read_text(encoding="utf-8")
    assert "console=0" in stdout and sys.executable in stdout
    assert "error capture" in next((tmp_path / "logs").glob("*.stderr.log")).read_text(encoding="utf-8")
    progress.write_text(json.dumps({"status": "custom_complete", "receipt": "x" * 3_000_000}), encoding="utf-8")
    assert subprocess.run([str(executable), str(config)], timeout=20).returncode == 0
    assert len(list((tmp_path / "logs").glob("*.stdout.log"))) == 1
