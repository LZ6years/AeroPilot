"""Prepare a process-flags-only revision from the ACTIVE frozen source.

No navigation, provider, scoring or recovery policy is changed. Activate via
the existing recovery_revision_handoff.py after the current leases drain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.benchmarks.core.campaign import parallel as control


def build(output):
    overlay = control.read_json(output / "parallel/execution_overlay.json")
    manifest = control.read_json(output / "manifest.json")
    control.verify_frozen(output, manifest)
    control.verify_overlay(overlay, manifest)
    original = Path(overlay.get("execution_source", output / "source"))
    expected = dict(overlay.get("execution_source_hashes", manifest["source_hashes"]))
    changes = {}
    for name in ("src/sim_server/remote/ssh.py", "src/sim_server/remote/tunnel.py"):
        text = (original / name).read_text(encoding="utf-8")
        if "creationflags=" in text:
            raise ValueError("Already configured or unexpected process policy: " + name)
        if name.endswith("ssh.py"):
            first = "result = subprocess.run(command, capture_output=True, text=True,"
            second = "completed = subprocess.run(\n            command,"
            if text.count(first) != 1 or text.count(second) != 1:
                raise ValueError("Unexpected SSH implementation")
            text = text.replace(first, first + '\n                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,')
            text = text.replace(second, second + '\n            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,')
        else:
            if text.count("stderr=log_file,") != 1:
                raise ValueError("Unexpected tunnel implementation")
            text = text.replace("import socket\n", "import socket\nimport os\n")
            text = text.replace("stderr=log_file,", 'stderr=log_file,\n                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,')
        changes[name] = text.encode("utf-8")
    for name, data in changes.items():
        expected[name] = hashlib.sha256(data).hexdigest()
    revision = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    directory = output / "parallel/windowless_revisions" / revision[:16]
    source = directory / "source"
    if not source.exists():
        shutil.copytree(original, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, data in changes.items():
            (source / name).write_bytes(data)
    for name, digest in expected.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source mismatch: " + name)
    candidate = {**overlay, "execution_source": str(source), "execution_source_hashes": expected,
        "execution_source_sha256": revision, "recovery_revision": "windowless_" + revision[:16],
        "windowless_revision": {"changed_files": list(changes), "previous_source": str(original)},
        "recovery_validation_required": True}
    path = directory / "candidate_overlay.json"
    control.write_json(path, candidate)
    print(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
