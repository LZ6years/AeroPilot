"""Shared source hashing for reproducible manual experiments, excluding secrets."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_hashes() -> dict[str, str]:
    """Capture executable source/config drift, excluding credentials and data."""
    files = sorted([*ROOT.glob('src/**/*.py'), *ROOT.glob('plugins/**/*.py'),
                    *ROOT.glob('plugins/**/*.json'), *ROOT.glob('configs/**/*.json'),
                    *ROOT.glob('evaluation/**/*.py'), *ROOT.glob('tests/manual/*.py'), *ROOT.glob('tests/manual/*.ps1'),
                    *ROOT.glob('src/agent/planning/prompts/*.txt'), *ROOT.glob('memory/*.md')])
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
