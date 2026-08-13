"""Compatibility alias for the production AerialVLN evaluation CLI.

Difficulty classification now belongs to
``evaluation.benchmarks.aerial_vln.selection`` and the production runner.
This historical manual entry intentionally contains no selection, simulator,
or reporting implementation of its own.

Use the same arguments as:

    python evaluation/benchmarks/aerial_vln/scripts/run_eval.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.scripts.run_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
