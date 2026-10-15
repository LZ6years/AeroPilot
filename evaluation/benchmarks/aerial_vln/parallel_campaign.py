"""AerialVLN compatibility entry for the dataset-independent campaign engine.

Existing imports share the actual core module object. Existing run/watch commands
resume their pinned controller; prepare/validate use the Aerial dataset adapter.
"""
import sys
from pathlib import Path
from evaluation.benchmarks.core.campaign import parallel as _implementation

if __name__ == "__main__":
    _implementation.main(default_dataset="aerial_vln", default_output=Path("H:/FlyAgent-Eval/aerialvln_s_20261007"))
else:
    sys.modules[__name__] = _implementation
