"""Independent offline evidence/experience services.

The optional read-only capability lives under plugins/capabilities and is disabled
by default. This package does not register tools or import Agent/Runtime types.
"""

from .evidence import EvidenceBundle, read_run
from .experiment import ExperimentSettings, run_experiment

__all__ = ["EvidenceBundle", "ExperimentSettings", "read_run", "run_experiment"]
