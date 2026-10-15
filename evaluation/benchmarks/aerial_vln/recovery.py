"""Compatibility import; implementation lives in core.campaign.recovery."""
import sys
from evaluation.benchmarks.core.campaign import recovery as _implementation
sys.modules[__name__] = _implementation
