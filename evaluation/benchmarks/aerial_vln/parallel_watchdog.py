"""Compatibility import; implementation lives in core.campaign.watchdog."""
import sys
from evaluation.benchmarks.core.campaign import watchdog as _implementation
sys.modules[__name__] = _implementation
