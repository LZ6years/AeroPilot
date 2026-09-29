"""Process-supervision fixtures used by product failure-contract tests."""

from tests.manual.process_diagnostics import supervise
from tests.manual.supervised_flight_probe import cleanup_owned, write_review

__all__ = ["cleanup_owned", "supervise", "write_review"]
