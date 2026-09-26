"""Human-only AirSim spans. Nested durations are inclusive, never additive.

Each backend operation drains the bounded span buffer to a JSONL audit, even
when RPC raises. RPC counts refer to submitted SDK methods; future waiting is
a child span and is not counted as a second request.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path


class AirSimPerformance:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.spans = []
        self.stack = []
        self.rpc_count_total = 0

    @contextmanager
    def span(self, name, *, rpc=False):
        self.rpc_count_total += int(rpc)
        item = {"name": name, "rpc": rpc, "parent": self.stack[-1] if self.stack else None,
                "started": time.monotonic()}
        index = len(self.spans)
        self.spans.append(item)
        self.stack.append(index)
        try:
            yield
        except BaseException as exc:
            item["error_type"] = type(exc).__name__
            raise
        finally:
            item["elapsed_sec"] = time.monotonic() - item["started"]
            self.stack.pop()

    def flush(self, operation):
        payload = {"operation": operation, "spans": self.spans,
                   "rpc_count": sum(s["rpc"] for s in self.spans)}
        self.spans = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")
        return payload


class MeasuredAirSimClient:
    """Transparent SDK facade; preserves methods, exceptions and native futures."""
    def __init__(self, client, performance):
        self._raw_client = client
        self._performance = performance

    def __getattr__(self, name):
        value = getattr(self._raw_client, name)
        if not callable(value):
            return value

        def measured(*args, **kwargs):
            with self._performance.span("rpc." + name, rpc=True):
                return value(*args, **kwargs)
        return measured
