"""One SDK-owning worker per session; no task runs on the ASGI event loop."""

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from typing import Any


class SessionWorker:
    def __init__(self, session_id: str):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"operator-{session_id}")

    def submit(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._executor.submit(operation, *args, **kwargs)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def retire(self) -> None:
        """Called by the owner thread after cleanup, without joining itself."""
        self._executor.shutdown(wait=False, cancel_futures=False)
