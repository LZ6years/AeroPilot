"""Release AirSim SDK wrappers without depending on a particular SDK version.

Ownership stays with the caller, which clears its reference after closing.
VehicleClient normally exposes the msgpack RPC transport as ``client`` rather
than providing a public close method itself. Cleanup failures are not hidden.
"""

from typing import Any


def close_airsim_client(client: Any, *, original_error: Exception | None = None) -> None:
    if client is None:
        return
    close = getattr(client, "close", None)
    if not callable(close):
        close = getattr(getattr(client, "client", None), "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            if original_error is None:
                raise
            # Connection setup failures remain primary; Benchmark can still
            # report a simultaneous release failure rather than hiding it.
            errors = list(getattr(original_error, "cleanup_errors", ()))
            errors.append({"stage": "airsim_client_close", "error_type": type(exc).__name__, "error": str(exc)})
            original_error.cleanup_errors = errors
