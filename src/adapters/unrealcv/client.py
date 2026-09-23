"""Probe and release UnrealCV connections without hiding primary failures."""

from typing import Any


def close_unrealcv_client(client: Any, *, original_error: Exception | None = None) -> None:
    """Preserve setup failures and attach simultaneous disconnect diagnostics."""
    if client is None:
        return
    try:
        client.disconnect()
    except Exception as exc:
        if original_error is None:
            raise
        errors = list(getattr(original_error, "cleanup_errors", ()))
        errors.append({"stage": "unrealcv_client_close", "error_type": type(exc).__name__, "error": str(exc)})
        original_error.cleanup_errors = errors


def probe_unrealcv_endpoint(host: str, port: int) -> tuple[bool, str]:
    """Return connectivity independently of temporary-client release errors."""
    client = None
    try:
        from unrealcv import Client

        client = Client((host, int(port)))
        client.connect()
        ready = bool(client.isconnected())
        message = "UnrealCV connected" if ready else "UnrealCV connection was rejected"
    except Exception as exc:
        ready, message = False, f"{type(exc).__name__}: {exc}"
    try:
        close_unrealcv_client(client)
    except Exception as exc:
        message += f"; cleanup error: {type(exc).__name__}: {exc}"
    return ready, message
