"""Shared error facts; boundaries supply context and Runtime owns recovery.

An exception after dispatch cannot establish whether a remote action occurred.
Legacy adaptation is for offline readers only and never rewrites raw records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ACTION_DIAGNOSTIC_KEY = "_flyagent_diagnostic"
EXECUTION_STATES = frozenset({"not_started", "started", "completed", "partial", "unknown"})


@dataclass(frozen=True)
class Diagnostic:
    code: str
    phase: str = "unknown"
    reason: str | None = None
    parameter_path: str | None = None
    scope: str | None = None
    source_tool_call_id: str | None = None
    role: str | None = None
    execution: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("diagnostic code must be non-empty")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("diagnostic phase must be non-empty; use unknown")
        if self.execution is not None and self.execution not in EXECUTION_STATES:
            raise ValueError("unsupported diagnostic execution state")
        for key, value in asdict(self).items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"diagnostic {key} must be a string")

    def to_dict(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def exception_diagnostic(error: BaseException, *, phase: str = "unknown",
                         execution: str | None = None, code: str = "execution_failed",
                         **facts: Any) -> dict[str, str]:
    """Convert error facts without importing LLM or Runtime implementations."""
    supplied = getattr(error, "diagnostic", None)
    data = dict(supplied) if isinstance(supplied, dict) else {}
    data.setdefault("code", getattr(error, "code", code))
    data.setdefault("phase", phase)
    data.setdefault("reason", str(error))
    if getattr(error, "path", None) is not None:
        data.setdefault("parameter_path", error.path)
    if execution is not None:
        data.setdefault("execution", execution)
    for key, value in facts.items():
        if value is not None:
            data.setdefault(key, value)
    return Diagnostic(**data).to_dict()


def action_diagnostic(data: dict[str, Any], *, call_id: str, ok: bool, phase: str,
                      execution: str = "unknown", reason: str = ""):
    """Promote local ActionResult facts without changing the remote RPC schema."""
    clean = dict(data)
    diagnostic = clean.pop(ACTION_DIAGNOSTIC_KEY, None)
    if diagnostic is None and not ok:
        diagnostic = Diagnostic("execution_failed", phase, reason=reason,
                                execution=execution).to_dict()
    if diagnostic is not None:
        diagnostic = {**diagnostic, "source_tool_call_id": call_id, "scope": "call"}
        diagnostic.setdefault("execution", execution)
        diagnostic = Diagnostic(**diagnostic).to_dict()
    return clean, diagnostic


def legacy_result_diagnostic(result: dict[str, Any]) -> dict[str, str] | None:
    """Build an isolated compatibility view of historical result diagnostics."""
    if isinstance(result.get("diagnostic"), dict):
        return Diagnostic(**result["diagnostic"]).to_dict()
    data = result.get("data") or {}
    rejection = data.get("rejection")
    if isinstance(rejection, dict):
        allowed = {key: value for key, value in rejection.items()
                   if key in Diagnostic.__dataclass_fields__}
        allowed.setdefault("code", "legacy_rejection")
        allowed.setdefault("phase", "validation")
        allowed.setdefault("execution", "not_started" if rejection.get("executed") is False else "unknown")
        allowed.setdefault("reason", result.get("message", ""))
        return Diagnostic(**allowed).to_dict()
    if result.get("status") in {"error", "rejected"} or data.get("error_type"):
        return Diagnostic(str(data.get("error_type") or "legacy_error"),
                          reason=result.get("message", ""), execution="unknown").to_dict()
    return None
