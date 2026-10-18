"""Bounded, credential-redacted diagnostics for provider HTTP error envelopes.

Only public error fields are retained, never the echoed request or headers.
These diagnostics do not change retry classification.
"""
import re


def http_error_diagnostic(exc, *, api_key=""):
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}
    envelope = body.get("error", body)
    if not isinstance(envelope, dict):
        return {}
    result = {}
    for field in ("message", "code", "type", "param"):
        value = envelope.get(field)
        if not isinstance(value, (str, int)):
            continue
        text = str(value)
        if api_key:
            text = text.replace(api_key, "[redacted]")
        text = re.sub(r"data:[^\s\"']+", "[data omitted]", text)
        text = re.sub(r"(?i)bearer\s+[^\s\"',;]+", "Bearer [redacted]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
        text = re.sub(r"https?://[^\s\"']+", "[url omitted]", text)
        result[field] = " ".join(text.split())[:1000]
    return result
