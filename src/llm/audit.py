"""Generic in-memory image separation and redaction; never owns output paths or agent labels."""
from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any

_PRIVATE_FIELDS = frozenset({
    "authorization", "api_key", "access_token", "reasoning_content",
    "reasoning", "thinking", "analysis",
})


def redact_provider_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[image_data_url]"
    if isinstance(value, list):
        return [redact_provider_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_provider_value(item) for key, item in value.items()}
    return value


def snapshot_request(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate provider image bytes from an otherwise exact request snapshot.

    The returned media entries remain in memory only. Runtime persists each
    digest once, replaces ``data`` with an asset reference, and only then writes
    the decision JSON. Message order and image positions are retained so review
    can reproduce what the provider received without exposing base64.
    """

    media: list[dict[str, Any]] = []

    def clean(value: Any, path: tuple[Any, ...] = ()) -> Any:
        if isinstance(value, str) and value.startswith("data:image/"):
            decoded = _decode_image_data_url(value)
            if decoded is None:
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                media.append(
                    {
                        "sha256": digest,
                        "mime_type": None,
                        "suffix": ".bin",
                        "data": None,
                        "request_path": list(path),
                        "unavailable_reason": "invalid_image_data_url",
                    }
                )
            else:
                content, mime_type, suffix = decoded
                digest = hashlib.sha256(content).hexdigest()
                media.append(
                    {
                        "sha256": digest,
                        "mime_type": mime_type,
                        "suffix": suffix,
                        "data": content,
                        "request_path": list(path),
                    }
                )
            return f"[provider_media sha256={digest}]"
        if isinstance(value, list):
            return [clean(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, dict):
            # Only provider-level private options are excluded. Arbitrary tool
            # schemas/arguments may legitimately contain an "analysis" field;
            # removing those would falsify the captured request.
            return {key: clean(item, (*path, key)) for key, item in value.items()
                    if path or key.lower() not in _PRIVATE_FIELDS}
        return value

    snapshot = clean(payload)
    return snapshot, media


def _decode_image_data_url(value: str) -> tuple[bytes, str, str] | None:
    try:
        header, encoded = value.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header.lower():
        return None
    mime_type = header[5:].split(";", 1)[0].lower()
    suffix = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type)
    if suffix is None:
        return None
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    return (data, mime_type, suffix) if data else None
