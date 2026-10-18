"""Provider diagnostics retain useful errors without keys or echoed payloads."""
from types import SimpleNamespace
from src.llm.http_errors import http_error_diagnostic


def test_public_error_fields_only():
    exc = SimpleNamespace(body={"error": {"message": "Invalid tool call id", "code": "invalid_request",
        "param": "messages.3", "type": "bad_request", "request": {"messages": "private"}}, "headers": "private"})
    assert http_error_diagnostic(exc) == {"message": "Invalid tool call id", "code": "invalid_request",
                                         "param": "messages.3", "type": "bad_request"}


def test_redacts_credentials_media_and_urls_before_truncation():
    exc = SimpleNamespace(body={"message": "key secret123 Bearer tokenXYZ sk-abc123 https://host/?key=xyz data:image/png;base64,abcdef " + "x" * 2000})
    out = http_error_diagnostic(exc, api_key="secret123")["message"]
    assert len(out) == 1000
    for secret in ("secret123", "tokenXYZ", "sk-abc123", "host", "abcdef"):
        assert secret not in out


def test_unknown_body_does_not_dump_html_or_request():
    assert http_error_diagnostic(SimpleNamespace(body="<html>secret</html>")) == {}
    assert http_error_diagnostic(SimpleNamespace(body={"request": "private"})) == {}
