"""Do not drop invalid/late calls or double count repeated cost snapshots."""

from evaluation.experiments.compression_cost import normalize_usage, summarize_calls


def test_usage_normalizes_cached_tokens_without_counting_them_twice():
    assert normalize_usage({
        "input_tokens": 100, "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 30},
    }) == {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30}
    assert normalize_usage(None)["input_tokens"] is None


def test_lifecycle_deduplication_preserves_invalid_response_cost_and_pending_calls():
    rows = [
        {"call_id": "a", "status": "started"},
        {"call_id": "a", "status": "error", "usage": {"prompt_tokens": 100, "completion_tokens": 7}},
        {"call_id": "a", "status": "started"},
        {"call_id": "b", "status": "started"},
    ]
    result = summarize_calls(rows)
    assert result["call_count"] == 2
    assert result["known_input_tokens"] == 100
    assert result["pending_call_count"] == 1
    assert result["error_call_count"] == 1
    assert result["missing_input_tokens_call_count"] == 1
