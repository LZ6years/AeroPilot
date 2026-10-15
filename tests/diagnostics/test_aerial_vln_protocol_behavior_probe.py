"""Regression checks keep saved context immutable and treat blocking honestly."""
import copy
import json

import pytest

from src.agent.planning.task_plan import TaskPlan
from src.agent.transcript import AgentToolCall
from tests.manual.aerial_vln_protocol_behavior_probe import candidate_request, check_decision, request_snapshot


def test_real_rgb_request_snapshot_is_binary_free_serializable_and_citable():
    import base64
    import hashlib
    import io
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 40, 60)).save(stream, format="PNG")
    raw = stream.getvalue()
    image = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    snapshot, media = request_snapshot({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}}]}]})
    saved = json.loads(json.dumps({"request": snapshot, "request_media": media}))
    assert saved["request_media"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert "data" not in saved["request_media"][0]
    assert saved["request_media"][0]["request_path"] == ["messages", 0, "content", 0, "image_url", "url"]


def test_protocol_replacement_preserves_all_other_input_and_original():
    original = {"messages": [
        {"role": "user", "content": "Land here. Continue toward the sign and stop."},
        {"role": "user", "content": "INSTRUCTION CONTEXT\n" + json.dumps({
            "protocol_context": "old", "knowledge_catalog": ["flight"]})},
        {"role": "system", "content": 'CURRENT TASK STATE\n{"task_plan":null}'},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "saved-image"}}]}],
        "tools": [{"function": {"name": "runtime_update_plan", "strict": True}}]}
    before = copy.deepcopy(original)
    candidate, old = candidate_request(original, "whole instruction")
    assert original == before and old == "old"
    restored, _ = candidate_request(candidate, old)
    assert restored == original
    assert candidate["tools"] == original["tools"]


@pytest.mark.parametrize("count", [0, 2])
def test_protocol_replacement_requires_unique_envelope(count):
    messages = [{"content": 'INSTRUCTION CONTEXT\n{"protocol_context":"old"}'}] * count
    with pytest.raises(ValueError, match="exactly one"):
        candidate_request({"messages": messages}, "new")


def test_only_premature_completion_and_remaining_cancellation_are_flagged():
    plan = TaskPlan().updated({"operation": "replace", "explanation": None, "steps": [
        {"id": "later", "description": "Continue past the chimneys", "status": "pending"}]})
    blocked = AgentToolCall("blocked", "runtime.report_blocked", {})
    assert not check_decision([blocked], plan, ["later"])["premature_completion"]
    completed = AgentToolCall("completed", "runtime.report_completed", {})
    assert check_decision([completed], plan, ["later"])["premature_completion"]
    cancel = AgentToolCall("cancel", "runtime.update_plan", {"operation": "patch", "explanation": None,
        "steps": [{"id": "later", "description": None, "status": "cancelled"}]})
    assert check_decision([cancel], plan, ["later"])["cancelled_remaining_step_ids"] == ["later"]
    assert plan.steps[0].status == "pending" and plan.revision == 1
