"""Tests for user-facing FlyAgent messages."""

from __future__ import annotations

import pytest

from src.agent.messages import AgentMessage


def test_agent_message_roundtrip():
    message = AgentMessage.from_dict(
        {
            "type": "progress",
            "content": "我正在观察前方区域。",
            "evidence_refs": [{"kind": "frame", "ref": "artifacts/rgb/frame.png"}],
            "confidence": "medium",
            "requires_user_reply": False,
        }
    )

    assert message.to_dict() == {
        "type": "progress",
        "content": "我正在观察前方区域。",
        "evidence_refs": [{"kind": "frame", "ref": "artifacts/rgb/frame.png"}],
        "confidence": "medium",
        "requires_user_reply": False,
    }


def test_agent_message_rejects_invalid_type_and_confidence():
    with pytest.raises(ValueError, match="type"):
        AgentMessage.from_dict({"type": "debug", "content": "bad", "confidence": "medium"})

    with pytest.raises(ValueError, match="confidence"):
        AgentMessage.from_dict({"type": "progress", "content": "bad", "confidence": "certain"})


def test_agent_message_question_can_require_user_reply():
    message = AgentMessage.from_dict(
        {
            "type": "question",
            "content": "你希望我优先看左侧还是右侧？",
            "confidence": "low",
            "requires_user_reply": True,
        }
    )

    assert message.requires_user_reply is True


def test_agent_message_rejects_empty_content_and_raw_thinking():
    with pytest.raises(ValueError, match="content"):
        AgentMessage.from_dict({"type": "progress", "content": "   ", "confidence": "medium"})

    with pytest.raises(ValueError, match="raw thinking"):
        AgentMessage.from_dict(
            {
                "type": "progress",
                "content": "我会继续观察。",
                "confidence": "medium",
                "chain_of_thought": "hidden reasoning",
            }
        )
