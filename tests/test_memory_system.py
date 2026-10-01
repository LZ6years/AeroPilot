"""Pure V2 context contract tests below Runtime and provider layers."""

from __future__ import annotations

import json

import pytest

from src.memory.context.contracts import (
    ContextFragment,
    FragmentAudit,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    assert_binary_free,
    checkpoint_fingerprint,
)


def test_context_fragment_is_immutable_from_caller_mutation():
    content = {"nested": {"value": 1}}
    inputs = [{"type": "image", "media_ref": {"ref": "frame.png"}}]
    fragment = ContextFragment("test", content, tuple(inputs))
    content["nested"]["value"] = 2
    inputs[0]["media_ref"]["ref"] = "changed.png"

    assert fragment.content == {"nested": {"value": 1}}
    assert fragment.media_refs == (
        {"type": "image", "media_ref": {"ref": "frame.png"}},
    )


def test_fragment_audit_fields_never_enter_model_fragment():
    fragment = ContextFragment("pose", {"agl_m": 3.0})
    audit = FragmentAudit(
        "pose",
        "DefaultPoseContextModule",
        confidence="high",
        warnings=("warning",),
        artifact_refs=("memory/poses.jsonl#1",),
        metadata={"world_pose": {"x": 20}},
        build_duration_ms=1.2,
    )

    assert set(fragment.to_dict()) == {"name", "content", "media_refs"}
    assert audit.to_dict()["metadata"]["world_pose"]["x"] == 20
    assert "world_pose" not in json.dumps(fragment.to_dict())


def test_checkpoint_fingerprint_covers_all_sensory_model_fields():
    first = ContextFragment("pose", {"agl_m": 2.0})
    changed = ContextFragment("pose", {"agl_m": 2.1})

    baseline = checkpoint_fingerprint("obs_1", (first,))

    assert baseline != checkpoint_fingerprint("obs_2", (first,))
    assert baseline != checkpoint_fingerprint("obs_1", (changed,))


def test_checkpoint_rejects_unknown_reason():
    fragment = ContextFragment("pose", {})
    with pytest.raises(ValueError, match="unsupported checkpoint reason"):
        StateCheckpoint(
            checkpoint_id="checkpoint_0001",
            reason="arbitrary",  # type: ignore[arg-type]
            observation_id=None,
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint(None, (fragment,)),
        )


def test_media_ref_requires_content_identity_and_contains_no_payload():
    with pytest.raises(ValueError, match="SHA-256"):
        MediaRef(
            ref="frame.png",
            sha256="bad",
            mime_type="image/png",
            observation_id="obs_1",
            frame_id="frame_1",
            role="current_observation",
        )

    ref = MediaRef(
        ref="frame.png",
        sha256="a" * 64,
        mime_type="image/png",
        observation_id="obs_1",
        frame_id="frame_1",
        role="current_observation",
    )
    assert set(ref.to_dict()).isdisjoint({"data", "base64", "data_url"})


def test_binary_free_guard_checks_nested_values():
    assert_binary_free({"items": [{"ref": "frame.png"}]})
    with pytest.raises(ValueError, match="inline image data"):
        assert_binary_free({"items": [{"data": "data:image/png;base64,AAAA"}]})


def test_instruction_envelope_and_checkpoint_have_distinct_lifetimes():
    envelope = InstructionEnvelope(
        instruction_id="inst_1",
        protocol_context="protocol",
        identity={"name": "FlyAgent"},
        user_preferences={"language": "zh"},
        knowledge_catalog=({"name": "guide"},),
        static_module_guidance=({"name": "pose"},),
    )
    fragment = ContextFragment("pose", {"agl_m": 3.0})
    checkpoint = StateCheckpoint(
        checkpoint_id="checkpoint_0001",
        reason="instruction_start",
        observation_id="obs_1",
        fragments=(fragment,),
        fingerprint=checkpoint_fingerprint("obs_1", (fragment,)),
    )

    assert "identity" in envelope.to_dict()
    assert "identity" not in checkpoint.to_dict()
    assert "fragments" in checkpoint.to_dict()
    assert "fragments" not in envelope.to_dict()
