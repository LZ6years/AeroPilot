"""Large-window, image-free and non-blocking history compression tests."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from io import BytesIO

import pytest
from PIL import Image

from src.agent.context.compression import (
    CompactionCallResult,
    CompressedHistorySection,
    CompressionDraft,
    CompressionRequest,
    CompressionWindow,
    EvidenceCatalogEntry,
    HistoryCompactionCoordinator,
    HistoryCompressionConfig,
    HistoryProjectionState,
    ModelHistoryCompactor,
)
from src.agent.context.compression.validator import CompressionValidator
from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.context.window_manager import ContextWindowManager
from src.agent.transcript import (
    AgentToolCall,
    AgentTranscript,
    AssistantTurn,
    ToolResult,
    TranscriptEvent,
)
from src.llm import ModelResponse
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    MediaRef,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.stores import MemoryEventWriter
from src.runtime.task.session import TaskSession
from src.tools.registry import default_registry


class StubCompactor:
    """Thread-safe fake with explicit start/release controls."""

    def __init__(
        self,
        draft: CompressionDraft | None = None,
        *,
        error: Exception | None = None,
        gate: threading.Event | None = None,
        tokens_per_window: int = 100,
    ) -> None:
        self.draft = draft
        self.error = error
        self.gate = gate
        self.tokens_per_window = tokens_per_window
        self.requests: list[CompressionRequest] = []
        self.started = threading.Event()

    def compact(self, request: CompressionRequest) -> CompactionCallResult:
        self.requests.append(request)
        self.started.set()
        if self.gate is not None and not self.gate.wait(5):
            raise TimeoutError("test gate was not released")
        if self.error is not None:
            raise self.error
        refs = sorted(request.allowed_evidence_refs)
        draft = self.draft or CompressionDraft(
            mission_progress=(
                {
                    "summary": "Earlier movement and observations were consolidated.",
                    "evidence_refs": refs[:1],
                },
            ),
        )
        return CompactionCallResult(
            draft=draft,
            usage={"prompt_tokens": self.estimate_input_tokens(request)},
            finish_reason="stop",
            elapsed_seconds=0.01,
        )

    def estimate_input_tokens(self, request: CompressionRequest) -> int:
        return max(1, len(request.windows) * self.tokens_per_window)


def _image_ref(tmp_path, index: int, observation_id: str) -> MediaRef:
    output = BytesIO()
    Image.new("RGB", (2, 2), (index, 0, 0)).save(output, format="PNG")
    raw = output.getvalue()
    path = tmp_path / f"image_{index}.png"
    path.write_bytes(raw)
    return MediaRef(
        ref=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="image/png",
        observation_id=observation_id,
        frame_id=f"frame_{index}",
        role="scene_rgb",
    )


def _checkpoint(index: int, *, image_ref: MediaRef | None = None) -> StateCheckpoint:
    fragments = [ContextFragment("pose", {"index": index, "full_pose": [1, 2, 3]})]
    if image_ref is not None:
        fragments.append(ContextFragment("image", {}, (image_ref.to_model_input(),)))
    fragments.append(
        ContextFragment(
            "motion",
            {
                "instruction_start_relative_uav_motion": {"forward_m": index},
                "last_window_uav_motion": {"forward_m": 1},
                "field_semantics": "debug prose excluded from compactor",
                "observation_policy": {"mode": "automatic"},
            },
        )
    )
    observation_id = f"obs_{index:04d}"
    fragment_tuple = tuple(fragments)
    return StateCheckpoint(
        checkpoint_id=f"checkpoint_{index:04d}",
        reason="instruction_start" if index == 0 else "physical_window_completed",
        observation_id=observation_id,
        fragments=fragment_tuple,
        fingerprint=checkpoint_fingerprint(observation_id, fragment_tuple),
    )


def _transcript(tmp_path, windows: int = 6, *, with_images: bool = False) -> AgentTranscript:
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        "inspect the route",
        "instruction",
        InstructionEnvelope("instruction", "", {}, {}, (), ()),
    )
    first_image = _image_ref(tmp_path, 0, "obs_0000") if with_images else None
    transcript.append_state(_checkpoint(0, image_ref=first_image))
    _append_windows(transcript, 1, windows + 1, tmp_path=tmp_path, with_images=with_images)
    return transcript


def _append_windows(
    transcript: AgentTranscript,
    start: int,
    stop: int,
    *,
    tmp_path=None,
    with_images: bool = False,
) -> None:
    for index in range(start, stop):
        call_id = f"call_{index:04d}"
        transcript.append_assistant(
            AssistantTurn(
                turn_id=f"turn_{index:04d}",
                public_intent=f"Move window {index}",
                tool_calls=[
                    AgentToolCall(call_id, "uav.turn_relative", {"yaw_deg": 1})
                ],
            )
        )
        transcript.append_tool_result(
            ToolResult(
                call_id,
                "uav.turn_relative",
                "ok",
                "executed",
                data={"audit_details": {"large": "excluded"}, "pose": [1, 2, 3]},
                requested_args={"yaw_deg": 1},
                executed_args={"yaw_deg": 0.8},
                safety_interventions=[{"kind": "clip"}],
            )
        )
        image_ref = (
            _image_ref(tmp_path, index, f"obs_{index:04d}")
            if with_images
            else None
        )
        transcript.append_state(_checkpoint(index, image_ref=image_ref))


def _manager(
    compactor,
    *,
    keep: int = 2,
    prefetch: int = 2,
    soft: int = 3,
    max_compactor_input: int = 32_000,
    assembler: RequestAssembler | None = None,
) -> ContextWindowManager:
    config = HistoryCompressionConfig(
        prefetch_trigger_tokens=prefetch,
        soft_trigger_tokens=soft,
        target_tokens=1,
        max_compactor_input_tokens=max_compactor_input,
        summary_max_tokens=4_096,
        keep_recent_decision_windows=keep,
        max_request_images=20,
        keep_recent_visual_events=3,
    )
    projector = HistoryProjector(
        assembler=assembler,
        media_policy={
            "max_request_images": 20,
            "keep_recent_visual_events": 3,
            "pin_instruction_start": True,
            "pin_current": True,
        },
    )
    return ContextWindowManager(projector, config, compactor=compactor)


def _project(
    manager,
    transcript,
    state,
    coordinator,
    *,
    fixed_size: int = 0,
    current_task_state: dict | None = None,
):
    task_state = current_task_state or {
        "task_plan": {
            "revision": 4,
            "steps": [{
                "id": "inspect",
                "description": "Inspect the route",
                "status": "in_progress",
                "evidence_refs": ["obs_0001"],
            }],
        },
        "loaded_knowledge": [{"name": "flight", "version": "1"}],
    }
    return manager.project(
        transcript,
        default_registry(),
        state=state,
        coordinator=coordinator,
        fixed_messages=[{"role": "system", "content": "x" * fixed_size}],
        tools=[],
        parameters={"temperature": 0},
        current_task_state=task_state,
    )


def _wait_status(coordinator: HistoryCompactionCoordinator, status: str) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if coordinator.poll().status == status:
            return
        threading.Event().wait(0.005)
    raise AssertionError(f"coordinator did not reach {status}: {coordinator.poll()}")


def test_compression_config_has_large_window_and_retry_constraints():
    config = HistoryCompressionConfig.from_mapping(
        {
            "compression": {
                "prefetch_trigger_tokens": 48_000,
                "soft_trigger_tokens": 64_000,
                "target_tokens": 32_000,
                "max_compactor_input_tokens": 32_000,
                "summary_max_tokens": 4_096,
                "compactor_timeout_sec": 300,
                "compactor_timeout_retries": 0,
                "compactor_connection_retries": 1,
            }
        }
    )
    assert config.compactor_timeout_sec == 300
    assert config.compactor_timeout_retries == 0
    with pytest.raises(ValueError, match="target < prefetch <= soft"):
        HistoryCompressionConfig(
            target_tokens=48_000,
            prefetch_trigger_tokens=48_000,
        )


def test_prefetch_uses_all_eligible_windows_and_image_free_semantic_view(tmp_path):
    transcript = _transcript(tmp_path, with_images=True)
    before = transcript.to_dict()
    compactor = StubCompactor()
    manager = _manager(compactor, soft=1_000_000)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()

    result = _project(manager, transcript, state, coordinator)
    _wait_status(coordinator, "ready")

    assert state.section is None
    assert result.audit["compaction_status"] == "prefetch_started"
    assert len(compactor.requests) == 1
    request = compactor.requests[0]
    assert request.current_state_hint == {
        "task_plan": {
            "revision": 4,
            "steps": [{
                "id": "inspect",
                "description": "Inspect the route",
                "status": "in_progress",
                "evidence_refs": ["obs_0001"],
            }],
        },
        "loaded_knowledge": [{"name": "flight", "version": "1"}],
    }
    assert len(request.windows) == 4
    semantic = json.dumps(
        [window.to_dict() for window in request.windows], ensure_ascii=False
    )
    assert all(term not in semantic for term in ("model_inputs", "media_ref", "base64"))
    assert "large_repeated_plan" not in semantic
    assert "audit_details" not in semantic
    assert "full_pose" not in semantic
    assert "instruction_start_relative_uav_motion" in semantic
    assert transcript.to_dict() == before


def test_64k_pending_falls_back_immediately_then_late_result_upgrades(tmp_path):
    gate = threading.Event()
    compactor = StubCompactor(gate=gate)
    manager = _manager(compactor)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()
    transcript = _transcript(tmp_path)

    started = time.monotonic()
    result = _project(manager, transcript, state, coordinator)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert compactor.started.wait(1)
    assert coordinator.poll().status == "pending"
    assert state.section is not None and state.section.fallback_used
    assert result.audit["compaction_status"] == "fallback_pending"
    first_generation = state.capsule_generation

    gate.set()
    _wait_status(coordinator, "ready")
    upgraded = _project(manager, transcript, state, coordinator)

    assert state.capsule_generation == first_generation
    assert state.section is not None and not state.section.fallback_used
    assert upgraded.audit["compaction_status"] == "late_upgrade"
    assert coordinator.poll().status == "idle"


def test_failed_digest_is_not_retried_until_boundary_changes(tmp_path):
    compactor = StubCompactor(error=TimeoutError("injected"))
    manager = _manager(compactor)
    transcript = _transcript(tmp_path)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()

    _project(manager, transcript, state, coordinator)
    _wait_status(coordinator, "failed")
    _project(manager, transcript, state, coordinator)

    assert len(compactor.requests) == 1
    assert state.section is not None and state.section.fallback_used
    _append_windows(transcript, 7, 10)
    _project(manager, transcript, state, coordinator)
    assert compactor.started.wait(1)
    deadline = time.monotonic() + 1
    while len(compactor.requests) < 2 and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    assert len(compactor.requests) == 2


def test_invalid_late_model_refs_keep_deterministic_fallback(tmp_path):
    gate = threading.Event()
    compactor = StubCompactor(
        CompressionDraft(
            mission_progress=(
                {"summary": "Invented evidence.", "evidence_refs": ["missing_ref"]},
            )
        ),
        gate=gate,
    )
    manager = _manager(compactor)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()
    transcript = _transcript(tmp_path)

    _project(manager, transcript, state, coordinator)
    assert state.section is not None and state.section.fallback_used
    gate.set()
    _wait_status(coordinator, "ready")
    result = _project(manager, transcript, state, coordinator)

    assert state.section is not None and state.section.fallback_used
    assert result.audit["compaction_status"] == "late_result_invalid"
    assert result.audit["compaction_errors"] == [
        "mission_progress[0]_unknown_ref:missing_ref"
    ]
    assert coordinator.poll().status == "failed"


def test_32k_budget_selects_largest_oldest_complete_prefix(tmp_path):
    compactor = StubCompactor(tokens_per_window=100)
    manager = _manager(
        compactor,
        soft=1_000_000,
        max_compactor_input=250,
    )
    coordinator = HistoryCompactionCoordinator()

    _project(
        manager,
        _transcript(tmp_path, windows=8),
        HistoryProjectionState(),
        coordinator,
    )
    _wait_status(coordinator, "ready")

    assert len(compactor.requests[0].windows) == 2
    windows = compactor.requests[0].windows
    assert windows[0].start_event_index < windows[0].end_event_index
    assert windows[0].end_event_index <= windows[1].start_event_index


def test_single_oversized_window_uses_fallback_without_model_call(tmp_path):
    compactor = StubCompactor(tokens_per_window=1_000)
    manager = _manager(compactor, max_compactor_input=100)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()

    result = _project(manager, _transcript(tmp_path), state, coordinator)

    assert compactor.requests == []
    assert state.section is not None and state.section.fallback_used
    assert result.audit["activated_candidate"]["decision_window_count"] == 1
    assert result.audit["activated_candidate"]["estimated_input_tokens"] > 100


def test_compressed_prefix_images_are_omitted_but_current_image_remains(tmp_path):
    current_ref = _image_ref(tmp_path, 25, "obs_current")
    fragment = ContextFragment("image", {}, (current_ref.to_model_input(),))
    current = StateCheckpoint(
        checkpoint_id="checkpoint_current",
        reason="physical_window_completed",
        observation_id="obs_current",
        fragments=(fragment,),
        fingerprint=checkpoint_fingerprint("obs_current", (fragment,)),
    )
    section = CompressedHistorySection(
        generation=1,
        through_event_index=30,
        source_digest="digest",
        summary=CompressionDraft(
            mission_progress=(
                {"summary": "Old visual history was summarized.", "evidence_refs": []},
            )
        ),
        deterministic_facts={},
    )
    projector = HistoryProjector(
        assembler=RequestAssembler(MediaHydrator(base_dir=tmp_path)),
        media_policy={
            "max_request_images": 20,
            "keep_recent_visual_events": 3,
            "pin_instruction_start": True,
            "pin_current": True,
        },
    )
    envelope = InstructionEnvelope("instruction", "", {}, {}, (), ())
    result = projector.project_compacted(
        (
            TranscriptEvent("instruction_envelope", envelope),
            TranscriptEvent("user_instruction", {"instruction": "inspect"}),
        ),
        (TranscriptEvent("state_checkpoint", current),),
        section,
        default_registry(),
    )
    rendered = json.dumps(result.messages, ensure_ascii=False)

    assert result.audit["total_image_count"] == 1
    assert rendered.count("data:image/png;base64,") == 1
    capsule = next(
        message for message in result.messages if "HISTORY CAPSULE" in str(message)
    )
    assert isinstance(capsule["content"], str)
    assert '"schema_version": "compressed_history_v2"' in capsule["content"]
    assert '"through_event_index": 30' in capsule["content"]
    assert "open_work" not in capsule["content"]
    assert "CURRENT observation_id=obs_current" in rendered


def test_exact_knowledge_is_reinjected_outside_semantic_request(tmp_path):
    transcript = _transcript(tmp_path, windows=0)
    transcript.append_assistant(
        AssistantTurn(
            "knowledge_turn",
            "Load exact knowledge",
            [AgentToolCall("knowledge_call", "knowledge.load", {"name": "flight"})],
        )
    )
    body = "EXACT KNOWLEDGE BODY — do not summarize"
    transcript.append_tool_result(
        ToolResult(
            "knowledge_call",
            "knowledge.load",
            "ok",
            "loaded",
            data={"name": "flight", "body": body, "references": ["large"]},
        )
    )
    transcript.append_state(_checkpoint(1))
    _append_windows(transcript, 2, 7)
    gate = threading.Event()
    manager = _manager(StubCompactor(gate=gate))
    state = HistoryProjectionState()
    coordinator = HistoryCompactionCoordinator()

    current_task_state = {
        "task_plan": {"revision": 7, "steps": []},
        "loaded_knowledge": [{"name": "flight", "version": "loaded"}],
    }
    result = _project(
        manager,
        transcript,
        state,
        coordinator,
        current_task_state=current_task_state,
    )
    candidate = coordinator.poll().candidate
    assert candidate is not None
    assert candidate.request.current_state_hint == current_task_state
    assert body in json.dumps(candidate.exact_loaded_knowledge, ensure_ascii=False)
    assert body not in json.dumps(coordinator.poll().candidate.request.windows, default=str)
    rendered = json.dumps(result.messages, ensure_ascii=False)
    assert rendered.count(body) == 1
    gate.set()


def test_coordinator_abandon_discards_late_result(tmp_path):
    gate = threading.Event()
    compactor = StubCompactor(gate=gate)
    manager = _manager(compactor, soft=1_000_000)
    coordinator = HistoryCompactionCoordinator()
    _project(
        manager,
        _transcript(tmp_path),
        HistoryProjectionState(),
        coordinator,
    )
    assert compactor.started.wait(1)

    coordinator.abandon()
    gate.set()
    assert coordinator.poll().status == "abandoned"


def test_ready_result_from_old_generation_is_discarded(tmp_path):
    compactor = StubCompactor()
    manager = _manager(compactor, soft=1_000_000)
    coordinator = HistoryCompactionCoordinator()
    state = HistoryProjectionState()
    _project(manager, _transcript(tmp_path), state, coordinator)
    _wait_status(coordinator, "ready")

    state.capsule_generation = 1
    transition = manager._poll_completed_job(state, coordinator)

    assert transition["compaction_status"] == "stale_result_discarded"
    assert coordinator.poll().status == "idle"


def test_second_generation_replaces_first_capsule(tmp_path):
    gate = threading.Event()
    compactor = StubCompactor(gate=gate)
    manager = _manager(compactor)
    coordinator = HistoryCompactionCoordinator()
    transcript = _transcript(tmp_path)
    state = HistoryProjectionState()

    first = _project(manager, transcript, state, coordinator)
    first_boundary = state.compacted_through_event_index
    assert first.audit["compaction_status"] == "fallback_pending"
    gate.set()
    _wait_status(coordinator, "ready")
    _project(manager, transcript, state, coordinator)
    assert state.capsule_generation == 1

    _append_windows(transcript, 7, 10)
    second = _project(manager, transcript, state, coordinator)

    assert state.capsule_generation == 2
    assert state.compacted_through_event_index > first_boundary
    assert json.dumps(second.messages, ensure_ascii=False).count("HISTORY CAPSULE") == 1


def test_fallback_has_no_stale_window_count(tmp_path):
    gate = threading.Event()
    manager = _manager(StubCompactor(gate=gate))
    state = HistoryProjectionState()
    coordinator = HistoryCompactionCoordinator()

    _project(manager, _transcript(tmp_path), state, coordinator)

    assert state.section is not None
    text = json.dumps(state.section.summary.to_dict(), ensure_ascii=False)
    assert "decision windows" not in text
    gate.set()


def test_usage_calibrates_only_task_local_projection_state():
    manager = _manager(StubCompactor())
    first = HistoryProjectionState(last_estimated_tokens=100)
    second = HistoryProjectionState(last_estimated_tokens=100)

    manager.observe_usage(first, {"prompt_tokens": 200})

    assert first.last_actual_prompt_tokens == 200
    assert first.calibration_ratio > 1.0
    assert second.last_actual_prompt_tokens is None
    assert second.calibration_ratio == 1.0


def test_task_sessions_do_not_share_projection_or_coordinator_state():
    first = TaskSession(None, "a", None, None, None, None, None)
    second = TaskSession(None, "b", None, None, None, None, None)

    first.history_projection_state.capsule_generation = 3
    first.history_compaction_coordinator.abandon()

    assert second.history_projection_state.capsule_generation == 0
    assert first.history_projection_state is not second.history_projection_state
    assert first.history_compaction_coordinator is not second.history_compaction_coordinator
    assert second.history_compaction_coordinator.poll().status == "idle"


def test_model_compactor_uses_strict_unbounded_schema_and_no_tools():
    class Client:
        def __init__(self):
            self.requests = []

        def request(self, request, **kwargs):
            self.requests.append(request)
            return ModelResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "mission_progress": [
                                            {
                                                "summary": "Moved away from start.",
                                                "evidence_refs": ["E0001"],
                                            }
                                        ],
                                        "decisions_and_outcomes": [],
                                        "failures_and_recovery": [],
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10},
                }
            )

    client = Client()
    compactor = ModelHistoryCompactor(client, summary_max_tokens=4_096)
    request = CompressionRequest(
        source_digest="digest",
        instruction="inspect",
        windows=(CompressionWindow("w1", 2, 5, ()),),
        evidence_catalog=(EvidenceCatalogEntry("E0001", "obs_1", "observation"),),
        current_state_hint={"reason": "physical_window_completed"},
        deterministic_facts={},
    )

    result = compactor.compact(request)
    provider_request = client.requests[0]
    schema = provider_request.parameters["response_format"]["json_schema"]["schema"]
    encoded_schema = json.dumps(schema)

    assert result.draft.mission_progress[0]["evidence_refs"] == ["E0001"]
    assert provider_request.tools is None
    assert provider_request.parameters["temperature"] == 0
    assert provider_request.parameters["max_tokens"] == 4_096
    assert provider_request.parameters["thinking"] == {"type": "disabled"}
    assert "maxItems" not in encoded_schema and "maxLength" not in encoded_schema
    refs_schema = schema["properties"]["mission_progress"]["items"]["properties"][
        "evidence_refs"
    ]["items"]
    assert refs_schema["enum"] == ["E0001"]
    assert "visual_milestones" not in encoded_schema
    assert all(
        not isinstance(message["content"], list)
        for message in provider_request.messages
    )


def test_rejected_tool_call_uses_alias_and_normalizes_to_canonical_ref(tmp_path):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction(
        "inspect the route",
        "instruction",
        InstructionEnvelope("instruction", "", {}, {}, (), ()),
    )
    transcript.append_state(_checkpoint(0))
    transcript.append_assistant(
        AssistantTurn(
            "turn_0001",
            "Move once",
            [AgentToolCall("call_ok", "uav.turn_relative", {"yaw_deg": 1})],
        )
    )
    transcript.append_tool_result(
        ToolResult(
            "call_ok",
            "uav.turn_relative",
            "ok",
            "executed",
            artifact_refs=["private/artifact.png"],
        )
    )
    transcript.append_state(_checkpoint(1))
    rejected_id = "call_rejected_invalid_json"
    transcript.append_provider_rejection(
        {
            "error": "invalid JSON number 00.5",
            "content_present": True,
            "tool_calls": [
                {
                    "tool_call_id": rejected_id,
                    "tool_name": "uav.move_by_velocity_body_frame",
                    "args": {"vy": "00.5"},
                }
            ],
        }
    )
    _append_windows(transcript, 2, 7)
    before = transcript.to_dict()
    events = transcript.snapshot()
    projector = HistoryProjector()
    plan = projector.analyze(
        transcript.snapshot_range(2),
        base_event_index=2,
        keep_recent_decision_windows=2,
        all_events_for_evidence=events,
        registry=default_registry(),
    )
    rejected_entry = next(
        entry
        for entry in plan.evidence_catalog
        if entry.canonical_ref == rejected_id
    )
    assert rejected_entry.kind == "rejected_tool_call"

    class Client:
        def __init__(self):
            self.requests = []

        def request(self, request, **kwargs):
            self.requests.append(request)
            return ModelResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "mission_progress": [],
                                        "decisions_and_outcomes": [],
                                        "failures_and_recovery": [
                                            {
                                                "summary": (
                                                    f"Rejected call {rejected_entry.alias} "
                                                    "had invalid JSON arguments."
                                                ),
                                                "evidence_refs": [rejected_entry.alias],
                                            }
                                        ],
                                    }
                                )
                            },
                        }
                    ]
                }
            )

    request = CompressionRequest(
        source_digest=str(plan.source_digest),
        instruction=plan.instruction,
        windows=plan.windows,
        evidence_catalog=plan.evidence_catalog,
        current_state_hint={
            **plan.current_state_hint,
            "checkpoint_id": "checkpoint_current",
            "observation_id": "obs_current",
        },
        deterministic_facts=plan.deterministic_facts,
    )
    client = Client()
    draft = ModelHistoryCompactor(client, summary_max_tokens=4_096).compact(request).draft
    provider_request = client.requests[0]
    model_payload = provider_request.messages[1]["content"]
    schema = provider_request.parameters["response_format"]["json_schema"]["schema"]
    enum = schema["properties"]["failures_and_recovery"]["items"]["properties"][
        "evidence_refs"
    ]["items"]["enum"]

    assert rejected_entry.alias in model_payload
    assert rejected_id not in model_payload
    assert "checkpoint_" not in model_payload
    assert "window_id" not in model_payload
    assert "decision_window_000" not in model_payload
    assert "start_event_index" not in model_payload
    assert "private/artifact.png" not in model_payload
    assert "obs_current" not in model_payload
    assert enum == sorted(entry.alias for entry in plan.evidence_catalog)

    validation = CompressionValidator(4_096).validate(draft, request)
    assert validation.accepted
    assert validation.normalized_draft is not None
    normalized_item = validation.normalized_draft.failures_and_recovery[0]
    assert normalized_item["evidence_refs"] == [rejected_id]
    assert rejected_id in normalized_item["summary"]
    assert rejected_entry.alias not in json.dumps(
        validation.normalized_draft.to_dict()
    )
    assert transcript.to_dict() == before


def test_previous_capsule_refs_are_remapped_and_unknown_alias_is_rejected():
    canonical = "call_from_previous_generation"
    request = CompressionRequest(
        source_digest="digest",
        instruction="inspect",
        windows=(CompressionWindow("internal_window", 2, 5, ()),),
        evidence_catalog=(EvidenceCatalogEntry("E0001", canonical, "tool_call"),),
        current_state_hint={},
        deterministic_facts={},
        previous_summary=CompressionDraft(
            mission_progress=(
                {
                    "summary": f"The earlier action {canonical} made progress.",
                    "evidence_refs": [canonical],
                },
            )
        ),
    )

    class Client:
        def __init__(self):
            self.captured_request = None

        def request(self, model_request, **kwargs):
            self.captured_request = model_request
            return ModelResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "mission_progress": [
                                            {
                                                "summary": "Prior evidence E0001 remains relevant.",
                                                "evidence_refs": ["E0001"],
                                            }
                                        ],
                                        "decisions_and_outcomes": [],
                                        "failures_and_recovery": [],
                                    }
                                )
                            },
                        }
                    ]
                }
            )

    client = Client()
    draft = ModelHistoryCompactor(client, summary_max_tokens=4_096).compact(request).draft
    assert client.captured_request is not None
    payload = client.captured_request.messages[1]["content"]
    assert canonical not in payload
    assert payload.count("E0001") >= 2

    validator = CompressionValidator(4_096)
    accepted = validator.validate(draft, request)
    assert accepted.accepted and accepted.normalized_draft is not None
    assert canonical in json.dumps(accepted.normalized_draft.to_dict())

    rejected = validator.validate(
        CompressionDraft(
            decisions_and_outcomes=(
                {"summary": "Unknown old alias.", "evidence_refs": ["E9999"]},
            )
        ),
        request,
    )
    assert not rejected.accepted
    assert rejected.normalized_draft is None
    assert rejected.errors == ("decisions_and_outcomes[0]_unknown_ref:E9999",)


def test_model_compactor_rejects_length_finish_reason():
    class Client:
        def request(self, request, **kwargs):
            return ModelResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "{}"},
                        }
                    ]
                }
            )

    compactor = ModelHistoryCompactor(Client(), summary_max_tokens=4_096)
    request = CompressionRequest(
        "digest", "inspect", (CompressionWindow("w", 0, 1, ()),),
        (), {}, {},
    )
    with pytest.raises(ValueError, match="truncated"):
        compactor.compact(request)
