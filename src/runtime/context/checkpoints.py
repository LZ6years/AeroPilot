"""Publish Model Context V3 at explicit sensory state boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.agent.transcript import AgentTranscript
from src.memory.context.builder import ModelContextBuilder
from src.memory.context.contracts import (
    CHECKPOINT_REASONS,
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    StateCheckpoint,
)
from src.memory.flight_memory import FlightMemory


class UnsupportedRunSchema(ValueError):
    """Raised instead of guessing how to interpret pre-V3 artifacts."""


class ContextCheckpointService:
    """Single owner of checkpoint timing, deduplication and persistence."""

    def __init__(
        self,
        builder: ModelContextBuilder,
        transcript: AgentTranscript,
        recorder: Any,
    ) -> None:
        self.builder = builder
        self.transcript = transcript
        self.recorder = recorder
        self._counter = 0
        self._last_fingerprint: str | None = None

    def begin_instruction(
        self, state: MissionState, memory: FlightMemory
    ) -> None:
        self._counter = 0
        self._last_fingerprint = None
        envelope = self.builder.build_instruction_envelope(state, memory)
        self.transcript.start_instruction(
            state.user_instruction, memory.active_instruction_id, envelope
        )
        self.recorder.write_json(
            "contexts/instruction_envelope.json", envelope.to_dict()
        )

    def publish(
        self,
        state: MissionState,
        memory: FlightMemory,
        reason: str,
    ) -> StateCheckpoint | None:
        if reason not in CHECKPOINT_REASONS:
            raise ValueError(f"unsupported checkpoint reason: {reason}")
        checkpoint_id = f"checkpoint_{self._counter + 1:04d}"
        checkpoint, fragment_audits = self.builder.build_checkpoint(
            state,
            memory,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )
        if checkpoint.fingerprint == self._last_fingerprint:
            return None
        self._counter += 1
        self._last_fingerprint = checkpoint.fingerprint
        self.transcript.append_state(checkpoint)
        payload = checkpoint.to_dict()
        self.recorder.write_json("contexts/latest_checkpoint.json", payload)
        self.recorder.write_json(
            f"debug/context_checkpoints/{checkpoint.checkpoint_id}.json",
            {
                "schema_version": "context_checkpoint_audit_v3",
                "checkpoint": payload,
                "fragment_audits": list(fragment_audits),
            },
        )
        return checkpoint


def load_v3_json(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    """Read a V3 artifact and reject missing or legacy schema markers."""

    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
        raise UnsupportedRunSchema(
            f"unsupported run schema in {artifact}: expected {expected_schema}"
        )
    return payload


def load_instruction_envelope(path: str | Path) -> dict[str, Any]:
    return load_v3_json(path, expected_schema=INSTRUCTION_ENVELOPE_SCHEMA)


def load_state_checkpoint(path: str | Path) -> dict[str, Any]:
    return load_v3_json(path, expected_schema=STATE_CHECKPOINT_SCHEMA)
