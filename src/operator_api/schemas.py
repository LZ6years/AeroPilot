"""HTTP and WebSocket contracts for the operator UI, separate from EnvClient v1."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(RequestModel):
    map_id: str = Field(min_length=1, max_length=200)
    execution: Literal["managed", "attach"] = "managed"
    preview_width: int = Field(default=960, ge=64, le=3840)
    preview_height: int = Field(default=540, ge=64, le=2160)
    preview_fps: float = Field(default=20, ge=1, le=60)


class SubmitInstruction(RequestModel):
    instruction: str = Field(min_length=1, max_length=10000)
    request_id: str = Field(min_length=1, max_length=128)

    @field_validator("instruction")
    @classmethod
    def clean_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value.strip()


class ControlRequest(RequestModel):
    action: Literal["pause", "resume", "stop"]


class UiEvent(BaseModel):
    """IDs retain their source identity; seq orders delivery within one session."""

    schema_version: Literal["operator_event_v1"] = "operator_event_v1"
    seq: int
    timestamp: str
    session_id: str | None = None
    instruction_id: str | None = None
    decision_id: str | None = None
    kind: str
    payload: dict[str, Any]
