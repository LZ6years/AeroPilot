"""Durable Runtime event, state and provider-exchange journal ownership."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from typing import Any

from src.llm.audit import redact_provider_value


PUBLIC_RUNTIME_EVENTS = frozenset(
    {
        "instruction_started",
        "model_turn_recorded",
        "task_control_recorded",
        "read_only_turn_recorded",
        "plan_read_only_turn_recorded",
        "physical_window_recorded",
        "task_terminal",
        "instruction_failed",
        "scene_cleanup_failed",
        "tool_step_completed",
        "observation_updated",
        "runtime_loop_guard_reached",
    }
)


class RuntimeJournal:
    """Own Runtime persistence and best-effort live event delivery.

    This service intentionally knows nothing about planning or context
    construction. It receives already-decided events and snapshots from the
    facade, making their durability policy independently testable.
    """

    def __init__(
        self,
        recorder: Any,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.recorder = recorder
        self.event_sink = event_sink
        self._persisted_provider_media: set[str] = set()

    def reset_instruction(self) -> None:
        self._persisted_provider_media.clear()

    def write_state(
        self,
        state: Any,
        flight_memory: Any,
        summary: dict[str, Any],
    ) -> None:
        self.recorder.write_json("state/latest_state.json", state.to_dict())
        self.recorder.write_json("state/flight_memory.json", flight_memory.to_dict())
        self.debug_event("mission_state_updated", summary)

    def event(self, event: str, payload: dict[str, Any]) -> None:
        self.recorder.event(event, payload)
        if event in PUBLIC_RUNTIME_EVENTS:
            self.emit_live({"kind": event, "payload": dict(payload)})

    def write_turn(self, event: str, payload: dict[str, Any]) -> None:
        """Persist a provider-turn decision or rejection."""

        if event not in {
            "model_turn_recorded",
            "tool_turn_rejected",
            "task_control_recorded",
            "read_only_turn_recorded",
            "plan_read_only_turn_recorded",
        }:
            raise ValueError(f"Unsupported turn event: {event}")
        self.event(event, payload)

    def write_window(self, payload: dict[str, Any]) -> None:
        """Persist one complete or failed physical-window envelope."""

        self.event("physical_window_recorded", payload)

    def write_terminal(self, payload: dict[str, Any]) -> None:
        """Persist the instruction terminal envelope."""

        self.event("task_terminal", payload)

    def debug_event(self, event: str, payload: dict[str, Any]) -> None:
        debug_event = getattr(self.recorder, "debug_event", None)
        if callable(debug_event):
            debug_event(event, payload)
        else:
            self.recorder.event(event, payload)

    def emit_live(self, event: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            self.event_sink(dict(event))
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.debug_event(
                "runtime_event_sink_failed",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "event_kind": event.get("kind"),
                },
            )

    def persist_provider_exchange_media(
        self, model_rounds: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Write request media once per digest and return base64-free rounds."""

        persisted = copy.deepcopy(model_rounds)
        save_bytes = getattr(self.recorder, "save_bytes", None)
        for model_round in persisted:
            media = model_round.get("request_media")
            if not isinstance(media, list):
                continue
            for item in media:
                if not isinstance(item, dict):
                    continue
                data = item.pop("data", None)
                digest = str(item.get("sha256") or "").strip().lower()
                suffix = str(item.get("suffix") or ".bin")
                if isinstance(data, bytes) and digest and callable(save_bytes):
                    relative = f"debug/provider_media/{digest}{suffix}"
                    if digest not in self._persisted_provider_media:
                        save_bytes(relative, data)
                        self._persisted_provider_media.add(digest)
                    item["asset_ref"] = relative
                elif data is not None:
                    item["unavailable_reason"] = "recorder_cannot_persist_media"
        return persisted

    def capture_provider_exchange(
        self,
        planner: Any,
        session: Any,
        decision_id: str,
        *,
        turn: Any | None = None,
        error_type: str | None = None,
    ) -> None:
        """Persist the exact provider exchange and its canonical Runtime turn."""

        record = self._provider_round_snapshot(
            planner,
            session.decision_index,
            canonical_turn=turn,
        )
        record["decision_id"] = decision_id
        if error_type:
            record["error_type"] = error_type
        record = self.persist_provider_exchange_media([record])[0]
        instruction_id = session.transcript.instruction_id
        namespace = hashlib.sha256(
            str(instruction_id).encode("utf-8")
        ).hexdigest()[:16]
        debug_ref = f"debug/tool_loop/{namespace}/{decision_id}.json"
        checkpoint = session.transcript.latest_checkpoint()
        exchange = getattr(planner, "last_exchange", None)
        current_task_state = (
            exchange.get("current_task_state")
            if isinstance(exchange, dict)
            else {}
        )
        self.recorder.write_json(
            debug_ref,
            {
                "loop_schema": "unified_tool_loop_v2",
                "instruction_id": instruction_id,
                "decision_id": decision_id,
                "decision_index": session.decision_index,
                "checkpoint_observation_id": session.transcript.latest_observation_id(),
                "model_context": redact_provider_value(
                    checkpoint.to_dict() if checkpoint is not None else {}
                ),
                "current_task_state": redact_provider_value(
                    current_task_state
                    if isinstance(current_task_state, dict)
                    else {}
                ),
                "model_rounds": [record],
            },
        )
        self.write_turn(
            "model_turn_recorded",
            {
                "decision_id": decision_id,
                "debug_ref": debug_ref,
                "instruction_id": instruction_id,
                "loop_schema": "unified_tool_loop_v2",
            },
        )

    @staticmethod
    def _provider_round_snapshot(
        planner: Any,
        round_index: int,
        *,
        canonical_turn: Any | None = None,
    ) -> dict[str, Any]:
        exchange = getattr(planner, "last_exchange", None)
        exchange = copy.deepcopy(exchange) if isinstance(exchange, dict) else {}
        record: dict[str, Any] = {
            "round_index": round_index,
            "request": exchange.get("request"),
            "response": exchange.get("response"),
        }
        projection = exchange.get("history_projection")
        if isinstance(projection, dict):
            record["history_projection"] = projection
        current_task_state = exchange.get("current_task_state")
        if isinstance(current_task_state, dict):
            record["current_task_state"] = current_task_state
        transport = exchange.get("transport")
        if isinstance(transport, dict):
            record["transport"] = transport
        media = exchange.get("media")
        if isinstance(media, list):
            record["request_media"] = media
        if canonical_turn is not None:
            record["canonical_turn"] = canonical_turn.to_dict()
        elif not exchange:
            record["capture_status"] = "provider_exchange_unavailable"
        return record
