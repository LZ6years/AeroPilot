"""Interactive natural-language task session built on top of TaskRuntime.

This module is the boundary between UI shells and the flight runtime. CLI/TUI
frontends submit text and render summaries; this session owns scene reuse,
best-effort safety commands, and compact per-turn status data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agent.mission_state import MissionState
from src.runtime.configs import TaskExecutionOptions
from src.runtime.runners.task_runtime import TaskRuntime
from src.uav import pose_delta, pose_from_observation


class InteractiveTaskSession:
    """Keep one reusable scene alive while processing natural-language turns."""

    def __init__(
        self,
        runtime: TaskRuntime,
        map_id: str,
        *,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
        available_map_ids: list[str] | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        execution_options: TaskExecutionOptions | None = None,
    ) -> None:
        self.runtime = runtime
        self.recorder = runtime.recorder
        self.map_id = map_id
        self.agent_id = agent_id
        self.scene_id = scene_id
        self.available_map_ids = list(available_map_ids or [map_id])
        # Kept on the session so future TUI shells can inspect or replace the
        # live event consumer without reaching into TaskRuntime internals.
        self.event_sink = event_sink
        # A continuous session must not silently fall back to default
        # modalities after the first turn. The CLI/Benchmark composition root
        # resolves one run-level profile and stores its Runtime half here.
        self.execution_options = execution_options or TaskExecutionOptions()
        self.last_state: MissionState | None = None
        self.last_summary: dict[str, Any] | None = None
        self.closed = False

    def start_scene(self) -> dict[str, Any]:
        """Open or reuse the selected scene before accepting instructions."""

        if self.closed:
            return {"ok": False, "reason": "session_closed"}
        scene = self.runtime.ensure_reused_scene(self.map_id, self.scene_id)
        self.scene_id = scene.scene_id
        payload = {"ok": True, "map_id": self.map_id, "scene": scene.to_dict(), "run_dir": str(self.run_dir)}
        self.recorder.event("interactive_scene_ready", payload)
        return payload

    @property
    def run_dir(self) -> Path:
        return Path(self.recorder.run_dir)

    def submit_instruction(self, instruction: str) -> dict[str, Any]:
        """Run one natural-language instruction while reusing the active scene."""

        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise ValueError("instruction must not be empty")
        if self.closed:
            raise RuntimeError("interactive session is already closed")

        try:
            state = self.runtime.run_task(
                self.map_id,
                clean_instruction,
                agent_id=self.agent_id,
                scene_policy="reuse_running",
                stop_scene_on_finish=False,
                scene_id=self.scene_id,
                execution_options=self.execution_options,
            )
            self.last_state = state
            summary = build_turn_summary(
                self.run_dir,
                clean_instruction,
                state.to_dict(),
                agent_messages=self.runtime.latest_turn_agent_messages(),
            )
        except Exception as exc:
            self.recorder.event(
                "interactive_instruction_failed",
                {"instruction": clean_instruction, "error_type": type(exc).__name__, "error": str(exc)},
            )
            self.runtime.safe_hover_reused_scene(self.map_id, self.agent_id, self.scene_id)
            summary = build_error_summary(
                self.run_dir,
                clean_instruction,
                self.last_state,
                exc,
                agent_messages=self.runtime.latest_turn_agent_messages(),
            )

        self.last_summary = summary
        self.recorder.write_json("interactive/latest_summary.json", summary)
        self.recorder.event("interactive_instruction_completed", summary)
        return summary

    def hover(self) -> dict[str, Any]:
        """Ask the active reused scene to hover without ending the session."""

        if self.closed:
            return {"ok": False, "reason": "session_closed"}
        self.runtime.safe_hover_reused_scene(self.map_id, self.agent_id, self.scene_id)
        payload = {"ok": True, "command": "hover", "run_dir": str(self.run_dir)}
        self.recorder.event("interactive_hover_requested", payload)
        return payload

    def status(self) -> dict[str, Any]:
        state = self.last_state
        state_data = state.to_dict() if state is not None else {}
        latest_observation = state_data.get("latest_observation", {})
        return {
            "run_dir": str(self.run_dir),
            "map_id": self.map_id,
            "agent_id": self.agent_id,
            "scene_id": self.scene_id,
            "closed": self.closed,
            "has_state": state is not None,
            "completed": state_data.get("completed"),
            "connected": state_data.get("connected"),
            "airborne": state_data.get("airborne"),
            "task_plan": state.task_plan.to_dict() if state is not None and state.task_plan else None,
            "latest_pose": pose_from_observation(latest_observation),
            "last_summary": compact_turn_summary(self.last_summary) if self.last_summary else None,
        }

    def shutdown(self, shutdown_policy: str | None = None) -> dict[str, Any]:
        """Apply shutdown policy, stop the reused scene, and close the run logger."""

        if self.closed:
            return {"ok": True, "already_closed": True, "run_dir": str(self.run_dir)}
        old_policy = self.runtime.shutdown_policy
        if shutdown_policy is not None:
            self.runtime.shutdown_policy = shutdown_policy
        try:
            self.runtime.finish_reused_scene(self.map_id, self.last_state, self.agent_id, self.scene_id)
            payload = {
                "ok": True,
                "command": "shutdown",
                "shutdown_policy": self.runtime.shutdown_policy,
                "run_dir": str(self.run_dir),
            }
            self.recorder.event("interactive_session_shutdown", payload)
            return payload
        finally:
            self.runtime.shutdown_policy = old_policy
            self.recorder.close()
            self.closed = True


def build_turn_summary(
    run_dir: Path,
    instruction: str,
    state: dict[str, Any],
    *,
    agent_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    latest_observation = state.get("latest_observation", {})
    progress = state.get("motion_summary", {}) if isinstance(state.get("motion_summary", {}), dict) else {}
    initial_pose = state.get("instruction_origin_pose", {})
    final_pose = pose_from_observation(latest_observation)
    tool_facts = progress.get("tool_facts", {})
    events = load_events(run_dir / "events.jsonl")
    segment_events = latest_instruction_events(events)
    messages = list(agent_messages) if agent_messages is not None else agent_messages_from_events(segment_events)
    return {
        "instruction": instruction,
        "run_dir": str(run_dir),
        "completed": state.get("completed"),
        "connected": state.get("connected"),
        "airborne": state.get("airborne"),
        "tools": [{"tool_summary": tool_facts}] if tool_facts else [],
        "initial_pose": initial_pose,
        "final_pose": final_pose,
        "pose_delta": pose_delta(initial_pose, final_pose),
        "motion_summary": progress,
        "observation_summary": observation_summary([latest_observation] if latest_observation else []),
        "artifact_inventory": artifact_inventory_from_observations([latest_observation] if latest_observation else []),
        "agent_messages": agent_messages_for_summary(messages),
        "agent_message_count": len(messages),
        "agent_message_history_ref": "memory/agent_messages.jsonl",
    }


def build_error_summary(
    run_dir: Path,
    instruction: str,
    last_state: MissionState | None,
    exc: Exception,
    *,
    agent_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages = (
        list(agent_messages)
        if agent_messages is not None
        else agent_messages_from_events(latest_instruction_events(load_events(run_dir / "events.jsonl")))
    )
    return {
        "instruction": instruction,
        "run_dir": str(run_dir),
        "completed": False,
        "connected": last_state.connected if last_state is not None else None,
        "airborne": last_state.airborne if last_state is not None else None,
        "tools": [],
        "initial_pose": {},
        "final_pose": {},
        "pose_delta": pose_delta({}, {}),
        "motion_summary": {},
        "observation_summary": {"has_pose": False, "has_rgb": False, "has_depth": False, "sources": []},
        "artifact_inventory": {"rgb": [], "depth": [], "other": [], "counts": {"rgb": 0, "depth": 0, "other": 0}},
        "agent_messages": agent_messages_for_summary(messages),
        "agent_message_count": len(messages),
        "agent_message_history_ref": "memory/agent_messages.jsonl",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def compact_turn_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    compact = {
        "instruction": summary.get("instruction"),
        "completed": summary.get("completed"),
        "airborne": summary.get("airborne"),
        "tools": summary.get("tools", []),
        "pose_delta": summary.get("pose_delta", {}),
        "motion_summary": summary.get("motion_summary", {}),
        "observation_summary": summary.get("observation_summary", {}),
        "artifact_counts": summary.get("artifact_inventory", {}).get("counts", {}),
        "agent_messages": summary.get("agent_messages", []),
        "agent_message_count": summary.get("agent_message_count"),
        "agent_message_history_ref": summary.get("agent_message_history_ref"),
        "run_dir": summary.get("run_dir"),
    }
    if "error" in summary:
        compact["error_type"] = summary.get("error_type")
        compact["error"] = summary.get("error")
    return compact


def agent_messages_for_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep interactive summaries display-sized while preserving durable history.

    ``memory/agent_messages.jsonl`` remains the append-only source of truth.
    The interactive summary is a UI artifact, so it should behave like the
    latest reply snapshot: prefer the final answer, otherwise show the newest
    available message.
    """

    normalized = [dict(message) for message in messages if isinstance(message, dict)]
    for message in reversed(normalized):
        if message.get("type") == "final":
            return [message]
    return normalized[-1:] if normalized else []


def observation_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "has_pose": any(bool(obs.get("pose")) for obs in observations if isinstance(obs, dict)),
        "has_rgb": any("rgb" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "has_depth": any("depth" in dict(obs.get("frames", {})) for obs in observations if isinstance(obs, dict)),
        "sources": [str(obs.get("source", "")) for obs in observations if isinstance(obs, dict) and obs.get("source")],
    }


def artifact_inventory_from_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    inventory: dict[str, list[dict[str, Any]]] = {"rgb": [], "depth": [], "other": []}
    for obs_index, observation in enumerate(observations):
        frames = observation.get("frames", {}) if isinstance(observation, dict) else {}
        if not isinstance(frames, dict):
            continue
        for modality, metadata in frames.items():
            if not isinstance(metadata, dict) or "artifact_path" not in metadata:
                continue
            normalized = "rgb" if str(modality).lower() in {"rgb", "scene", "color"} else str(modality).lower()
            bucket = normalized if normalized in {"rgb", "depth"} else "other"
            path = Path(str(metadata["artifact_path"]))
            inventory[bucket].append(
                {
                    "observation_index": obs_index,
                    "modality": str(modality),
                    "path": str(path),
                    "exists": path.exists(),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                }
            )
    return {**inventory, "counts": {name: len(items) for name, items in inventory.items()}}


def agent_messages_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract public agent messages from an instruction-scoped event segment."""

    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "agent_message":
            continue
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        message = payload.get("message", {}) if isinstance(payload.get("message", {}), dict) else {}
        if message:
            messages.append(
                {
                    "source": payload.get("source"),
                    "phase": payload.get("phase"),
                    **message,
                }
            )
    return messages


def latest_instruction_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return events for the latest instruction in an interactive run."""

    ordered = sorted(events, key=lambda item: str(item.get("timestamp", "")))
    start_index = 0
    for index, event in enumerate(ordered):
        if event.get("event") == "instruction_started":
            start_index = index
    return ordered[start_index:]


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import json

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events

