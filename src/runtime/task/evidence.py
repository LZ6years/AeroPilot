"""Synchronize task state, flight-memory references, and durable evidence."""

from __future__ import annotations

import time
from typing import Any

from src.perception.raw_observation import RawObservation
from src.runtime.context.observation_media import ObservationMediaStore
from src.runtime.task.session import TaskSession


class FlightEvidenceRecorder:
    """Single write boundary for action, observation, and task-state evidence."""

    def __init__(self, recorder: Any, tools: Any, memory_writer: Any, journal: Any) -> None:
        self.recorder = recorder
        self.tools = tools
        self.memory_writer = memory_writer
        self.journal = journal
        self.observation_media = ObservationMediaStore(recorder.run_dir)

    def record_action(
        self,
        session: TaskSession,
        tool: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        session.state.record_action(tool, args, result)
        started = time.monotonic()
        try:
            write_result = self.memory_writer.write_action(tool, args, result)
        finally:
            self.journal.debug_event(
                "artifact_write_timing",
                {"kind": "action", "elapsed_sec": time.monotonic() - started},
            )
        session.memory.append_action_ref(write_result["action_ref"])
        session.memory.append_pose_ref(write_result.get("pose_ref"))
        return write_result

    def record_observation(
        self,
        session: TaskSession,
        modalities: list[str],
        source: str,
        *,
        overwrite_task_initial: bool = False,
    ) -> dict[str, Any]:
        observation = self.tools.observe(
            session.scene.scene_instance_id,
            session.agent_id,
            modalities,
        )
        raw = RawObservation.from_env_observation(observation, source=source)
        data = raw.to_dict()
        started = time.monotonic()
        try:
            # Scene-owned capture storage can be shared by multiple Runtime
            # runs. Import originals before Memory or a checkpoint commits a
            # path that its run-relative media consumers cannot address.
            data["frames"] = self.observation_media.materialize_frames(data["frames"])
            write_result = self.memory_writer.write_observation(data)
        finally:
            self.journal.debug_event(
                "artifact_write_timing",
                {"kind": "observation", "elapsed_sec": time.monotonic() - started},
            )
        observation_id = write_result["observation_ref"]["id"]
        data["observation_id"] = observation_id
        frames = data.get("frames")
        if isinstance(frames, dict):
            for frame_ref in write_result.get("frame_refs", []):
                modality = frame_ref.get("modality")
                frame = frames.get(modality)
                if isinstance(frame, dict):
                    frame["id"] = frame_ref.get("id")
                    frame["frame_id"] = frame_ref.get("id")
                    frame["artifact_ref"] = frame_ref.get("ref")
                    frame["observation_id"] = observation_id
        session.state.record_observation(data)
        session.memory.append_observation_ref(write_result["observation_ref"])
        session.memory.append_pose_ref(write_result.get("pose_ref"))
        session.memory.append_frame_refs(write_result.get("frame_refs"))
        if source in {"preflight", "instruction_start"}:
            session.memory.set_observation_anchor(
                "task_initial",
                write_result["observation_ref"],
                overwrite=True,
            )
        self.recorder.write_json("debug/latest_observation.json", data)
        return data

    def mark_before_last_action_window(
        self,
        session: TaskSession,
        observation_before: dict[str, Any],
    ) -> None:
        """Anchor the observation shared by one ordered physical window."""

        memory = session.memory
        if not memory.recent_observation_refs:
            return
        expected_id = observation_before.get("observation_id")
        latest_ref = memory.recent_observation_refs[-1]
        if expected_id and latest_ref.get("id") != expected_id:
            matching = next(
                (
                    ref
                    for ref in reversed(memory.recent_observation_refs)
                    if ref.get("id") == expected_id
                ),
                None,
            )
            if matching is None:
                raise RuntimeError(
                    "cannot anchor before_last_action_window: observation is not in active memory"
                )
            latest_ref = matching
        memory.set_observation_anchor(
            "before_last_action_window", latest_ref, overwrite=True
        )

    def write_tool_completion(
        self,
        session: TaskSession,
        *,
        decision_id: str,
        call_index: int,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.memory_writer.write_tool_completion(
            instruction_id=session.transcript.instruction_id,
            decision_id=decision_id,
            call_index=call_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=args,
            result=result,
        )

    def write_state(self, session: TaskSession) -> None:
        self.journal.write_state(
            session.state,
            session.memory,
            self.state_summary(session.state),
        )

    @staticmethod
    def state_summary(state: Any) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "completed": state.completed,
            "stopped": state.stopped,
            "stopped_reason": state.stopped_reason,
            "stop_source": state.stop_source,
            "connected": state.connected,
            "airborne": state.airborne,
            "physical_window_count": state.physical_window_count,
            "plan_revision": state.task_plan.revision if state.task_plan else None,
        }

    @staticmethod
    def observation_summary(
        observation: dict[str, Any], source: str | None = None
    ) -> dict[str, Any]:
        frames = observation.get("frames") if isinstance(observation, dict) else {}
        frame_refs: dict[str, str] = {}
        if isinstance(frames, dict):
            for modality, frame in frames.items():
                if isinstance(frame, dict):
                    artifact_path = frame.get("artifact_path") or frame.get("path")
                    if artifact_path:
                        frame_refs[str(modality)] = str(artifact_path)
        return {
            "source": source or observation.get("source"),
            "observation_id": observation.get("observation_id"),
            "pose": dict(observation.get("pose") or {}),
            "timestamp": observation.get("timestamp"),
            "scene_instance_id": observation.get("scene_instance_id"),
            "agent_id": observation.get("agent_id"),
            "modalities": list(frames.keys()) if isinstance(frames, dict) else [],
            "has_pose": bool(observation.get("pose")),
            "frame_artifacts": frame_refs,
        }
