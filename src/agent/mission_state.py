"""Track the current mission control state for short-horizon runtime loops.

MissionState intentionally stores the latest runtime state, not flight history.
Historical actions, observations, poses, and frames live in FlightMemory JSONL
records and are summarized for models by context modules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.uav import pose_measurements, yaw_deg
from src.agent.semantic.schemas import SemanticPlan


@dataclass
class MissionState:
    run_id: str
    user_instruction: str
    semantic_plan: SemanticPlan
    current_phase_index: int = 0
    latest_observation: dict[str, Any] = field(default_factory=dict)
    latest_pose: dict[str, Any] = field(default_factory=dict)
    current_phase_progress: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    connected: bool = False
    airborne: bool = False
    hover_required: bool = False
    yaw_scan_accum_deg: float = 0.0
    current_phase_window_count_value: int = 0
    max_phase_window_count: int = 5
    completed: bool = False
    stopped: bool = False
    stopped_reason: str | None = None
    stop_source: str | None = None

    @property
    def current_phase_name(self) -> str:
        if self.current_phase_index >= len(self.semantic_plan.phases):
            return "completed"
        return self.semantic_plan.phases[self.current_phase_index].name

    @property
    def current_phase_key(self) -> str:
        if self.current_phase_index >= len(self.semantic_plan.phases):
            return "completed"
        phase = self.semantic_plan.phases[self.current_phase_index]
        return f"{phase.index}:{phase.name}"

    @property
    def is_final_phase(self) -> bool:
        """Return whether the active phase is the last semantic phase.

        Runtime completion is semantic: a model may request ``complete_task``
        only when the final phase is active. Earlier phases can be complete,
        but that should advance the mission rather than end the whole task.
        """

        phase_count = len(self.semantic_plan.phases)
        if phase_count == 0:
            return True
        return self.current_phase_index >= phase_count - 1

    def advance_phase(self) -> None:
        # Move to the next semantic phase after one short window has completed.
        self.current_phase_index += 1
        if self.current_phase_index >= len(self.semantic_plan.phases):
            self.completed = True
            return
        self.current_phase_window_count_value = 0
        self.current_phase_progress = {}

    @property
    def current_phase_window_count(self) -> int:
        return int(self.current_phase_window_count_value)

    def record_phase_window_started(self) -> None:
        # Count windows per semantic phase so runtime can prevent infinite loops.
        self.current_phase_window_count_value += 1

    def ensure_current_phase_progress(self, start_observation: dict[str, Any]) -> None:
        # Capture only the current phase measurement origin; older phase details live in events/jsonl.
        if self.current_phase_progress.get("phase_key") == self.current_phase_key:
            return
        start_pose = _pose_from_observation(start_observation)
        clean_start_pose = _clean_pose(start_pose)
        self.current_phase_progress = {
            "phase_key": self.current_phase_key,
            "phase": self.current_phase_name,
            "coordinate_frames": _coordinate_frames(),
            "world_pose": {
                "start": clean_start_pose,
                "latest": dict(clean_start_pose),
            },
            "phase_local_frame": {
                "origin": "current phase start pose",
                "origin_yaw_deg": round(yaw_deg(start_pose), 3),
                "description": "Used to project world dx/dy into phase-local forward/right displacement.",
            },
            "window_count": self.current_phase_window_count,
            "max_phase_window_count": self.max_phase_window_count,
            "last_window_measurements": {},
            "phase_local_measurements": pose_measurements(start_pose, start_pose),
            "cumulative_measurements": pose_measurements(start_pose, start_pose),
            "cumulative_tool_facts": _empty_tool_facts(),
            "note": "Current-phase facts only. Use phase_local_measurements.forward_displacement_m for forward movement progress; history is stored in memory/events.",
        }

    def update_current_phase_progress(
        self,
        latest_observation: dict[str, Any],
        last_window_measurements: dict[str, Any] | None = None,
        latest_window_measurements: dict[str, Any] | None = None,
        tool_facts_delta: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_current_phase_progress(self.latest_observation or latest_observation)
        latest_pose = _pose_from_observation(latest_observation)
        progress = copy.deepcopy(self.current_phase_progress)
        start_pose = dict(progress.get("world_pose", {}).get("start", {}))
        clean_latest_pose = _clean_pose(latest_pose)
        progress["world_pose"] = {
            "start": _clean_pose(start_pose),
            "latest": clean_latest_pose,
        }
        progress["window_count"] = self.current_phase_window_count
        progress["max_phase_window_count"] = self.max_phase_window_count
        window_measurements = last_window_measurements
        if window_measurements is None:
            window_measurements = latest_window_measurements
        progress["last_window_measurements"] = dict(window_measurements or progress.get("last_window_measurements", {}))
        progress.pop("latest_window_measurements", None)
        progress.pop("start_pose", None)
        progress.pop("latest_pose", None)
        if "phase_local_frame" not in progress:
            progress["phase_local_frame"] = {
                "origin": "current phase start pose",
                "origin_yaw_deg": round(yaw_deg(start_pose), 3),
                "description": "Used to project world dx/dy into phase-local forward/right displacement.",
            }
        progress.pop("phase_origin_yaw_deg", None)
        measurements = pose_measurements(start_pose, latest_pose)
        progress["phase_local_measurements"] = measurements
        progress["cumulative_measurements"] = measurements
        if tool_facts_delta:
            progress["cumulative_tool_facts"] = _merge_tool_facts(
                progress.get("cumulative_tool_facts", {}),
                tool_facts_delta,
            )
        self.current_phase_progress = progress

    @property
    def current_phase_execution_summary(self) -> dict[str, Any]:
        # PlannerContext keeps the public field name phase_execution_summary; MissionState stores only progress.
        return copy.deepcopy(self.current_phase_progress)

    def mark_completed(self) -> None:
        self.completed = True

    def mark_stopped(self, reason: str, source: str = "runtime") -> None:
        # Stopped means the runtime hit an abnormal guard. It is intentionally
        # separate from completed so eval/reporting can treat the task as
        # incomplete instead of semantic success.
        self.stopped = True
        self.stopped_reason = str(reason)
        self.stop_source = str(source)

    def record_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        # Update only live control flags; action history is written to memory/actions.jsonl.
        if result.get("ok") is True:
            if tool == "uav.connect":
                self.connected = True
            if tool == "uav.takeoff":
                self.airborne = True
            if tool == "uav.land":
                self.airborne = False
            if tool == "uav.turn_relative":
                self.yaw_scan_accum_deg += abs(float(args.get("yaw_deg", 0.0)))

    def record_observation(self, observation: dict[str, Any]) -> None:
        # Keep only the latest observation snapshot; history is written to memory/observations.jsonl.
        self.latest_observation = _clean_observation(observation)
        self.latest_pose = _clean_pose(_pose_from_observation(observation))

    def record_error(self, error_type: str, message: str) -> None:
        self.errors.append({"error_type": error_type, "message": message})
        self.hover_required = True

    def to_dict(self) -> dict[str, Any]:
        yaw_scan = {
            "accum_deg": self.yaw_scan_accum_deg,
        }
        return {
            "run_id": self.run_id,
            "user_instruction": self.user_instruction,
            "semantic_plan": self.semantic_plan.to_dict(),
            "current_phase_index": self.current_phase_index,
            "current_phase": self.current_phase_name,
            "current_phase_key": self.current_phase_key,
            "latest_observation": copy.deepcopy(self.latest_observation),
            "errors": list(self.errors),
            "safety": {"hover_required": self.hover_required, "yaw_scan": yaw_scan},
            "current_phase_progress": copy.deepcopy(self.current_phase_progress),
            "current_phase_window_count": self.current_phase_window_count,
            "max_phase_window_count": self.max_phase_window_count,
            "connected": self.connected,
            "airborne": self.airborne,
            "completed": self.completed,
            "stopped": self.stopped,
            "stopped_reason": self.stopped_reason,
            "stop_source": self.stop_source,
        }


def _pose_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    pose = observation.get("pose") if isinstance(observation, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def _clean_pose(pose: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(pose) if isinstance(pose, dict) else {}
    data.pop("velocity", None)
    return data


def _clean_observation(observation: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(observation) if isinstance(observation, dict) else {}
    if isinstance(data.get("pose"), dict):
        data["pose"] = _clean_pose(data["pose"])
    return data


def _coordinate_frames() -> dict[str, Any]:
    return {
        "world_ned": {
            "description": "AirSim/UE world frame. x/y are horizontal world axes; z is down-positive NED.",
            "source": "raw AirSim pose",
        },
        "mission_local": {
            "description": "Instruction-start local frame. Origin is the UAV pose at instruction start; initial yaw is 0; forward/right/up are for user movement semantics.",
        },
        "phase_local": {
            "description": "Current-phase local frame. Origin and yaw are captured at this phase start; use forward/right/up displacement to judge this phase.",
        },
    }


def _empty_tool_facts() -> dict[str, Any]:
    return {
        "total_steps": 0,
        "successful_steps": 0,
        "failed_steps": 0,
        "turn_relative_yaw_deg_attempted_sum": 0.0,
        "turn_relative_yaw_deg_success_sum": 0.0,
        "move_body_duration_attempted_sum": 0.0,
        "move_body_duration_success_sum": 0.0,
        "move_body_duration_by_direction_success": {},
        "tools_executed": {},
    }


def _merge_tool_facts(current: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = _empty_tool_facts()
    merged.update(copy.deepcopy(current) if isinstance(current, dict) else {})
    for key in (
        "total_steps",
        "successful_steps",
        "failed_steps",
        "turn_relative_yaw_deg_attempted_sum",
        "turn_relative_yaw_deg_success_sum",
        "move_body_duration_attempted_sum",
        "move_body_duration_success_sum",
    ):
        merged[key] = round(float(merged.get(key, 0.0) or 0.0) + float(delta.get(key, 0.0) or 0.0), 3)
        if key.endswith("_steps"):
            merged[key] = int(merged[key])
    directions = dict(merged.get("move_body_duration_by_direction_success", {}))
    for direction, duration in dict(delta.get("move_body_duration_by_direction_success", {})).items():
        directions[str(direction)] = round(float(directions.get(str(direction), 0.0) or 0.0) + float(duration or 0.0), 3)
    merged["move_body_duration_by_direction_success"] = directions
    tools = dict(merged.get("tools_executed", {}))
    for tool, count in dict(delta.get("tools_executed", {})).items():
        tools[str(tool)] = int(tools.get(str(tool), 0) or 0) + int(count or 0)
    merged["tools_executed"] = tools
    return merged
