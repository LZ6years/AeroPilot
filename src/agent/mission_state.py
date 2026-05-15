"""Track the current mission control state for short-horizon runtime loops.

MissionState intentionally stores the latest runtime state, not flight history.
Historical actions, observations, poses, and frames live in FlightMemory JSONL
records and are summarized for models by context modules.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

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
            "start_pose": clean_start_pose,
            "latest_pose": dict(clean_start_pose),
            "phase_origin_yaw_deg": round(_yaw_deg(start_pose), 3),
            "window_count": self.current_phase_window_count,
            "max_phase_window_count": self.max_phase_window_count,
            "latest_window_measurements": {},
            "phase_local_measurements": _pose_measurements(start_pose, start_pose),
            "cumulative_measurements": _pose_measurements(start_pose, start_pose),
            "cumulative_tool_facts": _empty_tool_facts(),
            "note": "Current-phase facts only. Use phase_local_measurements.forward_displacement_m for forward movement progress; history is stored in memory/events.",
        }

    def update_current_phase_progress(
        self,
        latest_observation: dict[str, Any],
        latest_window_measurements: dict[str, Any] | None = None,
        tool_facts_delta: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_current_phase_progress(self.latest_observation or latest_observation)
        latest_pose = _pose_from_observation(latest_observation)
        progress = copy.deepcopy(self.current_phase_progress)
        start_pose = dict(progress.get("world_pose", {}).get("start", progress.get("start_pose", {})))
        clean_latest_pose = _clean_pose(latest_pose)
        progress["latest_pose"] = clean_latest_pose
        progress["world_pose"] = {
            "start": _clean_pose(start_pose),
            "latest": clean_latest_pose,
        }
        progress["window_count"] = self.current_phase_window_count
        progress["max_phase_window_count"] = self.max_phase_window_count
        progress["latest_window_measurements"] = dict(latest_window_measurements or progress.get("latest_window_measurements", {}))
        measurements = _pose_measurements(start_pose, latest_pose)
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
        self.latest_observation = copy.deepcopy(observation)
        self.latest_pose = _clean_pose(_pose_from_observation(observation))

    def record_error(self, error_type: str, message: str) -> None:
        self.errors.append({"error_type": error_type, "message": message})
        self.hover_required = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_instruction": self.user_instruction,
            "semantic_plan": self.semantic_plan.to_dict(),
            "current_phase_index": self.current_phase_index,
            "current_phase": self.current_phase_name,
            "current_phase_key": self.current_phase_key,
            "latest_observation": copy.deepcopy(self.latest_observation),
            "latest_pose": copy.deepcopy(self.latest_pose),
            "errors": list(self.errors),
            "safety": {"hover_required": self.hover_required},
            "yaw_scan_accum_deg": self.yaw_scan_accum_deg,
            "current_phase_progress": copy.deepcopy(self.current_phase_progress),
            "current_phase_window_count": self.current_phase_window_count,
            "max_phase_window_count": self.max_phase_window_count,
            "connected": self.connected,
            "airborne": self.airborne,
            "completed": self.completed,
        }


def _pose_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    pose = observation.get("pose") if isinstance(observation, dict) else None
    return dict(pose) if isinstance(pose, dict) else {}


def _clean_pose(pose: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(pose) if isinstance(pose, dict) else {}
    data.pop("velocity", None)
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


def _pose_value(pose: dict[str, Any], key: str) -> float:
    if key in pose:
        return float(pose.get(key) or 0.0)
    position = pose.get("position")
    if isinstance(position, dict):
        return float(position.get(key) or 0.0)
    return 0.0


def _yaw_deg(pose: dict[str, Any]) -> float:
    if "yaw_deg" in pose:
        return float(pose.get("yaw_deg") or 0.0)
    orientation = pose.get("orientation")
    if isinstance(orientation, dict) and "yaw_deg" in orientation:
        return float(orientation.get("yaw_deg") or 0.0)
    euler_rad = pose.get("euler_rad")
    if isinstance(euler_rad, dict) and "yaw" in euler_rad:
        return float(euler_rad.get("yaw") or 0.0) * 180.0 / 3.141592653589793
    return 0.0


def _normalize_yaw_delta_deg(delta: float) -> float:
    normalized = (float(delta) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 and delta > 0 else normalized


def _pose_measurements(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    dx = _pose_value(after, "x") - _pose_value(before, "x")
    dy = _pose_value(after, "y") - _pose_value(before, "y")
    dz = _pose_value(after, "z") - _pose_value(before, "z")
    yaw_delta = _yaw_deg(after) - _yaw_deg(before) if before and after else None
    yaw_rad = math.radians(_yaw_deg(before))
    forward = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    right = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
    return {
        "world_delta": {"dx": round(dx, 3), "dy": round(dy, 3), "dz": round(dz, 3)},
        "pose_delta": {"dx": round(dx, 3), "dy": round(dy, 3), "dz": round(dz, 3)},
        "forward_displacement_m": round(forward, 3),
        "right_displacement_m": round(right, 3),
        "up_displacement_m": round(-dz, 3),
        "planar_displacement_m": round((dx * dx + dy * dy) ** 0.5, 3),
        "vertical_delta_m": round(dz, 3),
        "yaw_delta_deg": round(_normalize_yaw_delta_deg(yaw_delta), 3) if yaw_delta is not None else None,
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
