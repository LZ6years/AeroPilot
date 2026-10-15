"""Track the current mission control state for short-horizon runtime loops.

MissionState intentionally stores the latest runtime state, not flight history.
Historical actions, observations, poses, and frames live in FlightMemory JSONL
records and are summarized for models by context modules.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.uav import pose_measurements
from src.agent.planning.task_plan import TaskPlan


@dataclass
class MissionState:
    """Current instruction state for the unified loop, without semantic phases.

    History belongs to Transcript/FlightMemory. Plan changes never affect motion
    origins, physical-window guards or the nonphysical-round allowance. This
    state is wired by the unified runner, not inferred from a legacy Phase.
    """

    run_id: str
    user_instruction: str
    protocol_context: str = ""
    task_plan: TaskPlan | None = None
    loaded_knowledge: list[dict[str, Any]] = field(default_factory=list)
    latest_observation: dict[str, Any] = field(default_factory=dict)
    latest_pose: dict[str, Any] = field(default_factory=dict)
    instruction_origin_pose: dict[str, Any] = field(default_factory=dict)
    instruction_origin_observation_id: str | None = None
    before_last_action_window_id: str | None = None
    motion_summary: dict[str, Any] = field(default_factory=dict)
    physical_window_count: int = 0
    max_physical_windows: int = 300
    nonphysical_round_count: int = 0
    max_nonphysical_rounds: int = 10
    connected: bool = False
    airborne: bool = False
    observation_modalities: tuple[str, ...] = ("rgb", "depth", "pose")
    errors: list[dict[str, Any]] = field(default_factory=list)
    hover_required: bool = False
    yaw_scan_offset_deg: float = 0.0
    yaw_scan_extent_deg: float = 0.0
    yaw_scan_turn_count: int = 0
    yaw_scan_recharge_count: int = 0
    completed: bool = False
    stopped: bool = False
    stopped_reason: str | None = None
    stop_source: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_physical_windows", "max_nonphysical_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.user_instruction.strip():
            raise ValueError("instruction must not be empty")

    def begin_physical_window(self) -> None:
        """Consume one window only after entire-turn prevalidation and Safety."""
        if self.completed or self.stopped:
            raise ValueError("cannot execute a terminal instruction")
        if self.physical_window_count >= self.max_physical_windows:
            raise ValueError("instruction physical-window limit reached")
        self.physical_window_count += 1
        self.before_last_action_window_id = self.latest_observation.get("observation_id")

    def record_nonphysical_round(self) -> None:
        """Count knowledge, plan and repair turns; only a physical batch resets."""
        if self.completed or self.stopped:
            raise ValueError("cannot continue a terminal instruction")
        if self.nonphysical_round_count >= self.max_nonphysical_rounds:
            raise ValueError("nonphysical round limit reached")
        self.nonphysical_round_count += 1

    def finish_physical_window(self, observation: dict[str, Any], measurements: dict[str, Any]) -> None:
        self.record_observation(observation)
        self.motion_summary["last_window"] = _compact_measurements(measurements)
        self.nonphysical_round_count = 0

    def record_observation(self, observation: dict[str, Any]) -> None:
        self.latest_observation = _clean_observation(observation)
        self.latest_pose = _clean_pose(_pose_from_observation(observation))
        if not self.instruction_origin_pose and self.latest_pose:
            self.instruction_origin_pose = copy.deepcopy(self.latest_pose)
            self.instruction_origin_observation_id = observation.get("observation_id")
        self.motion_summary["measurements"] = _compact_measurements(
            pose_measurements(self.instruction_origin_pose, self.latest_pose)
            if self.instruction_origin_pose and self.latest_pose else {}
        )
        # Refreshes for knowledge/plan turns retain the most recent batch's
        # measured motion. Missing motion never gets replaced by invented zeros.
        self.motion_summary.setdefault("last_window", _compact_measurements({}))

    def mark_stopped(self, reason: str, source: str = "runtime") -> None:
        self.completed = False
        self.stopped = True
        self.stopped_reason = reason
        self.stop_source = source

    def record_action(self, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        _record_live_action(self, tool, args, result)

    def recharge_yaw_scan(self) -> dict[str, Any]:
        return _recharge_yaw_scan(self)

    def record_error(self, error_type: str, message: str, diagnostic: dict[str, Any] | None = None) -> None:
        error = {"error_type": error_type, "message": message}
        if diagnostic is not None:
            from src.tools.diagnostics import Diagnostic
            error["diagnostic"] = Diagnostic(**diagnostic).to_dict()
        self.errors.append(error)
        self.hover_required = True

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        result = asdict(self)
        result["task_plan"] = self.task_plan.to_dict() if self.task_plan else None
        result["observation_modalities"] = list(self.observation_modalities)
        return result


def _record_live_action(state: Any, tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    # Update only live control flags; action history is written to memory/actions.jsonl.
    if result.get("ok") is True:
        if tool == "uav.connect":
            state.connected = True
        if tool == "uav.takeoff":
            state.airborne = True
        if tool == "uav.land":
            state.airborne = False
        if tool == "uav.turn_relative":
            state.yaw_scan_offset_deg += float(args.get("yaw_deg", 0.0))
            state.yaw_scan_extent_deg = max(
                state.yaw_scan_extent_deg,
                abs(state.yaw_scan_offset_deg),
            )
            state.yaw_scan_turn_count += 1


def _recharge_yaw_scan(state: Any) -> dict[str, Any]:
    """Start a fresh stationary scan segment after verified translation.

    TaskRuntime owns the displacement threshold because it has both the
    post-action pose measurement and the active safety configuration.
    MissionState only performs the atomic state transition and returns the
    previous segment for audit logging.
    """

    previous = {
        "offset_deg": state.yaw_scan_offset_deg,
        "extent_deg": state.yaw_scan_extent_deg,
        "turn_count": state.yaw_scan_turn_count,
    }
    state.yaw_scan_offset_deg = 0.0
    state.yaw_scan_extent_deg = 0.0
    state.yaw_scan_turn_count = 0
    state.yaw_scan_recharge_count += 1
    return {
        "previous_segment": previous,
        "recharge_count": state.yaw_scan_recharge_count,
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


def _compact_measurements(measurements: dict[str, Any]) -> dict[str, Any]:
    return {
        "forward_m": _round_measurement(measurements.get("forward_displacement_m")),
        "right_m": _round_measurement(measurements.get("right_displacement_m")),
        "up_m": _round_measurement(measurements.get("up_displacement_m")),
        "planar_m": _round_measurement(measurements.get("planar_displacement_m")),
        "vertical_m": _round_measurement(measurements.get("vertical_delta_m")),
        "yaw_delta_deg": _round_measurement(measurements.get("yaw_delta_deg")),
    }


def _round_measurement(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
