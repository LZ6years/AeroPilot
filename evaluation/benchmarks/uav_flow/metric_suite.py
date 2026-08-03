"""UAV-Flow metric suite backed by the shared Runtime episode trace."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core import (
    BenchmarkExecutionOutcome,
    EpisodeTrace,
)
from evaluation.benchmarks.uav_flow.adapter import UavFlowEvaluationPayload
from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.metrics import evaluate_case_ndtw
from evaluation.benchmarks.uav_flow.trajectory import (
    UavFlowCoordinateAdapter,
    export_trajectory,
)
from src.runtime.configs.action_config import UavActionConfig
from src.uav import body_velocity_to_world_delta, move_body_velocity_from_args

TURN_TRACKING_TOLERANCE_DEG = 1.0
TRANSLATION_TRACKING_TOLERANCE_M = 0.05


class UavFlowMetricSuite:
    """Export the official trajectory shape and calculate UAV-Flow nDTW."""

    def __init__(
        self,
        case: UavFlowCase,
        case_dir: str | Path,
        *,
        action_config: UavActionConfig | None = None,
    ) -> None:
        self.case = case
        self.case_dir = Path(case_dir)
        self.action_config = action_config or UavActionConfig()
        self.coordinate_adapter = UavFlowCoordinateAdapter()
        self.trajectory_meta: dict[str, Any] = {}

    def evaluate(
        self,
        payload: UavFlowEvaluationPayload | None,
        trace: EpisodeTrace,
        outcome: BenchmarkExecutionOutcome,
    ) -> dict[str, Any]:
        if payload is None:
            raise ValueError("UAV-Flow evaluation payload is required")
        if payload.case_id != self.case.case_id:
            raise ValueError("UAV-Flow evaluation payload case mismatch")
        local_poses = self.coordinate_adapter.trace_poses(
            trace, payload.initial_pos
        )
        trajectory_path = self.case_dir / "uav_agent_trajectory.json"
        self.trajectory_meta = export_trajectory(
            self.case, local_poses, trajectory_path
        )
        ndtw = evaluate_case_ndtw(
            trajectory_path,
            [list(row) for row in payload.reference_path_preprocessed],
            class_name=payload.class_name,
        )
        action_tracking = _backend_action_tracking(
            trace,
            action_config=self.action_config,
            completed=outcome.completed,
        )
        return {
            "ndtw": ndtw,
            "trajectory": dict(self.trajectory_meta),
            "turn_diagnostics": _turn_diagnostics(
                self.case,
                trace,
                local_poses,
                action_tracking=action_tracking,
            ),
            "backend_action_tracking": action_tracking,
        }


def _turn_diagnostics(
    case: UavFlowCase,
    trace: EpisodeTrace,
    local_poses: list[dict[str, float]],
    *,
    action_tracking: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compare one real Turn trajectory with its evaluation-only truth.

    These values are written only to benchmark outputs. They must never flow
    back into Runtime or planner context because the reference yaw is hidden
    evaluation truth.
    """

    if str(case.class_name or "").strip().lower() not in {"turn", "rotate"}:
        return None

    reference_yaw_delta = _reference_yaw_delta(case)
    actual_yaw_delta = (
        _normalize_yaw(float(local_poses[-1]["yaw_deg"]))
        if local_poses
        else None
    )
    signed_yaw_error = (
        _normalize_yaw(actual_yaw_delta - reference_yaw_delta)
        if actual_yaw_delta is not None
        and reference_yaw_delta is not None
        else None
    )
    translation_drift = _translation_drift(local_poses)
    initial_position_error, initial_yaw_error = _initial_pose_errors(
        case,
        trace,
    )
    initial_pose_ok = (
        initial_position_error is not None
        and initial_yaw_error is not None
        and initial_position_error <= 0.05
        and initial_yaw_error <= 1.0
    )
    return {
        "reference_yaw_delta_deg": reference_yaw_delta,
        "actual_yaw_delta_deg": actual_yaw_delta,
        "final_yaw_error_deg": (
            abs(signed_yaw_error)
            if signed_yaw_error is not None
            else None
        ),
        "signed_final_yaw_error_deg": signed_yaw_error,
        "translation_drift_m": translation_drift,
        "unexpected_translation": (
            translation_drift is not None and translation_drift > 0.05
        ),
        "initial_position_error_m": initial_position_error,
        "initial_yaw_error_deg": initial_yaw_error,
        "initial_pose_within_tolerance": initial_pose_ok,
        "initial_pose_position_tolerance_m": 0.05,
        "initial_pose_yaw_tolerance_deg": 1.0,
        "unexpected_translation_threshold_m": 0.05,
        "backend_turn_tracking": _backend_turn_tracking(
            trace,
            action_tracking=action_tracking,
        ),
    }


def _backend_action_tracking(
    trace: EpisodeTrace,
    *,
    action_config: UavActionConfig,
    completed: bool,
) -> dict[str, Any]:
    """Compare requested primitives with settled post-action observations."""

    rows: list[dict[str, Any]] = []
    previous_pose = (
        dict(trace.samples[0].pose) if trace.samples else None
    )
    for sample in trace.samples[1:]:
        if sample.tool not in {"uav.turn_relative", "uav.move_body"}:
            observed_pose = sample.post_action_observation_pose
            if observed_pose is not None:
                previous_pose = dict(observed_pose)
            elif sample.pose:
                previous_pose = dict(sample.pose)
            continue
        observed_pose = sample.post_action_observation_pose
        row: dict[str, Any] = {
            "action_id": sample.action_id,
            "tool": sample.tool,
            "observation_timestamp": (
                sample.post_action_observation_timestamp
            ),
            "expected_delta_m": None,
            "observed_delta_m": None,
            "position_tracking_error_m": None,
            "requested_yaw_delta_deg": None,
            "observed_yaw_delta_deg": None,
            "yaw_tracking_error_deg": None,
            "within_tolerance": None,
            "unavailable_reason": None,
        }
        if previous_pose is None:
            row["unavailable_reason"] = "pre_action_pose_unavailable"
        elif observed_pose is None:
            row["unavailable_reason"] = (
                "post_action_observation_unavailable"
            )
        elif sample.tool == "uav.turn_relative":
            requested = _finite_float(sample.action_args.get("yaw_deg"))
            previous_yaw = _finite_float(previous_pose.get("yaw_deg"))
            observed_yaw = _finite_float(observed_pose.get("yaw_deg"))
            if requested is None:
                row["unavailable_reason"] = "requested_yaw_unavailable"
            elif previous_yaw is None or observed_yaw is None:
                row["unavailable_reason"] = "observation_yaw_unavailable"
            else:
                observed = _normalize_yaw(observed_yaw - previous_yaw)
                error = _normalize_yaw(observed - requested)
                row.update(
                    {
                        "requested_yaw_delta_deg": requested,
                        "observed_yaw_delta_deg": observed,
                        "yaw_tracking_error_deg": error,
                        "within_tolerance": (
                            abs(error) <= TURN_TRACKING_TOLERANCE_DEG
                        ),
                    }
                )
        else:
            expected = _expected_move_delta(
                sample.action_args,
                previous_pose,
                action_config,
            )
            observed = _observed_position_delta(
                previous_pose,
                observed_pose,
            )
            if expected is None:
                row["unavailable_reason"] = (
                    "requested_translation_unavailable"
                )
            elif observed is None:
                row["unavailable_reason"] = (
                    "observation_position_unavailable"
                )
            else:
                error = math.sqrt(
                    sum(
                        (observed[key] - expected[key]) ** 2
                        for key in ("dx", "dy", "dz")
                    )
                )
                row.update(
                    {
                        "expected_delta_m": expected,
                        "observed_delta_m": observed,
                        "position_tracking_error_m": error,
                        "within_tolerance": (
                            error <= TRANSLATION_TRACKING_TOLERANCE_M
                        ),
                    }
                )
        rows.append(row)
        if observed_pose is not None:
            previous_pose = dict(observed_pose)
        elif sample.pose:
            previous_pose = dict(sample.pose)

    position_errors = [
        float(row["position_tracking_error_m"])
        for row in rows
        if row["position_tracking_error_m"] is not None
    ]
    yaw_errors = [
        abs(float(row["yaw_tracking_error_deg"]))
        for row in rows
        if row["yaw_tracking_error_deg"] is not None
    ]
    observed_rows = [
        row for row in rows if row["unavailable_reason"] is None
    ]
    return {
        "formal_action_count": trace.action_count,
        "tracked_action_count": len(rows),
        "observed_count": len(observed_rows),
        "unavailable_count": len(rows) - len(observed_rows),
        "zero_action_completion": (
            bool(completed) and trace.action_count == 0
        ),
        "translation_tolerance_m": TRANSLATION_TRACKING_TOLERANCE_M,
        "yaw_tolerance_deg": TURN_TRACKING_TOLERANCE_DEG,
        "mean_position_error_m": (
            sum(position_errors) / len(position_errors)
            if position_errors
            else None
        ),
        "max_position_error_m": (
            max(position_errors) if position_errors else None
        ),
        "mean_abs_yaw_error_deg": (
            sum(yaw_errors) / len(yaw_errors) if yaw_errors else None
        ),
        "max_abs_yaw_error_deg": (
            max(yaw_errors) if yaw_errors else None
        ),
        "all_observed_within_tolerance": (
            all(row["within_tolerance"] is True for row in rows)
            if rows
            else None
        ),
        "samples": rows,
    }


def _backend_turn_tracking(
    trace: EpisodeTrace,
    *,
    action_tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure requested yaw against the first post-action observation.

    Action-result poses describe the Backend target pose.  This diagnostic
    deliberately uses the following observation instead so evaluation reports
    the settled simulator state rather than echoing the requested command.
    """

    if action_tracking is None:
        action_tracking = _backend_action_tracking(
            trace,
            action_config=UavActionConfig(),
            completed=False,
        )
    rows = []
    for item in action_tracking.get("samples", []):
        if not isinstance(item, dict) or item.get("tool") != "uav.turn_relative":
            continue
        error = item.get("yaw_tracking_error_deg")
        rows.append(
            {
                "action_id": item.get("action_id"),
                "requested_yaw_delta_deg": item.get(
                    "requested_yaw_delta_deg"
                ),
                "observed_yaw_delta_deg": item.get(
                    "observed_yaw_delta_deg"
                ),
                "tracking_error_deg": error,
                "abs_tracking_error_deg": (
                    abs(float(error)) if error is not None else None
                ),
                "within_tolerance": item.get("within_tolerance"),
                "observation_timestamp": item.get(
                    "observation_timestamp"
                ),
                "unavailable_reason": item.get("unavailable_reason"),
            }
        )

    observed_errors = [
        float(item["abs_tracking_error_deg"])
        for item in rows
        if item["abs_tracking_error_deg"] is not None
    ]
    return {
        "tolerance_deg": TURN_TRACKING_TOLERANCE_DEG,
        "turn_action_count": len(rows),
        "observed_count": len(observed_errors),
        "unavailable_count": len(rows) - len(observed_errors),
        "mean_abs_error_deg": (
            sum(observed_errors) / len(observed_errors)
            if observed_errors
            else None
        ),
        "max_abs_error_deg": (
            max(observed_errors) if observed_errors else None
        ),
        "all_observed_within_tolerance": (
            bool(observed_errors)
            and all(
                error <= TURN_TRACKING_TOLERANCE_DEG
                for error in observed_errors
            )
        ),
        "samples": rows,
    }


def _expected_move_delta(
    args: dict[str, Any],
    before_pose: dict[str, Any],
    action_config: UavActionConfig,
) -> dict[str, float] | None:
    try:
        vx, vy, vz, duration = move_body_velocity_from_args(
            args,
            action_config,
            vertical_sign=1.0,
        )
        yaw_deg = float(before_pose["yaw_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    delta = body_velocity_to_world_delta(
        vx,
        vy,
        vz,
        duration,
        yaw_deg,
    )
    return {
        key: float(delta[key])
        for key in ("dx", "dy", "dz")
    }


def _observed_position_delta(
    before_pose: dict[str, Any],
    after_pose: dict[str, Any],
) -> dict[str, float] | None:
    try:
        return {
            "dx": float(after_pose["x"]) - float(before_pose["x"]),
            "dy": float(after_pose["y"]) - float(before_pose["y"]),
            "dz": float(after_pose["z"]) - float(before_pose["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _reference_yaw_delta(case: UavFlowCase) -> float | None:
    rows = case.reference_path_preprocessed
    if len(rows) >= 2 and len(rows[0]) >= 5 and len(rows[-1]) >= 5:
        return _normalize_yaw(float(rows[-1][4]) - float(rows[0][4]))
    if len(case.initial_pos) >= 5 and len(case.end_pos) >= 5:
        return _normalize_yaw(
            float(case.end_pos[4]) - float(case.initial_pos[4])
        )
    return None


def _translation_drift(
    local_poses: list[dict[str, float]],
) -> float | None:
    if not local_poses:
        return None
    final = local_poses[-1]
    return math.sqrt(
        float(final.get("x", 0.0)) ** 2
        + float(final.get("y", 0.0)) ** 2
        + float(final.get("z", 0.0)) ** 2
    )


def _initial_pose_errors(
    case: UavFlowCase,
    trace: EpisodeTrace,
) -> tuple[float | None, float | None]:
    if not trace.poses or len(case.initial_pos) < 5:
        return None, None
    pose = trace.poses[0]
    expected = case.initial_pos
    try:
        position_error = math.sqrt(
            (float(pose["x"]) - float(expected[0]) / 100.0) ** 2
            + (float(pose["y"]) - float(expected[1]) / 100.0) ** 2
            + (float(pose["z"]) - float(expected[2]) / 100.0) ** 2
        )
        yaw_error = abs(
            _normalize_yaw(
                float(pose["yaw_deg"]) - float(expected[4])
            )
        )
    except (KeyError, TypeError, ValueError):
        return None, None
    return position_error, yaw_error


def _normalize_yaw(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if normalized == -180.0 else normalized
