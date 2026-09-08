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
    """Compare composed physical windows with settled observations.

    A Runtime window may contain several ordered physical calls but owns only
    one automatic observation. Per-action timestamp association is therefore
    invalid; only the explicit window boundary can measure settled tracking.
    """

    rows: list[dict[str, Any]] = []
    for window in trace.physical_windows:
        before = window.observation_before_pose
        after = window.observation_after_pose
        row: dict[str, Any] = {
            "decision_id": window.decision_id,
            "phase": window.phase,
            "window_index": window.window_index,
            "window_status": window.status,
            "observation_before_id": window.observation_before_id,
            "observation_after_id": window.observation_after_id,
            "observation_timestamp": window.observation_after_timestamp,
            "expected_delta_m": None,
            "observed_delta_m": None,
            "position_tracking_error_m": None,
            "expected_yaw_delta_deg": None,
            "observed_yaw_delta_deg": None,
            "yaw_tracking_error_deg": None,
            "within_tolerance": None,
            "unavailable_reason": window.unavailable_reason,
            "ordered_steps": [],
        }
        row["ordered_steps"] = [
            {
                "step_index": step.step_index,
                "tool_call_id": step.tool_call_id,
                "action_id": step.action_id,
                "requested_tool": step.requested_tool,
                "tool": step.tool,
                "status": step.status,
                "requested_args": dict(step.requested_args),
                "executed_args": (
                    dict(step.executed_args)
                    if step.executed_args is not None
                    else None
                ),
                "safety_interventions": [
                    dict(item) for item in step.safety_interventions
                ],
                "acknowledged_pose": (
                    dict(step.acknowledged_pose)
                    if step.acknowledged_pose is not None
                    else None
                ),
            }
            for step in window.ordered_steps
        ]
        expected_pose = dict(before) if before is not None else None
        if row["unavailable_reason"] is None and expected_pose is None:
            row["unavailable_reason"] = "pre_window_observation_unavailable"
        if row["unavailable_reason"] is None and after is None:
            row["unavailable_reason"] = "post_window_observation_unavailable"
        if row["unavailable_reason"] is None:
            for step in window.ordered_steps:
                if step.status != "ok":
                    continue
                if step.executed_args is None:
                    row["unavailable_reason"] = "executed_args_unavailable"
                    break
                if step.tool == "uav.hover":
                    continue
                if step.tool == "uav.turn_relative":
                    requested = _finite_float(step.executed_args.get("yaw_deg"))
                    current_yaw = _finite_float(expected_pose.get("yaw_deg"))
                    if requested is None or current_yaw is None:
                        row["unavailable_reason"] = "requested_yaw_unavailable"
                        break
                    expected_pose["yaw_deg"] = _normalize_yaw(
                        current_yaw + requested
                    )
                    continue
                if step.tool == "uav.move_body":
                    delta = _expected_move_delta(
                        step.executed_args,
                        expected_pose,
                        action_config,
                    )
                    if delta is None:
                        row["unavailable_reason"] = (
                            "requested_translation_unavailable"
                        )
                        break
                    try:
                        for axis, key in (("x", "dx"), ("y", "dy"), ("z", "dz")):
                            expected_pose[axis] = float(expected_pose[axis]) + delta[key]
                    except (KeyError, TypeError, ValueError):
                        row["unavailable_reason"] = "pre_window_position_unavailable"
                        break
                    continue
                row["unavailable_reason"] = "unsupported_physical_tool"
                break

        if row["unavailable_reason"] is None and expected_pose is not None and after is not None:
            expected_delta = _observed_position_delta(before, expected_pose)
            observed_delta = _observed_position_delta(before, after)
            before_yaw = _finite_float(before.get("yaw_deg"))
            expected_yaw = _finite_float(expected_pose.get("yaw_deg"))
            observed_yaw = _finite_float(after.get("yaw_deg"))
            if expected_delta is None or observed_delta is None:
                row["unavailable_reason"] = "observation_position_unavailable"
            elif before_yaw is None or expected_yaw is None or observed_yaw is None:
                row["unavailable_reason"] = "observation_yaw_unavailable"
            else:
                position_error = math.sqrt(
                    sum(
                        (observed_delta[key] - expected_delta[key]) ** 2
                        for key in ("dx", "dy", "dz")
                    )
                )
                expected_yaw_delta = _normalize_yaw(expected_yaw - before_yaw)
                observed_yaw_delta = _normalize_yaw(observed_yaw - before_yaw)
                yaw_error = _normalize_yaw(observed_yaw_delta - expected_yaw_delta)
                row.update(
                    {
                        "expected_delta_m": expected_delta,
                        "observed_delta_m": observed_delta,
                        "position_tracking_error_m": position_error,
                        "expected_yaw_delta_deg": expected_yaw_delta,
                        "observed_yaw_delta_deg": observed_yaw_delta,
                        "yaw_tracking_error_deg": yaw_error,
                        "within_tolerance": (
                            position_error <= TRANSLATION_TRACKING_TOLERANCE_M
                            and abs(yaw_error) <= TURN_TRACKING_TOLERANCE_DEG
                        ),
                    }
                )
        rows.append(row)

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
    observed_rows = [row for row in rows if row["unavailable_reason"] is None]
    trace_supported = trace.trace_schema_version == "physical_window_v2"
    trace_valid = trace_supported and trace.trace_integrity == "ok"
    return {
        "schema_version": "uav_flow_window_tracking_v2",
        "association_source": (
            "explicit_runtime_window"
            if trace_supported
            else "unsupported_legacy_trace_schema"
        ),
        "trace_valid": trace_valid,
        "tracking_valid": trace_valid and len(observed_rows) == len(rows),
        "trace_issues": list(trace.trace_issues),
        "physical_call_count": trace.physical_call_count,
        "motion_call_count": trace.motion_call_count,
        "physical_window_count": trace.physical_window_count,
        "failed_window_count": sum(
            window.status != "ok" for window in trace.physical_windows
        ),
        "failed_or_skipped_step_count": sum(
            step.status != "ok"
            for window in trace.physical_windows
            for step in window.ordered_steps
        ),
        "tracked_window_count": len(rows),
        "observed_window_count": len(observed_rows),
        "unavailable_window_count": len(rows) - len(observed_rows),
        "zero_motion_completion": bool(completed) and trace.motion_call_count == 0,
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
        "windows": rows,
    }


def _backend_turn_tracking(
    trace: EpisodeTrace,
    *,
    action_tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project window-level yaw tracking for Turn/Rotate diagnostics."""

    if action_tracking is None:
        action_tracking = _backend_action_tracking(
            trace,
            action_config=UavActionConfig(),
            completed=False,
        )
    rows = []
    for item in action_tracking.get("windows", []):
        if not isinstance(item, dict):
            continue
        steps = item.get("ordered_steps")
        if not isinstance(steps, list) or not any(
            isinstance(step, dict)
            and step.get("status") == "ok"
            and step.get("tool") == "uav.turn_relative"
            for step in steps
        ):
            continue
        error = item.get("yaw_tracking_error_deg")
        rows.append(
            {
                "decision_id": item.get("decision_id"),
                "window_index": item.get("window_index"),
                "expected_yaw_delta_deg": item.get("expected_yaw_delta_deg"),
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

        if error is not None:
            rows[-1]["within_tolerance"] = (
                abs(float(error)) <= TURN_TRACKING_TOLERANCE_DEG
            )

    observed_errors = [
        float(item["abs_tracking_error_deg"])
        for item in rows
        if item["abs_tracking_error_deg"] is not None
    ]
    return {
        "tolerance_deg": TURN_TRACKING_TOLERANCE_DEG,
        "turn_window_count": len(rows),
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
