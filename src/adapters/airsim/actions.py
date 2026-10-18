"""Execute and verify paused AirSim action windows, including failure cleanup."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.uav.frames import normalize_yaw_deg, normalize_yaw_delta_deg, position_distance_m, yaw_deg
from ..uav_primitives import PrimitiveResult


@dataclass(frozen=True)
class ActionVerificationPolicy:
    """Conservative gross-effect checks, not a trajectory-tracking controller.

    Bounds tolerate acceleration and collision deflection. They detect near-zero
    progress and discontinuities; they do not certify that a route was followed.
    The yaw timeout is simulator time, while SDK RPC timeouts are wall time.
    """

    yaw_timeout_sec: float = 8.0
    yaw_margin_deg: float = 5.0
    yaw_verification_slack_deg: float = 0.5
    min_progress_ratio: float = 0.1
    max_translation_ratio: float = 3.0
    translation_slack_m: float = 5.0

    def __post_init__(self) -> None:
        values = vars(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("action verification limits must be finite and positive")
        if self.min_progress_ratio >= 1 or self.max_translation_ratio <= 1:
            raise ValueError("progress ratios must bracket normal motion")


@dataclass(frozen=True)
class _PhysicalActionOutcome:
    """Capture one paused-Multirotor RPC window without losing failure detail."""

    before: dict[str, Any]
    after: dict[str, Any]
    rpc_result: Any
    rpc_error: dict[str, str] | None
    freeze_error: dict[str, str] | None
    elapsed_sec: float
    cancel_attempted: bool
    cancel_error: dict[str, str] | None
    timings_sec: dict[str, float]

    @property
    def failure_kind(self) -> str | None:
        if self.rpc_error is not None:
            return "rpc_error"
        if self.rpc_result is False:
            return "controller_timeout_or_rejected"
        if self.cancel_error is not None:
            return "cancel_cleanup_error"
        if self.freeze_error is not None:
            return "freeze_cleanup_error"
        return None


class PausedMultirotorActionExecutor:
    """Own RPC completion and effect verification; strategy supplies pose/freeze I/O."""

    _NO_EFFECT_MIN_REQUESTED_YAW_DEG = 5.0
    _NO_EFFECT_MAX_OBSERVED_YAW_DEG = 1.0
    _NO_EFFECT_MIN_EXPECTED_TRANSLATION_M = 0.1
    _NO_EFFECT_MAX_OBSERVED_TRANSLATION_M = 0.01

    def __init__(self, vehicle_name: str, airsim_provider: Callable[[], Any],
                 get_pose: Callable[[Any], PrimitiveResult],
                 freeze: Callable[[Any], PrimitiveResult],
                 canonical_pose_result: Callable[..., PrimitiveResult],
                 policy: ActionVerificationPolicy | None = None) -> None:
        self.vehicle_name = vehicle_name
        self._airsim = airsim_provider
        self.get_pose = get_pose
        self._freeze_at_current_pose = freeze
        self._canonical_pose_result = canonical_pose_result
        self.policy = policy or ActionVerificationPolicy()
        # Unknown server-task state must not leak into a later action window.
        # A new scene/adapter is required after unsuccessful cleanup.
        self._recovery_required = False

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        if vx == 0.0 and vy == 0.0 and vz == 0.0:
            # Runtime stabilization expresses braking as a zero-velocity move.
            # Reopening physics here would recreate the drift this strategy is
            # designed to eliminate, so freeze immediately and ignore duration.
            return self.hold(client)

        outcome = self._run_physical_action(
            client,
            lambda: client.moveByVelocityBodyFrameAsync(
                vx,
                vy,
                -vz,
                duration,
                vehicle_name=self.vehicle_name,
            ),
        )
        expected_distance = math.sqrt(vx * vx + vy * vy + vz * vz) * duration
        actual_distance = position_distance_m(
            self._snapshot_pose(outcome.before),
            self._snapshot_pose(outcome.after),
        )
        tracking = {
            "expected_translation_m": expected_distance,
            "observed_translation_m": actual_distance,
        }
        failure_kind = outcome.failure_kind
        collision_kind: str | None = None
        discontinuity = actual_distance is not None and actual_distance > max(
            self.policy.translation_slack_m,
            self.policy.max_translation_ratio * expected_distance,
        )
        tracking["motion_discontinuity"] = discontinuity
        # RPC failure and new contact do not make a position jump trustworthy.
        # Preserve the RPC result independently while poisoning further control.
        if discontinuity:
            failure_kind = "motion_discontinuity"
            self._recovery_required = True
        elif failure_kind is None and actual_distance is None:
            failure_kind = "rpc_error"
            tracking["verification_error"] = "post-action pose unavailable"
        elif (
            failure_kind is None
            and expected_distance >= self._NO_EFFECT_MIN_EXPECTED_TRANSLATION_M
            and actual_distance < max(self._NO_EFFECT_MAX_OBSERVED_TRANSLATION_M,
                                      expected_distance * self.policy.min_progress_ratio)
        ):
            failure_kind = self._no_effect_failure_kind(outcome.after)
            if (
                failure_kind == "collision_blocked"
                and self._has_trustworthy_final_pose(outcome)
            ):
                failure_kind = None
                collision_kind = "blocked"
            elif failure_kind == "control_no_effect" and actual_distance >= self._NO_EFFECT_MAX_OBSERVED_TRANSLATION_M:
                failure_kind = "control_insufficient_progress"
        tracking["verification_policy"] = vars(self.policy).copy()
        return self._physical_result(
            action_name="move_body",
            outcome=outcome,
            failure_kind=failure_kind,
            collision_kind=collision_kind,
            requested={
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "duration": duration,
            },
            tracking=tracking,
        )

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        target_started = time.monotonic()
        pose = self.get_pose(client)
        current_yaw = yaw_deg(pose.data or {})
        target_yaw = normalize_yaw_deg(current_yaw + yaw_delta_deg)
        target_elapsed = time.monotonic() - target_started
        outcome = self._run_physical_action(
            client,
            lambda: client.rotateToYawAsync(
                target_yaw,
                timeout_sec=self.policy.yaw_timeout_sec,
                margin=self.policy.yaw_margin_deg,
                vehicle_name=self.vehicle_name,
            ),
        )
        outcome.timings_sec["target_pose"] = target_elapsed
        outcome.timings_sec["total"] += target_elapsed
        after_yaw = self._snapshot_yaw_deg(outcome.after)
        observed_delta = (
            normalize_yaw_delta_deg(after_yaw - current_yaw)
            if after_yaw is not None
            else None
        )
        target_error = (
            normalize_yaw_delta_deg(after_yaw - target_yaw)
            if after_yaw is not None
            else None
        )
        tracking = {
            "requested_yaw_delta_deg": yaw_delta_deg,
            "target_yaw_deg": target_yaw,
            "observed_yaw_delta_deg": observed_delta,
            "target_yaw_error_deg": target_error,
            "observed_translation_m": position_distance_m(
                self._snapshot_pose(outcome.before), self._snapshot_pose(outcome.after)),
        }
        failure_kind = outcome.failure_kind
        collision_kind: str | None = None
        discontinuity = (
            tracking["observed_translation_m"] is not None
            and tracking["observed_translation_m"] > self.policy.translation_slack_m
        )
        tracking["motion_discontinuity"] = discontinuity
        if discontinuity:
            failure_kind = "motion_discontinuity"
            self._recovery_required = True
        if failure_kind is None and collision_kind is None and observed_delta is None:
            failure_kind = "rpc_error"
            tracking["verification_error"] = "post-action yaw unavailable"
        elif (
            failure_kind is None
            and collision_kind is None
            and abs(yaw_delta_deg) >= self._NO_EFFECT_MIN_REQUESTED_YAW_DEG
            and abs(observed_delta) < self._NO_EFFECT_MAX_OBSERVED_YAW_DEG
        ):
            failure_kind = self._no_effect_failure_kind(outcome.after)
            if (
                failure_kind == "collision_blocked"
                and self._has_trustworthy_final_pose(outcome)
            ):
                failure_kind = None
                collision_kind = "blocked"
        elif failure_kind is None and collision_kind is None and abs(target_error) > (
            self.policy.yaw_margin_deg + self.policy.yaw_verification_slack_deg
        ):
            failure_kind = "control_target_not_reached"
        return self._physical_result(
            action_name="turn_relative",
            outcome=outcome,
            failure_kind=failure_kind,
            collision_kind=collision_kind,
            requested={"yaw_delta_deg": yaw_delta_deg},
            tracking=tracking,
        )

    def hold(self, client: Any) -> PrimitiveResult:
        """Freeze failures poison control even when invoked by stabilization."""
        try:
            pose = self._freeze_at_current_pose(client)
            if not pose.ok:
                raise RuntimeError(pose.message)
            return PrimitiveResult(
                not self._recovery_required, "AirSim paused Multirotor pose held",
                {"pose": pose.data or {}},
                {"requires_scene_reset": self._recovery_required})
        except Exception as exc:
            self._recovery_required = True
            return PrimitiveResult(False, f"AirSim freeze failed: {exc}",
                                   {"failure_kind": "freeze_cleanup_error", "error": self._exception_payload(exc)},
                                   {"requires_scene_reset": True, "reason": "freeze_cleanup_error"})

    def _run_physical_action(
        self,
        client: Any,
        action: Callable[[], Any],
    ) -> _PhysicalActionOutcome:
        """Execute an async native action and retain both RPC and freeze errors.

        ``msgpackrpc.Future.join()`` only waits; it does not surface the stored
        RPC error or return value.  AirSim actions therefore must be completed
        through ``get()`` before the vehicle is frozen again.
        """

        if self._recovery_required:
            raise RuntimeError("AirSim action state requires recovery; recreate the scene before moving")
        total_started = time.monotonic()
        before = self._multirotor_snapshot(client)
        timings = {"before_snapshot": time.monotonic() - total_started}
        rpc_result: Any = None
        rpc_error: dict[str, str] | None = None
        freeze_error: dict[str, str] | None = None
        cancel_error: dict[str, str] | None = None
        cancel_attempted = False
        started = time.monotonic()
        try:
            stage = time.monotonic()
            try:
                client.simPause(False)
            finally:
                timings["resume"] = time.monotonic() - stage
            stage = time.monotonic()
            try:
                future = action()
            finally:
                timings["submit"] = time.monotonic() - stage
            get_result = getattr(future, "get", None)
            if not callable(get_result):
                raise RuntimeError("AirSim async future does not expose get()")
            stage = time.monotonic()
            try:
                rpc_result = get_result()
            finally:
                timings["future_wait"] = time.monotonic() - stage
        except Exception as exc:
            rpc_error = self._exception_payload(exc)
        finally:
            elapsed_sec = time.monotonic() - started
            timings["action_wait"] = elapsed_sec
            # Client timeout does not cancel an AirSim server task. Pause first
            # to stop unobserved motion, then cancel; freeze is always attempted.
            if rpc_error is not None or rpc_result is False:
                cancel_attempted = True
                cancel_started = time.monotonic()
                pause_error = None
                try:
                    client.simPause(True)
                except Exception as exc:
                    pause_error = self._exception_payload(exc)
                try:
                    client.cancelLastTask(vehicle_name=self.vehicle_name)
                except Exception as exc:
                    cancel_error = self._exception_payload(exc)
                if pause_error:
                    # Preserve both errors even if cancellation itself succeeded.
                    cancel_error = {**(cancel_error or {}), "pause_error": str(pause_error)}
                timings["cancel"] = time.monotonic() - cancel_started
            else:
                timings["cancel"] = 0.0
            freeze_started = time.monotonic()
            try:
                frozen = self._freeze_at_current_pose(client)
                if not frozen.ok:
                    raise RuntimeError(frozen.message)
            except Exception as exc:
                freeze_error = self._exception_payload(exc)
                # Best effort: a failed pose clamp must still leave physics
                # paused so an RPC error cannot turn into uncontrolled drift.
                try:
                    client.simPause(True)
                except Exception:
                    pass
            timings["freeze"] = time.monotonic() - freeze_started
            self._recovery_required = cancel_error is not None or freeze_error is not None
        after_started = time.monotonic()
        after = self._multirotor_snapshot(client)
        timings["after_snapshot"] = time.monotonic() - after_started
        timings["total"] = time.monotonic() - total_started
        return _PhysicalActionOutcome(
            before=before,
            after=after,
            rpc_result=rpc_result,
            rpc_error=rpc_error,
            freeze_error=freeze_error,
            elapsed_sec=elapsed_sec,
            cancel_attempted=cancel_attempted,
            cancel_error=cancel_error,
            timings_sec=timings,
        )

    def _physical_result(
        self,
        *,
        action_name: str,
        outcome: _PhysicalActionOutcome,
        failure_kind: str | None,
        collision_kind: str | None,
        requested: dict[str, Any],
        tracking: dict[str, Any],
    ) -> PrimitiveResult:
        """Convert one measured action window into the primitive result contract."""

        diagnostics = {
            "action": action_name,
            "requested": requested,
            "elapsed_sec": outcome.elapsed_sec,
            "timings_sec": outcome.timings_sec,
            "cancel": {"attempted": outcome.cancel_attempted, "error": outcome.cancel_error},
            "recovery_required": self._recovery_required,
            "simulation_elapsed_sec": self._simulation_elapsed(outcome),
            "rpc": {
                "result": self._plain_value(outcome.rpc_result),
                "error": outcome.rpc_error,
            },
            "freeze": {
                "ok": outcome.freeze_error is None,
                "error": outcome.freeze_error,
            },
            "before": outcome.before,
            "after": outcome.after,
            "tracking": tracking,
            "failure_kind": failure_kind,
            "rpc_failure_kind": outcome.failure_kind,
            "collision_kind": collision_kind,
        }
        collision = collision_kind is not None
        ok = failure_kind is None and not collision
        message = f"AirSim paused Multirotor {action_name}"
        if collision:
            message = (
                "Collision detected; the current physical window stopped for "
                "reassessment."
            )
        elif not ok:
            message = f"{message} failed: {failure_kind}"
        data = {
            "pose": self._snapshot_pose(outcome.after) or {},
            "failure_kind": failure_kind,
            "action_diagnostics": diagnostics,
        }
        if collision:
            data.pop("failure_kind", None)
            data["collision"] = self._collision_summary(
                outcome.after, collision_kind
            )
        declared_outcome = "collision" if collision else ("ok" if ok else "error")
        return PrimitiveResult(
            ok,
            message,
            data,
            execution_state={
                "outcome": declared_outcome,
                "requires_scene_reset": self._recovery_required,
                "reason": (
                    f"collision_{collision_kind}" if collision else failure_kind
                ),
                "verified_paused_pose": (
                    self._snapshot_pose(outcome.after)
                    if (ok or collision)
                    and not self._recovery_required
                    and not outcome.after.get("state_error")
                    else None
                ),
            },
        )

    @classmethod
    def _has_trustworthy_final_pose(cls, outcome: _PhysicalActionOutcome) -> bool:
        if outcome.failure_kind is not None or outcome.after.get("state_error"):
            return False
        pose = cls._snapshot_pose(outcome.after)
        if pose is None:
            return False
        position = pose.get("position")
        if not isinstance(position, dict):
            position = pose
        try:
            return all(
                math.isfinite(float(position[axis])) for axis in ("x", "y", "z")
            )
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _collision_summary(
        snapshot: dict[str, Any], kind: str
    ) -> dict[str, Any]:
        collision = snapshot.get("collision")
        collision = collision if isinstance(collision, dict) else {}
        return {
            "kind": kind,
            "object_name": str(collision.get("object_name") or ""),
        }

    @staticmethod
    def _simulation_elapsed(outcome: _PhysicalActionOutcome) -> float | None:
        """Keep simulator nanoseconds separate from wall-clock stage timings."""
        before, after = outcome.before.get("timestamp"), outcome.after.get("timestamp")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and after >= before:
            return (after - before) / 1e9
        return None

    def _multirotor_snapshot(self, client: Any) -> dict[str, Any]:
        """Read control and collision state without making diagnostics fatal."""

        snapshot: dict[str, Any] = {}
        try:
            state = client.getMultirotorState(vehicle_name=self.vehicle_name)
            kin = state.kinematics_estimated
            pose = self._canonical_pose_result(
                self._airsim().Pose(kin.position, kin.orientation),
                "AirSim paused Multirotor diagnostic pose",
                {
                    "x": kin.linear_velocity.x_val,
                    "y": kin.linear_velocity.y_val,
                    "z": -kin.linear_velocity.z_val,
                },
            )
            snapshot["pose"] = pose.data or {}
            for name in (
                "landed_state",
                "ready",
                "ready_message",
                "can_arm",
                "timestamp",
            ):
                if hasattr(state, name):
                    snapshot[name] = self._plain_value(getattr(state, name))
        except Exception as exc:
            snapshot["state_error"] = self._exception_payload(exc)

        collision_call = getattr(client, "simGetCollisionInfo", None)
        if callable(collision_call):
            try:
                snapshot["collision"] = self._collision_payload(
                    collision_call(vehicle_name=self.vehicle_name)
                )
            except Exception as exc:
                snapshot["collision"] = {
                    "available": False,
                    "error": self._exception_payload(exc),
                }
        else:
            snapshot["collision"] = {"available": False}
        return snapshot

    @staticmethod
    def _snapshot_pose(snapshot: dict[str, Any]) -> dict[str, Any] | None:
        pose = snapshot.get("pose")
        return pose if isinstance(pose, dict) else None

    @classmethod
    def _snapshot_yaw_deg(cls, snapshot: dict[str, Any]) -> float | None:
        pose = cls._snapshot_pose(snapshot)
        if pose is None:
            return None
        try:
            return math.degrees(float(pose["euler_rad"]["yaw"]))
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _exception_payload(exc: Exception) -> dict[str, str]:
        return {"error_type": type(exc).__name__, "error": str(exc)}

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        enum_value = getattr(value, "value", None)
        if enum_value is not None and enum_value is not value:
            return cls._plain_value(enum_value)
        return str(value)

    @classmethod
    def _collision_payload(cls, collision: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "available": True,
            "has_collided": bool(getattr(collision, "has_collided", False)),
        }
        for name in (
            "object_name",
            "object_id",
            "penetration_depth",
            "time_stamp",
        ):
            if hasattr(collision, name):
                payload[name] = cls._plain_value(getattr(collision, name))
        for name in ("position", "impact_point", "normal"):
            vector = getattr(collision, name, None)
            if vector is not None and all(
                hasattr(vector, axis) for axis in ("x_val", "y_val", "z_val")
            ):
                payload[name] = {
                    "x": float(vector.x_val),
                    "y": float(vector.y_val),
                    "z": float(vector.z_val),
                }
        return payload

    @staticmethod
    def _no_effect_failure_kind(snapshot: dict[str, Any]) -> str:
        collision = snapshot.get("collision")
        if isinstance(collision, dict) and collision.get("has_collided") is True:
            return "collision_blocked"
        return "control_no_effect"
