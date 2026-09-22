"""Execute and verify paused AirSim action windows, including failure cleanup."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.uav.frames import normalize_yaw_deg, normalize_yaw_delta_deg, position_distance_m, yaw_deg
from ..uav_primitives import PrimitiveResult


@dataclass(frozen=True)
class _PhysicalActionOutcome:
    """Capture one paused-Multirotor RPC window without losing failure detail."""

    before: dict[str, Any]
    after: dict[str, Any]
    rpc_result: Any
    rpc_error: dict[str, str] | None
    freeze_error: dict[str, str] | None
    elapsed_sec: float

    @property
    def failure_kind(self) -> str | None:
        if self.rpc_error is not None:
            return "rpc_error"
        if self.rpc_result is False:
            return "controller_timeout_or_rejected"
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
                 canonical_pose_result: Callable[..., PrimitiveResult]) -> None:
        self.vehicle_name = vehicle_name
        self._airsim = airsim_provider
        self.get_pose = get_pose
        self._freeze_at_current_pose = freeze
        self._canonical_pose_result = canonical_pose_result

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        if vx == 0.0 and vy == 0.0 and vz == 0.0:
            # Runtime stabilization expresses braking as a zero-velocity move.
            # Reopening physics here would recreate the drift this strategy is
            # designed to eliminate, so freeze immediately and ignore duration.
            pose = self._freeze_at_current_pose(client)
            return PrimitiveResult(
                True,
                "AirSim paused Multirotor move_body brake",
                {"pose": pose.data or {}},
            )

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
        if failure_kind is None and actual_distance is None:
            failure_kind = "rpc_error"
            tracking["verification_error"] = "post-action pose unavailable"
        elif (
            failure_kind is None
            and expected_distance >= self._NO_EFFECT_MIN_EXPECTED_TRANSLATION_M
            and actual_distance < self._NO_EFFECT_MAX_OBSERVED_TRANSLATION_M
        ):
            failure_kind = self._no_effect_failure_kind(outcome.after)
        return self._physical_result(
            action_name="move_body",
            outcome=outcome,
            failure_kind=failure_kind,
            requested={
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "duration": duration,
            },
            tracking=tracking,
        )

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        pose = self.get_pose(client)
        current_yaw = yaw_deg(pose.data or {})
        target_yaw = normalize_yaw_deg(current_yaw + yaw_delta_deg)
        outcome = self._run_physical_action(
            client,
            lambda: client.rotateToYawAsync(
                target_yaw,
                vehicle_name=self.vehicle_name,
            ),
        )
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
        }
        failure_kind = outcome.failure_kind
        if failure_kind is None and observed_delta is None:
            failure_kind = "rpc_error"
            tracking["verification_error"] = "post-action yaw unavailable"
        elif (
            failure_kind is None
            and abs(yaw_delta_deg) >= self._NO_EFFECT_MIN_REQUESTED_YAW_DEG
            and abs(observed_delta) < self._NO_EFFECT_MAX_OBSERVED_YAW_DEG
        ):
            failure_kind = self._no_effect_failure_kind(outcome.after)
        return self._physical_result(
            action_name="turn_relative",
            outcome=outcome,
            failure_kind=failure_kind,
            requested={"yaw_delta_deg": yaw_delta_deg},
            tracking=tracking,
        )

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

        before = self._multirotor_snapshot(client)
        rpc_result: Any = None
        rpc_error: dict[str, str] | None = None
        freeze_error: dict[str, str] | None = None
        started = time.monotonic()
        try:
            client.simPause(False)
            future = action()
            get_result = getattr(future, "get", None)
            if not callable(get_result):
                raise RuntimeError("AirSim async future does not expose get()")
            rpc_result = get_result()
        except Exception as exc:
            rpc_error = self._exception_payload(exc)
        finally:
            elapsed_sec = time.monotonic() - started
            try:
                self._freeze_at_current_pose(client)
            except Exception as exc:
                freeze_error = self._exception_payload(exc)
                # Best effort: a failed pose clamp must still leave physics
                # paused so an RPC error cannot turn into uncontrolled drift.
                try:
                    client.simPause(True)
                except Exception:
                    pass
        after = self._multirotor_snapshot(client)
        return _PhysicalActionOutcome(
            before=before,
            after=after,
            rpc_result=rpc_result,
            rpc_error=rpc_error,
            freeze_error=freeze_error,
            elapsed_sec=elapsed_sec,
        )

    def _physical_result(
        self,
        *,
        action_name: str,
        outcome: _PhysicalActionOutcome,
        failure_kind: str | None,
        requested: dict[str, Any],
        tracking: dict[str, Any],
    ) -> PrimitiveResult:
        """Convert one measured action window into the primitive result contract."""

        diagnostics = {
            "action": action_name,
            "requested": requested,
            "elapsed_sec": outcome.elapsed_sec,
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
        }
        ok = failure_kind is None
        message = f"AirSim paused Multirotor {action_name}"
        if not ok:
            message = f"{message} failed: {failure_kind}"
        return PrimitiveResult(
            ok,
            message,
            {
                "pose": self._snapshot_pose(outcome.after) or {},
                "failure_kind": failure_kind,
                "action_diagnostics": diagnostics,
            },
        )

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
