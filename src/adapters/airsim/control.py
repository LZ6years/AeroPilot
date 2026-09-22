"""Select and implement AirSim pose and paused-physics control strategies."""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol

from src.uav import body_velocity_to_world_delta, yaw_deg
from ..uav_primitives import PrimitiveResult
from .client import close_airsim_client
from .actions import PausedMultirotorActionExecutor

POSE_KINEMATIC_CONTROL = "pose_kinematic"
PAUSED_MULTIROTOR_CONTROL = "paused_multirotor"


def resolve_airsim_control_strategy(
    mode: str,
    configured: str | None = None,
) -> str:
    """Resolve and validate the two supported AirSim control contracts."""

    normalized_mode = str(mode).strip().lower()
    is_cv = normalized_mode in {"computervision", "computer_vision", "cv"}
    is_multirotor = normalized_mode == "multirotor"
    if not is_cv and not is_multirotor:
        raise ValueError(f"Unsupported AirSim mode: {mode}")
    inferred = POSE_KINEMATIC_CONTROL if is_cv else PAUSED_MULTIROTOR_CONTROL
    strategy = str(configured or inferred).strip().lower()
    expected = {
        POSE_KINEMATIC_CONTROL: True,
        PAUSED_MULTIROTOR_CONTROL: False,
    }
    if strategy not in expected:
        available = ", ".join(sorted(expected))
        raise ValueError(
            f"Unsupported AirSim control strategy {strategy!r}; "
            f"available: {available}"
        )
    if expected[strategy] != is_cv:
        expected_mode = (
            "ComputerVision"
            if strategy == POSE_KINEMATIC_CONTROL
            else "Multirotor"
        )
        raise ValueError(
            f"AirSim control strategy {strategy} requires mode={expected_mode}; "
            f"got {mode}"
        )
    return strategy


class AirSimControlStrategy(Protocol):
    """Define control-mode behavior while frame capture stays in the adapter."""

    def connect(self) -> Any: ...
    def reset(self, client: Any) -> PrimitiveResult: ...
    def takeoff(self, client: Any) -> PrimitiveResult: ...
    def hover(self, client: Any) -> PrimitiveResult: ...
    def land(self, client: Any) -> PrimitiveResult: ...
    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult: ...
    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult: ...
    def set_pose(
        self,
        client: Any,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ignore_collision: bool,
    ) -> PrimitiveResult: ...
    def get_pose(self, client: Any) -> PrimitiveResult: ...


class _AirSimStrategyBase:
    """Share endpoint, vehicle identity, and canonical pose conversion."""

    def __init__(
        self,
        host: str,
        port: int,
        vehicle_name: str,
        airsim_provider: Callable[[], Any],
    ) -> None:
        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self._airsim_provider = airsim_provider

    def _airsim(self) -> Any:
        return self._airsim_provider()

    def _pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
    ) -> Any:
        airsim = self._airsim()
        return airsim.Pose(
            airsim.Vector3r(x, y, -z),
            airsim.to_quaternion(
                math.radians(pitch_deg),
                math.radians(roll_deg),
                math.radians(yaw_value_deg),
            ),
        )

    def set_pose(
        self,
        client: Any,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ignore_collision: bool,
    ) -> PrimitiveResult:
        client.simSetVehiclePose(
            self._pose(x, y, z, yaw_value_deg, pitch_deg, roll_deg),
            ignore_collision=ignore_collision,
            vehicle_name=self.vehicle_name,
        )
        return PrimitiveResult(
            True,
            "AirSim UAV set_pose",
            {
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "euler_deg": {
                        "pitch": pitch_deg,
                        "roll": roll_deg,
                        "yaw": yaw_value_deg,
                    },
                    "ignore_collision": ignore_collision,
                }
            },
        )

    def _canonical_pose_result(
        self, pose: Any, message: str, velocity: dict[str, float] | None = None
    ) -> PrimitiveResult:
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        return PrimitiveResult(
            True,
            message,
            {
                "position": {
                    "x": pose.position.x_val,
                    "y": pose.position.y_val,
                    "z": -pose.position.z_val,
                },
                "euler_rad": {"pitch": pitch, "roll": roll, "yaw": yaw},
                "velocity": velocity or {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        )


class PoseKinematicControlStrategy(_AirSimStrategyBase):
    """Control AirSim ComputerVision by directly changing vehicle pose.

    This preserves the deterministic pose-action contract used by AerialVLN.
    It does not claim swept-path collision behavior between two poses.
    """

    def connect(self) -> Any:
        client = self._airsim().VehicleClient(
            ip=self.host, port=self.port, timeout_value=10
        )
        try:
            client.confirmConnection()
        except Exception as exc:
            close_airsim_client(client, original_error=exc)
            raise
        return client

    def reset(self, client: Any) -> PrimitiveResult:
        client.reset()
        return PrimitiveResult(True, "AirSim ComputerVision reset")

    def takeoff(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(False, "ComputerVision does not support takeoff")

    def hover(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(
            True,
            "AirSim ComputerVision pose held",
            {"pose": self.get_pose(client).data or {}},
        )

    def land(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(False, "ComputerVision does not support land")

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        delta = body_velocity_to_world_delta(
            vx, vy, vz, duration, math.degrees(yaw)
        )
        pose.position.x_val += delta["dx"]
        pose.position.y_val += delta["dy"]
        pose.position.z_val -= delta["dz"]
        client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )
        result = self._canonical_pose_result(
            pose, "AirSim ComputerVision move_body"
        )
        return PrimitiveResult(True, result.message, {"pose": result.data or {}})

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pitch, roll, yaw = self._airsim().to_eularian_angles(pose.orientation)
        pose.orientation = self._airsim().to_quaternion(
            pitch, roll, yaw + math.radians(yaw_delta_deg)
        )
        client.simSetVehiclePose(
            pose, ignore_collision=True, vehicle_name=self.vehicle_name
        )
        result = self._canonical_pose_result(
            pose, "AirSim ComputerVision turn_relative"
        )
        return PrimitiveResult(True, result.message, {"pose": result.data or {}})

    def get_pose(self, client: Any) -> PrimitiveResult:
        pose = client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        return self._canonical_pose_result(pose, "AirSim ComputerVision pose")


class PausedMultirotorControlStrategy(_AirSimStrategyBase):
    """Run physical Multirotor actions inside deterministic paused windows.

    Episode placement and action boundaries use full kinematics with zero
    velocity. Translation and yaw actions themselves execute while physics is
    unpaused, preserving AirSim collision handling without allowing idle drift
    between Runtime observations.
    """

    _PLACEMENT_REPETITIONS = 3
    def __init__(self, host: str, port: int, vehicle_name: str,
                 airsim_provider: Callable[[], Any]) -> None:
        super().__init__(host, port, vehicle_name, airsim_provider)
        self._actions = PausedMultirotorActionExecutor(
            vehicle_name, airsim_provider, self.get_pose,
            self._freeze_at_current_pose, self._canonical_pose_result,
        )

    def connect(self) -> Any:
        client = self._airsim().MultirotorClient(
            ip=self.host, port=self.port, timeout_value=10
        )
        try:
            client.confirmConnection()
            client.enableApiControl(True, self.vehicle_name)
            client.armDisarm(True, self.vehicle_name)
            client.simPause(True)
        except Exception as exc:
            close_airsim_client(client, original_error=exc)
            raise
        return client

    def reset(self, client: Any) -> PrimitiveResult:
        # AirSim reset is a simulator operation rather than an action window;
        # resume first so a paused server cannot retain the previous frame.
        # reset() revokes SimpleFlight API control and arm state. The adapter
        # reconnects immediately after this strategy call, which restores both;
        # pause here first so the reset vehicle cannot drift in that interval.
        client.simPause(False)
        client.reset()
        client.simPause(True)
        return PrimitiveResult(True, "AirSim paused Multirotor reset")

    def takeoff(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(
            False,
            "Paused Multirotor uses an episode initial pose instead of takeoff",
        )

    def hover(self, client: Any) -> PrimitiveResult:
        pose = self._freeze_at_current_pose(client)
        return PrimitiveResult(
            True,
            "AirSim paused Multirotor pose held",
            {"pose": pose.data or {}},
        )

    def land(self, client: Any) -> PrimitiveResult:
        return PrimitiveResult(
            False,
            "Paused Multirotor does not expose automatic land",
        )

    def set_pose(
        self,
        client: Any,
        x: float,
        y: float,
        z: float,
        yaw_value_deg: float,
        pitch_deg: float,
        roll_deg: float,
        ignore_collision: bool,
    ) -> PrimitiveResult:
        pose = self._pose(
            x,
            y,
            z,
            yaw_value_deg,
            pitch_deg,
            roll_deg,
        )
        client.simPause(True)
        for _ in range(self._PLACEMENT_REPETITIONS):
            client.simSetKinematics(
                self._zeroed_kinematics(pose.position, pose.orientation),
                ignore_collision=ignore_collision,
                vehicle_name=self.vehicle_name,
            )
            # Continue exactly one rendered physics frame, after which AirSim
            # pauses again. Reapplying the state prevents SimpleFlight's prior
            # controller target from pulling the dataset pose elsewhere.
            client.simContinueForFrames(1)
            client.simPause(True)
        # The last continued frame can accumulate a small gravity/controller
        # velocity before AirSim pauses. Clamp once more while already paused
        # so set_pose itself, not a later Runtime hover, establishes rest.
        actual = self._freeze_at_current_pose(client)
        return PrimitiveResult(
            True,
            "AirSim paused Multirotor set_pose",
            {
                "pose": actual.data or {},
                "requested_pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "euler_deg": {
                        "pitch": pitch_deg,
                        "roll": roll_deg,
                        "yaw": yaw_value_deg,
                    },
                    "ignore_collision": ignore_collision,
                },
            },
        )

    def move_body(
        self, client: Any, vx: float, vy: float, vz: float, duration: float
    ) -> PrimitiveResult:
        return self._actions.move_body(client, vx, vy, vz, duration)

    def turn_relative(self, client: Any, yaw_delta_deg: float) -> PrimitiveResult:
        return self._actions.turn_relative(client, yaw_delta_deg)

    def get_pose(self, client: Any) -> PrimitiveResult:
        state = client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        return self._canonical_pose_result(
            self._airsim().Pose(kin.position, kin.orientation),
            "AirSim UAV pose",
            {
                "x": kin.linear_velocity.x_val,
                "y": kin.linear_velocity.y_val,
                "z": -kin.linear_velocity.z_val,
            },
        )

    def _zeroed_kinematics(self, position: Any, orientation: Any) -> Any:
        """Build an independent zero-motion state at an existing pose."""

        airsim = self._airsim()
        state = airsim.KinematicsState()
        state.position = position
        state.orientation = orientation
        state.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
        state.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
        state.linear_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
        state.angular_acceleration = airsim.Vector3r(0.0, 0.0, 0.0)
        return state

    def _freeze_at_current_pose(self, client: Any) -> PrimitiveResult:
        """Clamp current pose to zero motion and leave simulation paused."""

        # Pause before the state read/write. Writing zero velocity while
        # physics is live leaves a race in which gravity or SimpleFlight can
        # advance one more tick before the later pause call.
        client.simPause(True)
        state = client.getMultirotorState(
            vehicle_name=self.vehicle_name
        ).kinematics_estimated
        client.simSetKinematics(
            self._zeroed_kinematics(
                state.position,
                state.orientation,
            ),
            ignore_collision=False,
            vehicle_name=self.vehicle_name,
        )
        return self.get_pose(client)
