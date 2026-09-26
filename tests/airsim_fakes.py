"""Small SDK doubles shared by AirSim adapter and action-executor tests."""

from types import SimpleNamespace
import numpy as np

class FakeAirSim:
    # Provide the AirSim API surface used by pose and frame tests.
    class ImageType:
        Scene = 0
        DepthVis = 3
        DepthPerspective = 2
        Segmentation = 5

    class ImageRequest:
        def __init__(
            self,
            camera_name: str,
            image_type: int,
            pixels_as_float: bool,
            compress: bool,
        ) -> None:
            self.camera_name = camera_name
            self.image_type = image_type
            self.pixels_as_float = pixels_as_float
            self.compress = compress

    class Vector3r:
        def __init__(self, x_val: float, y_val: float, z_val: float) -> None:
            # Store vector coordinates with AirSim-compatible attribute names.
            self.x_val = x_val
            self.y_val = y_val
            self.z_val = z_val

    class Pose:
        def __init__(self, position: "FakeAirSim.Vector3r", orientation: tuple[float, float, float]) -> None:
            # Store position and orientation exactly like a lightweight AirSim Pose.
            self.position = position
            self.orientation = orientation

    class KinematicsState:
        def __init__(self) -> None:
            self.position = FakeAirSim.Vector3r(0.0, 0.0, 0.0)
            self.orientation = (0.0, 0.0, 0.0)
            self.linear_velocity = FakeAirSim.Vector3r(0.0, 0.0, 0.0)
            self.angular_velocity = FakeAirSim.Vector3r(0.0, 0.0, 0.0)
            self.linear_acceleration = FakeAirSim.Vector3r(0.0, 0.0, 0.0)
            self.angular_acceleration = FakeAirSim.Vector3r(0.0, 0.0, 0.0)

    @staticmethod
    def to_quaternion(pitch: float, roll: float, yaw: float) -> tuple[float, float, float]:
        # Return radians directly so the test can inspect degree-to-radian conversion.
        return (pitch, roll, yaw)

    @staticmethod
    def to_eularian_angles(orientation):
        return orientation


class FakeAirSimClient:
    def __init__(self) -> None:
        # Capture simSetVehiclePose calls for assertions.
        self.set_pose_calls = []
        self.move_body_calls = []
        self.image_requests = []
        self.image_responses = []
        self.pause_calls = []
        self.continue_frame_calls = []
        self.set_kinematics_calls = []
        self.rotate_calls = []
        self.cancel_calls = []
        self.cancel_error = None
        self.kinematics = FakeAirSim.KinematicsState()
        self.connection_calls = []
        self.reset_calls = 0
        self.future_result = True
        self.future_error = None
        self.action_effect_scale = 1.0
        self.freeze_error = None
        self.collision = SimpleNamespace(
            has_collided=False,
            object_name="",
            object_id=-1,
            penetration_depth=0.0,
            time_stamp=0,
        )

    def confirmConnection(self) -> None:
        self.connection_calls.append("confirm")

    def enableApiControl(self, enabled: bool, vehicle_name: str) -> None:
        self.connection_calls.append(("api", enabled, vehicle_name))

    def armDisarm(self, armed: bool, vehicle_name: str) -> None:
        self.connection_calls.append(("arm", armed, vehicle_name))

    def reset(self) -> None:
        self.reset_calls += 1

    def simSetVehiclePose(self, pose, ignore_collision: bool, vehicle_name: str) -> None:
        # Record the pose placement request that would have gone to AirSim.
        self.set_pose_calls.append(
            {
                "pose": pose,
                "ignore_collision": ignore_collision,
                "vehicle_name": vehicle_name,
            }
        )

    def moveByVelocityBodyFrameAsync(self, vx: float, vy: float, vz: float, duration: float, vehicle_name: str):
        # Record the call and defer its effect until Future.get(), matching
        # msgpackrpc instead of making join() responsible for RPC errors.
        self.move_body_calls.append(
            {
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "duration": duration,
                "vehicle_name": vehicle_name,
            }
        )
        def effect() -> None:
            scale = self.action_effect_scale
            self.kinematics.position.x_val += vx * duration * scale
            self.kinematics.position.y_val += vy * duration * scale
            self.kinematics.position.z_val += vz * duration * scale

        return FakeAsyncFuture(self, effect)

    def cancelLastTask(self, vehicle_name: str):
        self.cancel_calls.append(vehicle_name)
        if self.cancel_error:
            raise self.cancel_error

    def rotateToYawAsync(self, yaw: float, vehicle_name: str, timeout_sec=3e38, margin=5):
        self.rotate_calls.append({"yaw": yaw, "vehicle_name": vehicle_name})
        self.rotate_options = {"timeout_sec": timeout_sec, "margin": margin}

        def effect() -> None:
            if self.action_effect_scale == 0.0:
                return
            current = float(self.kinematics.orientation[2])
            target = np.deg2rad(yaw)
            self.kinematics.orientation = (
                0.0,
                0.0,
                current + (target - current) * self.action_effect_scale,
            )

        return FakeAsyncFuture(self, effect)

    def simPause(self, paused: bool) -> None:
        self.pause_calls.append(paused)

    def simContinueForFrames(self, frames: int) -> None:
        self.continue_frame_calls.append(frames)

    def simSetKinematics(
        self,
        state,
        ignore_collision: bool,
        vehicle_name: str,
    ) -> None:
        if self.freeze_error is not None:
            raise self.freeze_error
        self.kinematics = state
        self.set_kinematics_calls.append(
            {
                "state": state,
                "ignore_collision": ignore_collision,
                "vehicle_name": vehicle_name,
            }
        )

    def getMultirotorState(self, vehicle_name: str):
        return SimpleNamespace(
            kinematics_estimated=self.kinematics,
            landed_state=1,
            ready=True,
            ready_message="ready",
            can_arm=True,
            timestamp=123,
        )

    def simGetCollisionInfo(self, vehicle_name: str):
        return self.collision

    def simGetImages(self, requests, vehicle_name: str):
        self.image_requests = list(requests)
        return list(self.image_responses)


class FakeAsyncFuture:
    """Model real msgpackrpc semantics: join waits, get raises/returns."""

    def __init__(self, client: FakeAirSimClient, effect) -> None:
        self.client = client
        self.effect = effect
        self.joined = False

    def join(self) -> None:
        self.joined = True

    def get(self):
        self.join()
        if self.client.future_error is not None:
            raise self.client.future_error
        if self.client.future_result is not False:
            self.effect()
        return self.client.future_result


class FakeImageResponse:
    def __init__(
        self,
        image_data_uint8: bytes = b"",
        image_data_float: list[float] | None = None,
        *,
        width: int = 2,
        height: int = 2,
    ) -> None:
        self.width = width
        self.height = height
        self.image_data_uint8 = image_data_uint8
        self.image_data_float = image_data_float or []
