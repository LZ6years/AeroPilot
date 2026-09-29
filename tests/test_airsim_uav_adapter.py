"""Test AirSimUavAdapter behavior without launching a real AirSim scene."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.adapters.airsim.uav import AirSimUavAdapter


from tests.airsim_fakes import FakeAirSim, FakeAirSimClient, FakeImageResponse


def test_pose_kinematic_set_pose_calls_airsim_sim_set_vehicle_pose(monkeypatch):
    # Verify set_pose translates FlyAgent x/y/z/yaw inputs into an AirSim pose call.
    adapter = AirSimUavAdapter(
        vehicle_name="Drone1",
        mode="ComputerVision",
        control_strategy="pose_kinematic",
    )
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.set_pose(1.0, 2.0, -3.0, yaw_deg=90.0, ignore_collision=False)

    assert result.ok
    assert result.data["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": -3.0}
    assert result.data["pose"]["euler_deg"]["yaw"] == 90.0
    assert len(client.set_pose_calls) == 1
    call = client.set_pose_calls[0]
    assert call["vehicle_name"] == "Drone1"
    assert call["ignore_collision"] is False
    assert call["pose"].position.x_val == 1.0
    assert call["pose"].position.y_val == 2.0
    assert call["pose"].position.z_val == 3.0
    assert round(call["pose"].orientation[2], 6) == 1.570796


def test_ascend_and_descend_use_speed_and_duration(monkeypatch):
    # Verify vertical primitives call AirSim body velocity with correct NED z signs.
    adapter = AirSimUavAdapter(vehicle_name="Drone1")
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    ascend = adapter.ascend(0.5, 4.0)
    descend = adapter.descend(0.25, 2.0)

    assert ascend.ok
    assert descend.ok
    assert client.move_body_calls[0] == {
        "vx": 0.0,
        "vy": 0.0,
        "vz": -0.5,
        "duration": 4.0,
        "vehicle_name": "Drone1",
    }
    assert client.move_body_calls[1] == {
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.25,
        "duration": 2.0,
        "vehicle_name": "Drone1",
    }
    assert client.pause_calls == [False, True, False, True]


def test_paused_multirotor_set_pose_repeats_zero_kinematics_and_reads_back(
    monkeypatch,
):
    adapter = AirSimUavAdapter(
        vehicle_name="Drone1",
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.set_pose(
        1.0,
        2.0,
        3.0,
        yaw_deg=90.0,
        ignore_collision=True,
    )

    assert result.ok
    assert result.data["pose"]["position"] == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
    }
    assert len(client.set_kinematics_calls) == 4
    assert client.continue_frame_calls == [1, 1, 1]
    assert client.pause_calls == [True, True, True, True, True]
    assert result.data["pose"]["velocity"] == {
        "x": 0.0,
        "y": 0.0,
        "z": -0.0,
    }
    for call in client.set_kinematics_calls[:3]:
        state = call["state"]
        assert call["ignore_collision"] is True
        assert state.linear_velocity.x_val == 0.0
        assert state.angular_velocity.z_val == 0.0
    assert client.set_kinematics_calls[-1]["ignore_collision"] is False


def test_paused_multirotor_zero_velocity_brake_never_unpauses(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.move_body(0.0, 0.0, 0.0, 2.0)

    assert result.ok
    assert client.move_body_calls == []
    assert client.pause_calls == [True]


def test_paused_multirotor_connect_arms_then_pauses(monkeypatch):
    client = FakeAirSimClient()
    monkeypatch.setattr(
        FakeAirSim,
        "MultirotorClient",
        lambda **kwargs: client,
        raising=False,
    )
    adapter = AirSimUavAdapter(
        vehicle_name="Drone1",
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.connect()

    assert result.ok
    assert result.data["control_strategy"] == "paused_multirotor"
    assert client.connection_calls == [
        "confirm",
        ("api", True, "Drone1"),
        ("arm", True, "Drone1"),
    ]
    assert client.pause_calls == [True]


@pytest.mark.parametrize("failure", ["api", "arm"])
def test_paused_multirotor_connect_rejects_failed_control_handshake(monkeypatch, failure):
    client = FakeAirSimClient()
    if failure == "api":
        monkeypatch.setattr(client, "isApiControlEnabled", lambda name: False)
    else:
        monkeypatch.setattr(client, "armDisarm", lambda armed, name: False)
    monkeypatch.setattr(FakeAirSim, "MultirotorClient", lambda **kwargs: client, raising=False)
    adapter = AirSimUavAdapter(vehicle_name="Drone1", mode="Multirotor",
                              control_strategy="paused_multirotor")
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)
    with pytest.raises(RuntimeError, match="API control|arming"):
        adapter.connect()
    assert adapter._client is None


def test_paused_multirotor_action_failure_still_pauses(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.future_error = RuntimeError("motion failed")
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.move_body(1.0, 0.0, 0.0, 1.0)

    assert not result.ok
    assert result.data["failure_kind"] == "rpc_error"
    assert result.data["action_diagnostics"]["rpc"]["error"] == {
        "error_type": "RuntimeError",
        "error": "motion failed",
    }
    assert client.pause_calls == [False, True, True]


def test_paused_multirotor_false_rpc_result_is_not_reported_as_success(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.future_result = False
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.turn_relative(30.0)

    assert not result.ok
    assert result.data["failure_kind"] == "controller_timeout_or_rejected"
    assert result.data["action_diagnostics"]["rpc"]["result"] is False
    assert client.pause_calls == [False, True, True]


def test_paused_multirotor_turn_wraps_target_and_measures_crossing(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.kinematics.orientation = (0.0, 0.0, np.deg2rad(170.0))
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.turn_relative(30.0)

    tracking = result.data["action_diagnostics"]["tracking"]
    assert result.ok
    assert client.rotate_calls[0]["yaw"] == pytest.approx(-160.0)
    assert tracking["observed_yaw_delta_deg"] == pytest.approx(30.0)
    assert tracking["target_yaw_error_deg"] == pytest.approx(0.0)


def test_paused_multirotor_turn_no_effect_is_failure(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.action_effect_scale = 0.0
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.turn_relative(-79.0)

    assert not result.ok
    assert result.data["failure_kind"] == "control_no_effect"
    tracking = result.data["action_diagnostics"]["tracking"]
    assert tracking["observed_yaw_delta_deg"] == pytest.approx(0.0)


def test_paused_multirotor_move_no_effect_reports_collision(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.action_effect_scale = 0.0
    client.collision.has_collided = True
    client.collision.object_name = "Building_42"
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.move_body(2.0, 0.5, 0.0, 1.5)

    assert not result.ok
    assert result.data["failure_kind"] == "collision_blocked"
    diagnostics = result.data["action_diagnostics"]
    assert diagnostics["tracking"]["observed_translation_m"] == pytest.approx(0.0)
    assert diagnostics["after"]["collision"]["object_name"] == "Building_42"


def test_paused_multirotor_negligible_partial_motion_is_rejected(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.action_effect_scale = 0.05
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.move_body(2.0, 0.0, 0.0, 1.0)

    assert not result.ok
    assert result.data["failure_kind"] == "control_insufficient_progress"
    assert result.data["action_diagnostics"]["tracking"][
        "observed_translation_m"
    ] == pytest.approx(0.1)


def test_paused_multirotor_preserves_rpc_and_freeze_errors(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.future_error = RuntimeError("rpc timed out")
    client.freeze_error = RuntimeError("freeze failed")
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.move_body(1.0, 0.0, 0.0, 1.0)

    assert not result.ok
    assert result.data["failure_kind"] == "rpc_error"
    diagnostics = result.data["action_diagnostics"]
    assert diagnostics["rpc"]["error"]["error"] == "rpc timed out"
    assert diagnostics["freeze"]["error"]["error"] == "freeze failed"
    assert client.pause_calls == [False, True, True, True]


def test_paused_multirotor_freeze_failure_has_distinct_failure_kind(monkeypatch):
    adapter = AirSimUavAdapter(
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    client.freeze_error = RuntimeError("freeze failed")
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.turn_relative(15.0)

    assert not result.ok
    assert result.data["failure_kind"] == "freeze_cleanup_error"
    diagnostics = result.data["action_diagnostics"]
    assert diagnostics["rpc"]["result"] is True
    assert diagnostics["freeze"]["error"]["error"] == "freeze failed"


def test_paused_multirotor_reset_restores_control_and_pause(monkeypatch):
    adapter = AirSimUavAdapter(
        vehicle_name="Drone1",
        mode="Multirotor",
        control_strategy="paused_multirotor",
    )
    client = FakeAirSimClient()
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)
    monkeypatch.setattr(
        FakeAirSim,
        "MultirotorClient",
        lambda **kwargs: client,
        raising=False,
    )

    result = adapter.reset()

    assert result.ok
    assert client.reset_calls == 1
    assert client.connection_calls == [
        "confirm",
        ("api", True, "Drone1"),
        ("arm", True, "Drone1"),
    ]
    assert client.pause_calls == [False, True, True]


def test_frame_artifacts_do_not_overwrite_existing_files(tmp_path):
    # Separate-scene real runs recreate adapters, so artifact naming must scan
    # the run directory instead of trusting the in-memory frame counter.
    adapter = AirSimUavAdapter(artifact_dir=tmp_path / "artifacts", save_artifacts=True)
    existing_rgb = tmp_path / "artifacts" / "rgb" / "frame_000001.png"
    existing_depth = tmp_path / "artifacts" / "depth" / "frame_000001.npy"
    existing_rgb.parent.mkdir(parents=True)
    existing_depth.parent.mkdir(parents=True)
    existing_rgb.write_bytes(b"old-rgb")
    existing_depth.write_bytes(b"old-depth")

    rgb_payload = bytes(range(12))
    rgb_path = _saved_frame(adapter,
        "rgb", FakeImageResponse(image_data_uint8=rgb_payload)
    )
    adapter.capture._frame_index = 0
    depth_path = _saved_frame(adapter, "depth", FakeImageResponse(image_data_float=[1.0, 2.0, 3.0, 4.0]))

    assert rgb_path == tmp_path / "artifacts" / "rgb" / "frame_000002.png"
    assert depth_path == tmp_path / "artifacts" / "depth" / "frame_000002.npy"
    assert existing_rgb.read_bytes() == b"old-rgb"
    assert existing_depth.read_bytes() == b"old-depth"
    with Image.open(rgb_path) as image:
        assert image.size == (2, 2)
        assert image.mode == "RGB"
    assert np.load(depth_path).shape == (2, 2)


def test_observation_profile_routes_depth_vis_without_map_special_case(
    monkeypatch,
):
    adapter = AirSimUavAdapter(camera_name="front")
    client = FakeAirSimClient()
    client.image_responses = [
        FakeImageResponse(image_data_uint8=bytes(range(12))),
        _depth_vis_response(),
    ]
    adapter._client = client
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    result = adapter.get_frames(
        ["rgb", "depth"],
        {"rgb": "scene", "depth": "depth_vis"},
    )

    assert result.ok
    rgb_request, depth_request = client.image_requests
    assert (
        rgb_request.image_type,
        rgb_request.pixels_as_float,
        rgb_request.compress,
    ) == (FakeAirSim.ImageType.Scene, False, False)
    assert (
        depth_request.image_type,
        depth_request.pixels_as_float,
        depth_request.compress,
    ) == (FakeAirSim.ImageType.DepthVis, False, True)
    assert result.data["frames"]["depth"]["capture_source"] == "depth_vis"


def test_artifact_setting_does_not_change_airsim_request(monkeypatch, tmp_path):
    requests = []
    for save_artifacts in (False, True):
        adapter = AirSimUavAdapter(
            artifact_dir=tmp_path / str(save_artifacts),
            save_artifacts=save_artifacts,
        )
        client = FakeAirSimClient()
        client.image_responses = [
            FakeImageResponse(image_data_uint8=bytes(range(12))),
            FakeImageResponse(image_data_float=[1.0, 2.0, 3.0, 4.0]),
        ]
        adapter._client = client
        monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

        adapter.get_frames(["rgb", "depth"])
        requests.append(
            [
                (item.image_type, item.pixels_as_float, item.compress)
                for item in client.image_requests
            ]
        )

    assert requests[0] == requests[1] == [
        (FakeAirSim.ImageType.Scene, False, False),
        (FakeAirSim.ImageType.DepthPerspective, True, False),
    ]


def test_depth_vis_saves_normalized_array_and_preview(tmp_path):
    adapter = AirSimUavAdapter(
        artifact_dir=tmp_path / "artifacts",
        save_artifacts=True,
    )

    path = _saved_frame(adapter,
        "depth",
        _depth_vis_response(),
        "depth_vis",
    )

    values = np.load(path)
    assert values.shape == (2, 2)
    assert values.min() == 0.0
    assert values.max() == 1.0
    assert adapter.capture._depth_preview_path(path).is_file()


def test_depth_vis_accepts_float_payload_from_compatible_servers(tmp_path):
    adapter = AirSimUavAdapter(
        artifact_dir=tmp_path / "artifacts",
        save_artifacts=True,
    )

    path = _saved_frame(adapter,
        "depth",
        FakeImageResponse(image_data_float=[0.0, 0.25, 0.5, 1.0]),
        "depth_vis",
    )

    assert np.load(path).shape == (2, 2)
    assert adapter.capture._depth_preview_path(path).is_file()


def test_unknown_capture_source_fails_before_rpc(monkeypatch):
    adapter = AirSimUavAdapter()
    adapter._client = FakeAirSimClient()
    monkeypatch.setattr(adapter, "_airsim", lambda: FakeAirSim)

    with pytest.raises(ValueError, match="Unsupported AirSim capture source"):
        adapter.get_frames(["depth"], {"depth": "benchmark_private_depth"})


def _depth_vis_response() -> FakeImageResponse:
    """Return a small compressed DepthVis-like PNG with varying green values."""

    pixels = np.array(
        [
            [[0, 0, 0], [0, 85, 0]],
            [[0, 170, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    buffer = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return FakeImageResponse(image_data_uint8=buffer.getvalue())


def _saved_frame(adapter, modality, response, source=None):
    source = source or adapter.capture._capture_source(modality, None)
    result = adapter.capture._frame_metadata(modality, source, adapter.capture._capture_spec(source), response)
    return Path(result["artifact_path"])
