"""Diagnostic snapshots tolerate unavailable SDK fields without losing others."""
from types import SimpleNamespace

from tests.manual.travel_control_state_probe import independent_snapshot


def test_partial_sdk_does_not_abort_independent_snapshot():
    sdk = SimpleNamespace(
        getMultirotorState=lambda **_: SimpleNamespace(timestamp=42),
        simGetGroundTruthKinematics=lambda **_: SimpleNamespace(position=SimpleNamespace(x_val=1.0)),
        simGetVehiclePose=lambda **_: SimpleNamespace(position=SimpleNamespace(x_val=2.0)),
        simGetCameraInfo=lambda *_, **__: SimpleNamespace(fov=90),
        simGetCollisionInfo=lambda **_: SimpleNamespace(has_collided=False),
    )
    result = independent_snapshot(sdk)
    assert result["estimated"]["timestamp"] == 42
    assert result["physics"]["position"]["x_val"] == 1.0
    assert result["rendered_pose"]["position"]["x_val"] == 2.0
    assert "AttributeError" in result["paused"]["error"]
