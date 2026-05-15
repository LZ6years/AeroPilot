"""Test helper behavior for the manual real-engine instruction demo."""

from __future__ import annotations

from types import SimpleNamespace

from demos.test_real_engine_basic_instruction_demo import (
    action_sequence_from_events,
    artifact_inventory_from_observations,
    build_state_summary,
    write_common_config,
)
from src.env_client.schemas import MapPackage
from src.runtime.configs.action_config import UavActionConfig
from src.runtime.configs.model_config import ModelConfig
from src.runtime.configs.safety_config import UavSafetyConfig


class CaptureRecorder:
    def __init__(self) -> None:
        self.files = {}

    def write_json(self, name, data):
        self.files[name] = data


def test_real_engine_demo_summary_reports_pose_delta_and_artifacts(tmp_path):
    rgb_path = tmp_path / "artifacts" / "rgb" / "frame_000001.png"
    depth_path = tmp_path / "artifacts" / "depth" / "frame_000002.npy"
    rgb_path.parent.mkdir(parents=True)
    depth_path.parent.mkdir(parents=True)
    rgb_path.write_bytes(b"rgb")
    depth_path.write_bytes(b"depth")
    state = {
        "completed": True,
        "connected": True,
        "airborne": True,
        "latest_observation": {
            "source": "automatic_phase_window",
            "pose": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}, "orientation": {"yaw_deg": 0.0}},
            "frames": {
                "rgb": {"artifact_path": str(rgb_path), "width": 224, "height": 224},
                "depth": {"artifact_path": str(depth_path), "width": 224, "height": 224},
            },
        },
        "latest_pose": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}, "orientation": {"yaw_deg": 0.0}},
        "current_phase_progress": {
            "phase": "move_forward",
            "start_pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}, "orientation": {"yaw_deg": 0.0}},
            "cumulative_tool_facts": {"move_body_duration_by_direction_success": {"forward": 2.5}},
        },
    }

    summary = build_state_summary(tmp_path, "向前飞行5米，然后悬停", state)

    assert summary["pose_delta"]["dx"] == 5.0
    assert summary["pose_delta"]["dy"] == 0.0
    assert summary["observation_summary"]["has_pose"] is True
    assert summary["observation_summary"]["has_rgb"] is True
    assert summary["observation_summary"]["has_depth"] is True
    assert summary["phase_execution_summary"]["phase"] == "move_forward"
    assert summary["artifact_inventory"]["counts"] == {"rgb": 1, "depth": 1, "other": 0}
    assert summary["artifact_inventory"]["rgb"][0]["exists"] is True


def test_real_engine_demo_action_sequence_reads_events():
    events = [
        {
            "event": "step_completed",
            "payload": {
                "phase": "move_forward",
                "step": {"tool": "uav.move_body", "args": {"direction": "forward", "duration": 1.0}},
                "result": {"ok": True},
            },
        }
    ]

    assert action_sequence_from_events(events) == [
        {"phase": "move_forward", "tool": "uav.move_body", "args": {"direction": "forward", "duration": 1.0}, "ok": True}
    ]


def test_real_engine_demo_artifact_inventory_ignores_missing_artifact_metadata():
    inventory = artifact_inventory_from_observations(
        [
            {"frames": {"rgb": {"width": 224}}},
            {"frames": {"segmentation": {"artifact_path": "missing.bin"}}},
        ]
    )

    assert inventory["counts"] == {"rgb": 0, "depth": 0, "other": 1}
    assert inventory["other"][0]["exists"] is False


def test_real_engine_demo_config_summary_redacts_model_api_key():
    recorder = CaptureRecorder()
    selected_map = MapPackage.from_dict(
        {
            "map_id": "m1",
            "provider": "packaged_unreal",
            "scene_id": "1",
            "root": None,
            "launch": {"manual": True},
            "control": {"protocol": "airsim"},
            "airsim": {"vehicle_name": "Drone_1", "camera_name": "front_0"},
        }
    )
    model_config = ModelConfig(model_name="stub", api_key="secret-key", base_url="https://example.test/v1")
    args = SimpleNamespace(parser_mode="model", shutdown_policy="hover", separate_scenes=False, save_artifacts=True)

    write_common_config(
        recorder,
        selected_map,
        model_config,
        UavActionConfig(),
        UavSafetyConfig(),
        args,
    )

    model_summary = recorder.files["config/model_config.json"]["model_config"]
    assert "api_key" not in model_summary
    assert "secret-key" not in str(recorder.files["config/model_config.json"])
    assert model_summary["has_api_key"] is True
    assert recorder.files["config/model_config.json"]["save_artifacts"] is True
