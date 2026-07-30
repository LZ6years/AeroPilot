"""Test helper behavior for the manual real-engine instruction demo."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from demos.real_engine_instruction_demo import (
    action_sequence_from_events,
    action_windows_from_events,
    agent_messages_from_events,
    artifact_inventory_from_observations,
    build_state_summary,
    compact_summary,
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
        "current_phase_progress": {
            "phase": "move_forward",
            "tool_facts": {"move_duration_by_direction_success": {"forward": 2.5}},
        },
    }

    messages = [
        {
            "source": "phase_judge",
            "phase": "move_forward",
            "type": "final",
            "content": "已完成向前飞行。",
            "evidence_refs": [],
            "confidence": "high",
            "requires_user_reply": False,
        }
    ]
    summary = build_state_summary(tmp_path, "向前飞行5米，然后悬停", state, agent_messages=messages)

    assert summary["pose_delta"] == {"dx": None, "dy": None, "dz": None, "dyaw_deg": None}
    assert summary["observation_summary"]["has_pose"] is True
    assert summary["observation_summary"]["has_rgb"] is True
    assert summary["observation_summary"]["has_depth"] is True
    assert summary["phase_execution_summary"]["phase"] == "move_forward"
    assert summary["agent_messages"] == messages
    assert summary["artifact_inventory"]["counts"] == {"rgb": 1, "depth": 1, "other": 0}
    assert summary["artifact_inventory"]["rgb"][0]["exists"] is True


def test_real_engine_demo_action_sequence_reads_events():
    events = [
        {
            "event": "tool_step_completed",
            "payload": {
                "phase": "move_forward",
                "tool": "uav.move_body",
                "ok": True,
                "stabilization": {"status": "completed"},
            },
        }
    ]

    assert action_sequence_from_events(events) == [
        {"phase": "move_forward", "tool": "uav.move_body", "args": {}, "ok": True}
    ]


def test_real_engine_demo_action_sequence_prefers_debug_tool_call_args_over_compact_events():
    events = [
        {
            "event": "tool_step_completed",
            "payload": {
                "phase": "move_forward",
                "tool": "uav.move_body",
                "ok": True,
                "stabilization": {"status": "completed"},
            },
        },
        {
            "event": "tool_call_completed",
            "payload": {
                "phase": "move_forward",
                "tool": "uav.move_body",
                "args": {"direction": "forward", "duration": 1.0},
                "result": {"ok": True},
            },
        }
    ]

    assert action_sequence_from_events(events) == [
        {"phase": "move_forward", "tool": "uav.move_body", "args": {"direction": "forward", "duration": 1.0}, "ok": True}
    ]


def test_real_engine_demo_compact_summary_keeps_terminal_output_readable(tmp_path):
    summary = {
        "instruction": "向前飞行5米，然后悬停",
        "run_dir": str(tmp_path),
        "completed": True,
        "airborne": True,
        "tools": [{"phase": "move_forward", "tool": "uav.move_body", "args": {"duration": 2.5}, "ok": True}],
        "action_windows": [
            {
                "intent": "finish forward motion",
                "scope": "current_phase_only",
                "requires_intermediate_observation": False,
                "expected_effect": "forward displacement increases",
            }
        ],
        "pose_delta": {"dx": 5.0, "dy": 0.0, "dz": 0.0, "dyaw_deg": 0.0},
        "phase_execution_summary": {
            "phase": "move_forward",
            "phase_key": "1:move_forward",
            "window_count": 1,
            "max_window_count": 5,
            "measurements": {
                "forward_m": 4.9,
                "right_m": 0.0,
                "up_m": 0.0,
                "yaw_delta_deg": 0.0,
                "planar_m": 4.9,
            },
            "tool_facts": {
                "successful_steps": 1,
                "move_duration_by_direction_success": {"forward": 2.5},
                "turn_yaw_success_sum_deg": 0.0,
            },
        },
        "observation_summary": {"has_pose": True, "has_rgb": True, "has_depth": True, "sources": ["automatic_phase_window"]},
        "artifact_inventory": {"counts": {"rgb": 1, "depth": 1, "other": 0}},
        "shutdown_events": [],
    }

    compact = compact_summary(summary)

    assert compact["action_count"] == 1
    assert compact["action_window"]["intent"] == "finish forward motion"
    assert compact["action_window"]["expected_effect"] == "forward displacement increases"
    assert compact["phase_progress"]["measurements"]["forward_m"] == 4.9
    assert "world_pose" not in compact["phase_progress"]
    assert compact["inspect_files"]["planner_context"].endswith(str(Path("contexts") / "latest_context.json"))


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
            "connection": {"mode": "local"},
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


def test_real_engine_demo_action_windows_read_phase_events():
    events = [
        {
            "event": "phase_plan_ready",
            "payload": {
                "phase": "move_forward",
                "action_window": {
                    "intent": "move forward once",
                    "scope": "current_phase_only",
                    "requires_intermediate_observation": False,
                    "expected_effect": "forward displacement increases",
                },
            },
        },
        {
            "event": "phase_window_recorded",
            "payload": {
                "phase": "move_forward",
                "action_window": {
                    "intent": "move forward once",
                    "scope": "current_phase_only",
                    "requires_intermediate_observation": False,
                    "expected_effect": "forward displacement increases",
                },
            },
        },
    ]

    assert action_windows_from_events(events) == [
        {
            "intent": "move forward once",
            "scope": "current_phase_only",
            "requires_intermediate_observation": False,
            "expected_effect": "forward displacement increases",
        }
    ]


def test_real_engine_demo_agent_messages_read_events():
    events = [
        {
            "event": "agent_message",
            "payload": {
                "source": "action_planner",
                "phase": "move_forward",
                "message": {
                    "type": "progress",
                    "content": "我准备向前移动。",
                    "evidence_refs": [],
                    "confidence": "medium",
                    "requires_user_reply": False,
                },
            },
        }
    ]

    messages = agent_messages_from_events(events)

    assert messages[0]["source"] == "action_planner"
    assert messages[0]["content"] == "我准备向前移动。"
