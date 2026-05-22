"""Test interactive natural-language runtime helpers without launching UE."""

from __future__ import annotations

from pathlib import Path

from src.runtime.runners.interactive_runtime import (
    InteractiveTaskSession,
    build_turn_summary,
    compact_turn_summary,
)


class FakeScene:
    scene_id = "scene_a"

    def to_dict(self):
        return {"scene_id": self.scene_id, "status": "running"}


class FakeRecorder:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.events = []
        self.json_files = {}
        self.closed = False

    def event(self, event, payload):
        self.events.append((event, payload))

    def write_json(self, name, data):
        self.json_files[name] = data

    def close(self):
        self.closed = True


class FakeState:
    connected = True
    airborne = True
    completed = True
    current_phase_name = "move_forward"

    def to_dict(self):
        return {
            "completed": True,
            "connected": True,
            "airborne": True,
            "latest_observation": {
                "source": "automatic_phase_window",
                "pose": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}},
                "frames": {"rgb": {"artifact_path": "missing.png"}},
            },
            "current_phase_progress": {
                "world_pose": {
                    "start": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}},
                    "latest": {"position": {"x": 5.0, "y": 0.0, "z": -2.0}},
                },
                "cumulative_tool_facts": {"move_body_duration_success_sum": 2.5},
            },
        }


class FakeRuntime:
    def __init__(self, recorder):
        self.recorder = recorder
        self.shutdown_policy = "hover"
        self.run_calls = []
        self.hover_calls = []
        self.finish_calls = []
        self.ensure_calls = []
        self.messages = [
            {
                "source": "phase_judge",
                "phase": "move_forward",
                "type": "final",
                "content": "已完成本轮指令。",
                "evidence_refs": [],
                "confidence": "high",
                "requires_user_reply": False,
            }
        ]

    def run_task(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return FakeState()

    def safe_hover_reused_scene(self, *args):
        self.hover_calls.append(args)

    def finish_reused_scene(self, *args):
        self.finish_calls.append(args)

    def ensure_reused_scene(self, *args):
        self.ensure_calls.append(args)
        return FakeScene()

    def latest_turn_agent_messages(self):
        return [dict(item) for item in self.messages]


def test_build_turn_summary_reports_pose_delta_and_artifact_counts(tmp_path):
    summary = build_turn_summary(tmp_path, "向前飞行5米", FakeState().to_dict())

    assert summary["pose_delta"]["dx"] == 5.0
    assert summary["observation_summary"]["has_pose"] is True
    assert summary["observation_summary"]["has_rgb"] is True
    assert summary["artifact_inventory"]["counts"] == {"rgb": 1, "depth": 0, "other": 0}
    assert compact_turn_summary(summary)["artifact_counts"]["rgb"] == 1


def test_build_turn_summary_prefers_in_memory_agent_messages(tmp_path):
    messages = [
        {
            "source": "phase_judge",
            "phase": "move_forward",
            "type": "final",
            "content": "已完成本轮指令。",
            "evidence_refs": [],
            "confidence": "high",
            "requires_user_reply": False,
        }
    ]

    summary = build_turn_summary(tmp_path, "向前飞行5米", FakeState().to_dict(), agent_messages=messages)

    assert summary["agent_messages"] == messages


def test_interactive_session_reuses_scene_for_natural_language_turn(tmp_path):
    recorder = FakeRecorder(tmp_path)
    runtime = FakeRuntime(recorder)
    live_events = []
    live_sink = live_events.append
    session = InteractiveTaskSession(runtime, "m1", agent_id="uav_1", scene_id="scene_a", event_sink=live_sink)

    summary = session.submit_instruction("向前飞行5米")

    assert summary["completed"] is True
    assert summary["agent_messages"][0]["content"] == "已完成本轮指令。"
    assert runtime.run_calls[0][0][:2] == ("m1", "向前飞行5米")
    assert runtime.run_calls[0][1]["scene_policy"] == "reuse_running"
    assert runtime.run_calls[0][1]["stop_scene_on_finish"] is False
    assert runtime.run_calls[0][1]["scene_id"] == "scene_a"
    assert session.event_sink is live_sink
    assert "interactive/latest_summary.json" in recorder.json_files
    assert recorder.events[-1][0] == "interactive_instruction_completed"


def test_interactive_session_can_start_scene_before_first_instruction(tmp_path):
    recorder = FakeRecorder(tmp_path)
    runtime = FakeRuntime(recorder)
    session = InteractiveTaskSession(runtime, "m1", scene_id="scene_a")

    result = session.start_scene()

    assert result["ok"] is True
    assert runtime.ensure_calls == [("m1", "scene_a")]
    assert recorder.events[-1][0] == "interactive_scene_ready"


def test_interactive_session_shutdown_can_override_policy(tmp_path):
    recorder = FakeRecorder(tmp_path)
    runtime = FakeRuntime(recorder)
    session = InteractiveTaskSession(runtime, "m1")

    result = session.shutdown("land")

    assert result["shutdown_policy"] == "land"
    assert runtime.finish_calls
    assert runtime.shutdown_policy == "hover"
    assert recorder.closed is True
    assert session.closed is True
