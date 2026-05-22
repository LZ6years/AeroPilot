"""Test the interactive CLI command layer without creating a real runtime."""

from __future__ import annotations

import json
from pathlib import Path

from src.cli.display import CliNarrativePanel, print_summary
from src.cli.main import build_parser, default_chat_map_id, handle_slash_command


class FakeCliSession:
    """Small test double for slash commands.

    The CLI command layer should only call the public session API. Keeping this
    fake tiny makes the test protect the shell behavior without pulling in UE,
    AirSim, model config, or a real TaskRuntime.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.map_id = "m1"
        self.available_map_ids = ["m1", "cityenviron_win64"]
        self.last_summary = {
            "instruction": "向前飞行5米",
            "completed": True,
            "agent_messages": [
                {
                    "type": "progress",
                    "content": "我准备向前移动并保持安全悬停。",
                    "confidence": "medium",
                    "requires_user_reply": False,
                }
            ],
        }
        self.shutdown_calls: list[str | None] = []
        self.hover_calls = 0

    def status(self) -> dict:
        return {"map_id": "m1", "closed": False, "current_phase": "move_forward"}

    def hover(self) -> dict:
        self.hover_calls += 1
        return {"ok": True, "tool": "uav.hover"}

    def shutdown(self, shutdown_policy: str | None = None) -> dict:
        self.shutdown_calls.append(shutdown_policy)
        return {"closed": True, "shutdown_policy": shutdown_policy or "default"}


def test_status_command_prints_session_status(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/status", session)  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert should_continue is True
    assert '"map_id": "m1"' in out
    assert '"current_phase": "move_forward"' in out


def test_context_command_prints_debug_artifact_paths(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/context", session)  # type: ignore[arg-type]

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert should_continue is True
    assert payload["latest_context"] == str(tmp_path / "contexts" / "latest_context.json")
    assert payload["latest_state"] == str(tmp_path / "state" / "latest_state.json")
    assert payload["flight_memory"] == str(tmp_path / "state" / "flight_memory.json")
    assert payload["semantic_plan"] == str(tmp_path / "state" / "semantic_plan.json")
    assert payload["latest_observation_debug"] == str(tmp_path / "debug" / "latest_observation.json")


def test_maps_command_prints_available_map_ids(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/maps", session)  # type: ignore[arg-type]

    payload = json.loads(capsys.readouterr().out)
    assert should_continue is True
    assert payload["current_map_id"] == "m1"
    assert "cityenviron_win64" in payload["available_map_ids"]


def test_help_command_describes_clear_command(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/help", session)  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert should_continue is True
    assert "/clear" in out
    assert "向前飞行5米" in out


def test_summary_command_prints_agent_messages(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/summary", session)  # type: ignore[arg-type]

    out = capsys.readouterr().out
    assert should_continue is True
    assert "我准备向前移动并保持安全悬停。" in out
    assert '"agent_messages"' in out


def test_default_summary_is_compact_after_live_messages(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    print_summary(session.last_summary)

    out = capsys.readouterr().out
    assert "我准备向前移动并保持安全悬停。" not in out
    assert '"agent_messages"' not in out
    assert '"completed": true' in out


def test_cli_narrative_panel_prints_steps_status_and_final(capsys):
    panel = CliNarrativePanel()

    summary = {
        "completed": True,
        "run_dir": "runs/interactive_cli_test",
        "agent_messages": [{"type": "final", "content": "任务已完成。"}],
    }

    panel({"kind": "instruction_started", "payload": {"phase_names": ["move_forward", "rotate_right"]}})
    panel(
        {
            "kind": "agent_message",
            "source": "action_planner",
            "phase": "move_forward",
            "message": {
                "type": "progress",
                "content": "我准备向前飞行。",
                "confidence": "medium",
                "requires_user_reply": False,
            },
        }
    )
    panel(
        {
            "kind": "tool_step_completed",
            "payload": {
                "phase": "move_forward",
                "tool": "uav.move_body",
                "ok": True,
                "stabilization": {"status": "completed"},
            },
        }
    )
    panel(
        {
            "kind": "observation_updated",
            "payload": {
                "source": "automatic_phase_window",
                "frame_refs": {"rgb": "memory/frames.jsonl#1", "depth": "memory/frames.jsonl#2"},
                "has_pose": True,
            },
        }
    )
    panel({"kind": "phase_transition_applied", "payload": {"phase": "move_forward", "decision": "complete_task"}})
    panel(
        {
            "kind": "agent_message",
            "message": {
                "type": "final",
                "content": "任务已完成。",
                "confidence": "high",
                "requires_user_reply": False,
            },
        }
    )
    panel.finish_turn(summary)

    out = capsys.readouterr().out
    assert "任务步骤：" in out
    assert "1. 向前飞行" in out
    assert "状态：我准备向前飞行。" in out
    assert "状态：动作完成 1/2：向前飞行；已完成移动，已稳定。" in out
    assert "状态：观察完成：已更新图像、深度、位姿。" in out
    assert "FlyAgent: 任务已完成。" in out
    assert "运行记录：runs/interactive_cli_test" in out


def test_clear_command_keeps_cli_open(tmp_path, capsys):
    session = FakeCliSession(tmp_path)

    should_continue = handle_slash_command("/clear", session)  # type: ignore[arg-type]

    assert should_continue is True
    assert capsys.readouterr().out == "\n"


def test_cli_defaults_to_chat_model_artifacts():
    args = build_parser().parse_args([])

    assert args.command == "chat"
    assert args.env_path == ".env"
    assert args.parser_mode is None
    assert args.save_artifacts is True


def test_cli_start_alias_enters_chat():
    args = build_parser().parse_args(["start"])

    assert args.command == "start"


def test_default_chat_map_prefers_cityenviron():
    class Package:
        def __init__(self, map_id: str) -> None:
            self.map_id = map_id

    assert default_chat_map_id([Package("m1"), Package("cityenviron_win64")]) == "cityenviron_win64"
    assert default_chat_map_id([Package("m1")]) == "m1"
