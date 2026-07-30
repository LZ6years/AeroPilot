"""Test the interactive CLI command layer without creating a real runtime."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.cli import display as cli_display
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


def test_cli_narrative_panel_prints_persistent_phase_completion(capsys):
    panel = CliNarrativePanel()
    panel({"kind": "instruction_started", "payload": {"phase_names": ["move_forward", "rotate_right"]}})

    panel(
        {
            "kind": "phase_transition_applied",
            "payload": {
                "phase": "move_forward",
                "decision": "advance_to_next_phase",
                "phase_completion_evaluation": {
                    "evaluations": [
                        {
                            "criteria": {"unit": "m"},
                            "target": 50.0,
                            "actual": 49.841,
                            "error": -0.159,
                        }
                    ]
                },
            },
        }
    )
    panel(
        {
            "kind": "phase_transition_applied",
            "payload": {
                "phase": "rotate_right",
                "decision": "complete_task",
                "phase_completion_evaluation": {
                    "evaluations": [
                        {
                            "criteria": {"unit": "deg"},
                            "target": 90.0,
                            "actual": 87.294,
                            "error": -2.706,
                        }
                    ]
                },
            },
        }
    )

    out = capsys.readouterr().out
    assert "完成 1/2：向前飞行" in out
    assert "完成 2/2：向右旋转" in out
    assert "49.84/50 m" in out
    assert "误差 -0.16m" in out
    assert "87.29/90 deg" in out
    assert "误差 -2.71deg" in out


def test_cli_narrative_panel_truncates_tty_status_to_one_line(monkeypatch, capsys):
    panel = CliNarrativePanel()
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli_display.shutil, "get_terminal_size", lambda fallback: os.terminal_size((24, 20)))

    panel._set_status("继续当前步骤：" + "向前飞行" * 20)

    out = capsys.readouterr().out
    assert "\n" not in out
    assert "..." in out


def test_cli_narrative_panel_preserves_non_tty_status_output(capsys):
    panel = CliNarrativePanel()

    panel._set_status("继续当前步骤：" + "向前飞行" * 3)

    out = capsys.readouterr().out
    assert "状态：" in out
    assert "向前飞行" * 3 in out


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
    assert args.execution == "managed"
    assert args.defer_scene_start is False
    assert args.observation_profile is None


def test_cli_accepts_observation_profile_path():
    args = build_parser().parse_args(
        ["--observation-profile", "configs/observations/custom.json"]
    )

    assert args.observation_profile == "configs/observations/custom.json"


def test_cli_rejects_removed_simulator_dry_run_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--execution", "dry-run"])


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--" + "launch",
        "--connect-" + "existing",
        "--no-" + "launch",
    ],
)
def test_cli_rejects_removed_simulator_boolean_flags(removed_flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args([removed_flag])


def test_cli_start_alias_enters_chat():
    args = build_parser().parse_args(["start"])

    assert args.command == "start"


def test_default_chat_map_prefers_cityenviron():
    class Package:
        def __init__(self, map_id: str) -> None:
            self.map_id = map_id

    assert default_chat_map_id([Package("m1"), Package("cityenviron_win64")]) == "cityenviron_win64"
    assert default_chat_map_id([Package("m1")]) == "m1"
