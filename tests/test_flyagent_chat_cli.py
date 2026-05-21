"""Test the interactive CLI command layer without creating a real runtime."""

from __future__ import annotations

import json
from pathlib import Path

from src.cli.flyagent_chat import handle_slash_command
from src.runtime.cli.main import build_parser, default_chat_map_id


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
        self.last_summary = {"instruction": "向前飞行5米", "completed": True}
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


def test_default_chat_map_prefers_cityenviron():
    class Package:
        def __init__(self, map_id: str) -> None:
            self.map_id = map_id

    assert default_chat_map_id([Package("m1"), Package("cityenviron_win64")]) == "cityenviron_win64"
    assert default_chat_map_id([Package("m1")]) == "m1"
