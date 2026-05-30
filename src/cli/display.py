"""Terminal display helpers for the FlyAgent CLI.

This module keeps terminal rendering separate from command parsing.  The UI
strings that must be Chinese are written with Unicode escapes so future edits
do not depend on the active Windows or PowerShell code page.
"""

from __future__ import annotations

import json
import shutil
import sys
import unicodedata
from typing import Any

from src.runtime.runners.interactive_runtime import compact_turn_summary


ANSI_BLUE = "\033[34m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def color_text(text: str, color: str) -> str:
    # Keep redirected output and tests plain while making the interactive shell easier to scan.
    if not sys.stdout.isatty():
        return text
    return f"{color}{text}{ANSI_RESET}"


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
        return
    print("")


def print_summary(summary: dict[str, Any], *, include_messages: bool = False) -> None:
    compact = compact_turn_summary(summary)
    messages = compact.get("agent_messages", []) if include_messages else []
    if include_messages and messages:
        print(color_text("FlyAgent messages:", ANSI_DIM))
        for message in messages:
            content = str(message.get("content", "")).strip()
            if content:
                print(f"- [{message.get('type', 'progress')}] {content}")
    if include_messages:
        print_json(compact)
        return
    print_json(summary_for_terminal(compact))


def summary_for_terminal(compact: dict[str, Any]) -> dict[str, Any]:
    """Return the small end-of-turn summary shown when no live panel is active."""

    return {
        "instruction": compact.get("instruction"),
        "completed": compact.get("completed"),
        "airborne": compact.get("airborne"),
        "pose_delta": compact.get("pose_delta", {}),
        "artifact_counts": compact.get("artifact_counts", {}),
        "run_dir": compact.get("run_dir"),
        **(
            {"error_type": compact.get("error_type"), "error": compact.get("error")}
            if compact.get("error")
            else {}
        ),
    }


class CliNarrativePanel:
    """Render one instruction as steps, one live status line, and one final reply."""

    def __init__(self) -> None:
        self.phase_names: list[str] = []
        self.current_phase: str | None = None
        self.final_message: str | None = None
        self._status_visible = False
        self._last_status = ""

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        if kind == "instruction_started":
            self._start_instruction(payload)
            return
        if kind == "agent_message":
            self._handle_agent_message(event)
            return
        if kind == "phase_plan_ready":
            self.current_phase = str(payload.get("phase") or self.current_phase or "")
            self._set_status(self._phase_status(_ZH["executing"], self.current_phase, self._action_window_text(payload)))
            return
        if kind == "tool_step_completed":
            phase = str(payload.get("phase") or self.current_phase or "")
            self._set_status(self._phase_status(_ZH["action_done"], phase, self._tool_status(payload)))
            return
        if kind == "observation_updated":
            self._set_status(f'{_ZH["observation_done"]}{self._modalities_text(payload)}{_ZH["period"]}')
            return
        if kind == "phase_transition_continued":
            phase = str(payload.get("phase") or self.current_phase or "")
            self._set_status(self._phase_status(_ZH["continue_step"], phase, self._reason_text(payload)))
            return
        if kind == "phase_transition_window_limit_reached":
            phase = str(payload.get("phase") or self.current_phase or "")
            self._set_status(self._phase_status(_ZH["window_limit"], phase, _ZH["safe_next"]))
            return
        if kind == "runtime_loop_guard_reached":
            phase = str(payload.get("phase") or self.current_phase or "")
            self._print_persistent(self._phase_status(_ZH["window_limit"], phase, _ZH["safe_next"]), color=ANSI_YELLOW)
            return
        if kind == "phase_transition_applied":
            decision = str(payload.get("applied_decision") or payload.get("decision") or "")
            if decision == "advance_to_next_phase":
                self._print_persistent(self._phase_completion_line(payload, fallback=_ZH["advance"]))
            elif decision == "complete_task":
                self._print_persistent(self._phase_completion_line(payload, fallback=_ZH["complete整理"]))
                self._set_status(_ZH["complete整理"])
            else:
                self._set_status(_ZH["phase_updated"])

    def finish_turn(self, summary: dict[str, Any]) -> None:
        """Clear temporary status and print only the final answer plus run path."""

        self._clear_status()
        final = self.final_message or self._final_message_from_summary(summary)
        if final:
            print(color_text(f"FlyAgent: {final}", ANSI_GREEN))
        else:
            completed = _ZH["completed"] if summary.get("completed") else _ZH["not_completed"]
            color = ANSI_GREEN if summary.get("completed") else ANSI_YELLOW
            print(color_text(f'FlyAgent: {_ZH["turn_task"]}{completed}{_ZH["period"]}', color))
        run_dir = summary.get("run_dir")
        if run_dir:
            print(color_text(f'{_ZH["run_record"]}{run_dir}', ANSI_DIM))
        self.final_message = None
        self.current_phase = None
        self.phase_names = []

    def _start_instruction(self, payload: dict[str, Any]) -> None:
        self._clear_status()
        self.final_message = None
        self.phase_names = [str(name) for name in payload.get("phase_names", []) if name]
        self.current_phase = self.phase_names[0] if self.phase_names else None
        if not self.phase_names:
            return
        print(_ZH["task_steps"])
        total = len(self.phase_names)
        for index, name in enumerate(self.phase_names, start=1):
            print(f"  {index}. {self._humanize_phase(name)}")
        self._set_status(f'{_ZH["prepare"]} 1/{total}{_ZH["colon"]}{self._humanize_phase(self.phase_names[0])}')

    def _handle_agent_message(self, event: dict[str, Any]) -> None:
        message = event.get("message", {}) if isinstance(event.get("message", {}), dict) else {}
        content = str(message.get("content", "")).strip()
        if not content:
            return
        message_type = str(message.get("type", "progress"))
        if message_type == "final":
            self.final_message = content
            self._set_status(_ZH["complete整理"])
            return
        if message_type == "warning":
            self._set_status(f'{_ZH["warning"]}{content}')
            return
        if message_type == "question":
            self._set_status(f'{_ZH["question"]}{content}')
            return
        self._set_status(content)

    def _set_status(self, text: str) -> None:
        clean = " ".join(str(text).split())
        if not clean or clean == self._last_status:
            return
        self._last_status = clean
        if sys.stdout.isatty():
            display_text = _truncate_display_width(clean, max(20, shutil.get_terminal_size((100, 24)).columns - 1))
            print(f"\r\033[2K{color_text(display_text, ANSI_DIM)}", end="", flush=True)
            self._status_visible = True
            return
        print(color_text(f'{_ZH["status"]}{clean}', ANSI_DIM))

    def _clear_status(self) -> None:
        if sys.stdout.isatty() and self._status_visible:
            print("\r\033[2K", end="", flush=True)
        self._status_visible = False
        self._last_status = ""

    def _phase_status(self, prefix: str, phase: str, detail: str) -> str:
        index_text = self._phase_index_text(phase)
        name = self._humanize_phase(phase)
        detail_text = f'{_ZH["semicolon"]}{detail}' if detail else ""
        return f'{prefix}{index_text}{_ZH["colon"]}{name}{detail_text}'

    def _print_persistent(self, text: str, *, color: str = ANSI_DIM) -> None:
        clean = " ".join(str(text).split())
        if not clean:
            return
        self._clear_status()
        print(color_text(clean, color))

    def _phase_completion_line(self, payload: dict[str, Any], *, fallback: str) -> str:
        phase = str(payload.get("phase") or self.current_phase or "")
        index_text = self._phase_index_text(phase).strip()
        phase_name = self._humanize_phase(phase)
        prefix = f'{_ZH["phase_done"]} {index_text}{_ZH["colon"]}{phase_name}' if index_text else f'{_ZH["phase_done"]}{_ZH["colon"]}{phase_name}'
        metric = self._phase_completion_metric(payload)
        return f"{prefix}{metric}{_ZH['period']}" if metric else fallback

    @staticmethod
    def _phase_completion_metric(payload: dict[str, Any]) -> str:
        evaluation = payload.get("phase_completion_evaluation", {})
        evaluations = evaluation.get("evaluations", []) if isinstance(evaluation, dict) else []
        if not evaluations or not isinstance(evaluations[0], dict):
            return ""
        item = evaluations[0]
        criteria = item.get("criteria", {}) if isinstance(item.get("criteria", {}), dict) else {}
        actual = _format_number(item.get("actual"))
        target = _format_number(item.get("target"))
        error = _format_number(item.get("error"), signed=True)
        unit = str(criteria.get("unit") or "").strip()
        if actual is None or target is None:
            return ""
        unit_text = f" {unit}" if unit else ""
        error_text = f"{_ZH['comma']}{_ZH['error']} {error}{unit}" if error is not None else ""
        return f"{_ZH['left_paren']}{actual}/{target}{unit_text}{error_text}{_ZH['right_paren']}"

    def _phase_index_text(self, phase: str) -> str:
        if not self.phase_names or not phase:
            return ""
        try:
            index = self.phase_names.index(phase) + 1
        except ValueError:
            return ""
        return f" {index}/{len(self.phase_names)}"

    def _action_window_text(self, payload: dict[str, Any]) -> str:
        action_window = payload.get("action_window", {}) if isinstance(payload.get("action_window", {}), dict) else {}
        return str(action_window.get("intent") or action_window.get("expected_effect") or "").strip()

    def _tool_status(self, payload: dict[str, Any]) -> str:
        tool = str(payload.get("tool") or "")
        status = _ZH["stable"] if (payload.get("stabilization") or {}).get("status") == "completed" else _ZH["executed"]
        if tool == "uav.hover":
            return f'{_ZH["hover_done"]}{_ZH["comma"]}{status}{_ZH["period"]}'
        if tool == "uav.turn_relative":
            return f'{_ZH["turn_done"]}{_ZH["comma"]}{status}{_ZH["period"]}'
        if tool == "uav.move_body":
            return f'{_ZH["move_done"]}{_ZH["comma"]}{status}{_ZH["period"]}'
        return f'{_ZH["generic_done"]}{_ZH["comma"]}{status}{_ZH["period"]}'

    @staticmethod
    def _modalities_text(payload: dict[str, Any]) -> str:
        names = []
        frame_refs = payload.get("frame_refs", {}) if isinstance(payload.get("frame_refs", {}), dict) else {}
        if "rgb" in frame_refs:
            names.append(_ZH["image"])
        if "depth" in frame_refs:
            names.append(_ZH["depth"])
        if payload.get("has_pose"):
            names.append(_ZH["pose"])
        return _ZH["list_sep"].join(names) or _ZH["observation_result"]

    @staticmethod
    def _reason_text(payload: dict[str, Any]) -> str:
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            return _ZH["continue_fill"]
        return reason[:80] + ("..." if len(reason) > 80 else "")

    @staticmethod
    def _final_message_from_summary(summary: dict[str, Any]) -> str:
        for message in reversed(summary.get("agent_messages", []) or []):
            if message.get("type") == "final" and message.get("content"):
                return str(message["content"])
        return ""

    @staticmethod
    def _humanize_phase(name: str) -> str:
        text = str(name or "").strip().replace("_", " ")
        lowered = text.lower()
        for src, dst in _PHASE_REPLACEMENTS:
            lowered = lowered.replace(src, dst)
        return lowered


def _format_number(value: Any, *, signed: bool = False) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:+.2f}" if signed else f"{number:.2f}"
    return text.rstrip("0").rstrip(".")


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _truncate_display_width(text: str, max_width: int) -> str:
    if max_width <= 0 or _display_width(text) <= max_width:
        return text
    suffix = "..."
    suffix_width = _display_width(suffix)
    result = []
    width = 0
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        if width + char_width + suffix_width > max_width:
            break
        result.append(char)
        width += char_width
    return "".join(result).rstrip() + suffix


_ZH = {
    "action_done": "\u52a8\u4f5c\u5b8c\u6210",
    "advance": "\u9636\u6bb5\u5b8c\u6210\uff1a\u51c6\u5907\u8fdb\u5165\u4e0b\u4e00\u6b65\u3002",
    "colon": "\uff1a",
    "comma": "\uff0c",
    "complete整理": "\u4efb\u52a1\u5b8c\u6210\uff1a\u6b63\u5728\u6574\u7406\u6700\u7ec8\u56de\u7b54\u3002",
    "completed": "\u5df2\u5b8c\u6210",
    "continue_fill": "\u7ee7\u7eed\u8865\u8db3\u5f53\u524d\u76ee\u6807\u3002",
    "continue_step": "\u7ee7\u7eed\u5f53\u524d\u6b65\u9aa4",
    "depth": "\u6df1\u5ea6",
    "error": "\u8bef\u5dee",
    "executed": "\u5df2\u6267\u884c",
    "executing": "\u6b63\u5728\u6267\u884c",
    "generic_done": "\u52a8\u4f5c\u5df2\u5b8c\u6210",
    "hover_done": "\u5df2\u60ac\u505c",
    "image": "\u56fe\u50cf",
    "left_paren": "\uff08",
    "list_sep": "\u3001",
    "move_done": "\u5df2\u5b8c\u6210\u79fb\u52a8",
    "not_completed": "\u672a\u5b8c\u6210",
    "observation_done": "\u89c2\u5bdf\u5b8c\u6210\uff1a\u5df2\u66f4\u65b0",
    "observation_result": "\u89c2\u5bdf\u7ed3\u679c",
    "period": "\u3002",
    "phase_done": "\u5b8c\u6210",
    "phase_updated": "\u9636\u6bb5\u72b6\u6001\u5df2\u66f4\u65b0\u3002",
    "pose": "\u4f4d\u59ff",
    "prepare": "\u51c6\u5907\u6267\u884c",
    "question": "\u9700\u8981\u786e\u8ba4\uff1a",
    "right_paren": "\uff09",
    "run_record": "\u8fd0\u884c\u8bb0\u5f55\uff1a",
    "safe_next": "\u5c06\u5b89\u5168\u60ac\u505c\u5e76\u505c\u6b62\u672c\u8f6e\u3002",
    "semicolon": "\uff1b",
    "stable": "\u5df2\u7a33\u5b9a",
    "status": "\u72b6\u6001\uff1a",
    "task_steps": "\u4efb\u52a1\u6b65\u9aa4\uff1a",
    "turn_done": "\u5df2\u5b8c\u6210\u65cb\u8f6c",
    "turn_task": "\u672c\u8f6e\u4efb\u52a1",
    "warning": "\u63d0\u9192\uff1a",
    "window_limit": "\u8fbe\u5230\u672c\u6b65\u5c1d\u8bd5\u4e0a\u9650",
}

_PHASE_REPLACEMENTS = [
    ("move forward", "\u5411\u524d\u98de\u884c"),
    ("rotate right", "\u5411\u53f3\u65cb\u8f6c"),
    ("rotate left", "\u5411\u5de6\u65cb\u8f6c"),
    ("strafe left", "\u5411\u5de6\u5e73\u79fb"),
    ("strafe right", "\u5411\u53f3\u5e73\u79fb"),
    ("adjust altitude up", "\u4e0a\u5347"),
    ("adjust altitude down", "\u4e0b\u964d"),
    ("stabilize hover", "\u60ac\u505c\u7a33\u5b9a"),
    ("hover", "\u60ac\u505c"),
]
