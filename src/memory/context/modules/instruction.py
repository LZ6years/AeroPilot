"""Instruction context module.

The flight memory keeps full instruction records for audit and replay, but the
planner only needs a compact view of the active instruction plus recent history.
This module therefore summarizes instruction records deterministically instead
of forwarding complete semantic plans every planning window.
"""

from __future__ import annotations

from typing import Any

from src.memory.context.modules.base import ContextModule
from src.memory.context.schemas import BudgetSpec, ContextFragment, MemorySourceBundle, StageContextInput


class DefaultInstructionContextModule(ContextModule):
    name = "instruction"
    priority = 100
    max_items = 5

    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        memory = sources.flight_memory
        reader = memory.memory_reader
        plan = stage_context.semantic_plan
        current_phase = stage_context.current_phase
        raw_history = reader.read_refs(memory.instruction_history[-self.max_items :]) if reader else []
        history_summary = _summarize_instruction_history(
            raw_history,
            active_instruction=memory.active_instruction,
            max_items=self.max_items,
        )
        current_instruction = _current_instruction_summary(memory.active_instruction, plan)
        current_phase_summary = _phase_summary(current_phase)
        goals = [
            {
                "type": "mission_goal",
                "value": current_instruction.get("goal", {}),
                "source": self.__class__.__name__,
            }
        ]
        if current_phase_summary:
            goals.append({"type": "phase_goal", "value": current_phase_summary, "source": self.__class__.__name__})
        return ContextFragment(
            name=self.name,
            enabled=True,
            source=self.__class__.__name__,
            content={
                "current_instruction": current_instruction,
                "instruction_history": history_summary,
                "current_phase": current_phase_summary,
                "uncertainties": _compact_uncertainties(plan.get("uncertainties", [])),
            },
            goals=goals,
            confidence="high",
            budget=BudgetSpec(priority=self.priority, tokens_estimate=150, max_items=self.max_items),
            metadata={"reset_epoch": memory.reset_epoch},
        )

    def describe_output(self) -> dict[str, Any]:
        return {
            "summary": "Compact current instruction, recent instruction summaries, and current phase intent.",
            "content_fields": {
                "current_instruction": "Compact structured summary of the active instruction and parsed mission goal.",
                "instruction_history": "Recent instruction summaries; full semantic plans remain in memory/debug artifacts.",
                "current_phase": "Compact current phase name and meaning without duplicating full completion criteria.",
                "uncertainties": "Known uncertainties from task parsing that should make planning more conservative.",
            },
            "goal_fields": {
                "mission_goal": "Overall parsed mission goal.",
                "phase_goal": "The current phase objective that should guide the next 1-3 actions.",
            },
        }


def _current_instruction_summary(instruction: str, plan: dict[str, Any]) -> dict[str, Any]:
    goal = plan.get("goal", {}) if isinstance(plan.get("goal"), dict) else {}
    return {
        "text": instruction,
        "intent": plan.get("intent", ""),
        "goal": _compact_goal(goal),
    }


def _summarize_instruction_history(
    records: list[Any],
    *,
    active_instruction: str,
    max_items: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in records[-max_items:]:
        if not isinstance(record, dict):
            continue
        instruction = str(record.get("instruction") or "")
        plan = record.get("semantic_plan") if isinstance(record.get("semantic_plan"), dict) else {}
        summary = {
            "id": record.get("id"),
            "instruction": instruction,
            "intent": plan.get("intent", ""),
            "target": _target_label(plan.get("goal", {})),
            "status": "active" if instruction and instruction == active_instruction else "previous",
        }
        if not summary["target"]:
            summary.pop("target")
        summaries.append({key: value for key, value in summary.items() if value not in (None, "", [], {})})
    return summaries


def _compact_goal(goal: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(goal, dict):
        return {}
    compact: dict[str, Any] = {}
    target = goal.get("target")
    if isinstance(target, dict):
        target_summary: dict[str, Any] = {}
        if target.get("description"):
            target_summary["description"] = target.get("description")
        attributes = target.get("attributes")
        if isinstance(attributes, dict) and attributes.get("category"):
            target_summary["category"] = attributes.get("category")
        if target_summary:
            compact["target"] = target_summary
    if goal.get("final_state"):
        compact["final_state"] = goal.get("final_state")
    return compact


def _target_label(goal: Any) -> str:
    if not isinstance(goal, dict):
        return ""
    target = goal.get("target")
    if isinstance(target, dict):
        attributes = target.get("attributes")
        if isinstance(attributes, dict) and attributes.get("category"):
            return str(attributes.get("category"))
        if target.get("description"):
            return str(target.get("description"))
    return ""


def _phase_summary(phase: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(phase, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("index", "total", "name", "meaning"):
        if key in phase:
            summary[key] = phase[key]
    return summary


def _compact_uncertainties(items: Any, max_items: int = 5) -> list[Any]:
    if not isinstance(items, list):
        return []
    return list(items[:max_items])
