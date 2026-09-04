"""Deterministic parser/planners used only by real-engine probes."""

from __future__ import annotations

from src.agent.phase import (
    COMPLETE_TASK,
    PhaseTransition,
)
from src.agent.planning.schemas import (
    CoupledActionPhasePlan,
)
from src.agent.semantic.schemas import TaskParseResult
from src.agent.semantic.validator import semantic_plan_from_dict


class StaticProbeTaskParser:
    """Create one semantic phase without calling a model."""

    def parse(self, instruction: str) -> TaskParseResult:
        plan = semantic_plan_from_dict(
            {
                "intent": "benchmark_probe",
                "task_goal": {
                    "target": "probe waypoint",
                    "target_attributes": {},
                    "final_state": "probe complete",
                },
                "phases": [
                    {
                        "name": "probe",
                        "goal": "exercise simulator primitives",
                        "done_when": ["probe actions completed"],
                        "continue_when": ["probe actions not completed"],
                    }
                ],
            }
        )
        return TaskParseResult(
            instruction=instruction,
            parser_mode="manual_probe",
            plan=plan,
        )


class CompleteProbePlanner:
    """Complete after Runtime initialization and the first observation."""

    planner_mode = "manual_probe"

    def plan_next_with_phase_decision(
        self, state, context
    ) -> CoupledActionPhasePlan:
        return CoupledActionPhasePlan(
            phase=state.current_phase_name,
            phase_decision=PhaseTransition(
                COMPLETE_TASK, "initialization probe complete"
            ),
            steps=[],
            agent_messages=[],
        )
