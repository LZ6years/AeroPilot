"""Deterministic parser/planners used only by real-engine probes."""

from src.agent.transcript import AgentToolCall, AssistantTurn


class CompleteProbePlanner:
    """Scripted terminal for initialization probes; never judges navigation."""
    planner_mode = "manual_probe"

    def plan_turn(self, transcript, registry, **kwargs):
        return AssistantTurn("probe_terminal", "Initialization observation obtained", [
            AgentToolCall("probe_done", "runtime.report_completed", {
                "reason": "Initialization probe complete, not a navigation assessment",
                "evidence_refs": [transcript.latest_observation_id()],
            })
        ])
