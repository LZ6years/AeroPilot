"""Print a compact native tool-call transcript for manual inspection.

This probe does not contact a model or simulator. It exercises the exact
provider-message projection used by ModelActionPlanner.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.stores import MemoryEventWriter
from src.plugins import PluginRegistry
from src.tools.registry import default_registry


def main() -> None:
    plugins = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.visual_grounding"],
            }
        }
    )
    registry = default_registry(plugins)
    with tempfile.TemporaryDirectory(prefix="flyagent_transcript_") as temp_dir:
        transcript = AgentTranscript(MemoryEventWriter(temp_dir))
        transcript.start_instruction(
            "确认画面中的人，然后小幅转向并前进",
            "inst_demo",
            InstructionEnvelope(
                instruction_id="inst_demo",
                protocol_context="",
                identity={},
                user_preferences={},
                knowledge_catalog=(),
                static_module_guidance=(),
            ),
        )
        fragment = ContextFragment("probe", {"debug_only": True})
        transcript.append_state(
            StateCheckpoint(
                checkpoint_id="checkpoint_0001",
                reason="instruction_start",
                observation_id="obs_0001",
                task_plan=None,
                loaded_knowledge=(),
                fragments=(fragment,),
                fingerprint=checkpoint_fingerprint(
                    "obs_0001", None, (), (fragment,)
                ),
            )
        )
        transcript.append_assistant(
            AssistantTurn(
                "assistant_detect",
                "先确认当前观察中的目标。",
                [
                    AgentToolCall(
                        "call_detect",
                        "skill.visual_grounding.locate",
                        {
                            "observation_id": "obs_0001",
                            "query": "画面中的人",
                            "include_distance": False,
                        },
                    )
                ],
            )
        )
        transcript.append_tool_result(
            ToolResult(
                "call_detect",
                "skill.visual_grounding.locate",
                "ok",
                "one grounded candidate",
                data={
                    "observation_id": "obs_0001",
                    "query": "画面中的人",
                    "include_distance": False,
                    "candidates": [
                        {
                            "candidate_id": "candidate_01",
                            "label": "person",
                            "confidence": 0.91,
                            "box_normalized": [0.4, 0.2, 0.6, 0.8],
                            "distance_m": None,
                            "distance_status": "not_requested",
                        }
                    ],
                },
                requested_args={
                    "observation_id": "obs_0001",
                    "query": "画面中的人",
                    "include_distance": False,
                },
            )
        )

        messages = transcript.provider_messages(registry)
        for index, message in enumerate(messages):
            row = {"index": index, "role": message["role"]}
            if "tool_calls" in message:
                row["tool_call_ids"] = [call["id"] for call in message["tool_calls"]]
            if "tool_call_id" in message:
                row["tool_call_id"] = message["tool_call_id"]
            print(json.dumps(row, ensure_ascii=False))

        state_text = messages[1]["content"][-1]["text"]
        if "allowed_action_constraints" in state_text:
            raise RuntimeError("provider state unexpectedly duplicated tool constraints")
        print("probe_ok")


if __name__ == "__main__":
    main()
