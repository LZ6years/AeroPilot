"""AerialVLN benchmark protocol identity and parser-only semantics.

This context clarifies the official episode-termination meaning of landing
language for the shared TaskParser. It does not change the raw dataset
instruction, Runtime tools, simulator control, or metric computation.
"""

from __future__ import annotations


AERIAL_VLN_PROTOCOL = "aerial_vln.uav_agent_native_actions.v2"

AERIAL_VLN_PARSER_PROTOCOL_CONTEXT = """AerialVLN episode protocol:
- Interpret \"land on X\", \"land in X\", and \"land by X\" as reaching the target region associated with X and then ending the episode.
- The final completion conditions may require confirmation of the target region and the position relation requested by the instruction.
- This episode-level landing language does not require physical ground contact, an AirSim landed state, or calling uav.land.
- In completion conditions, reaching the target region and confirming the requested position relation is the complete result. Do not require or claim a landed, landing, touchdown, ground-contact, landing-readiness, or suitable-for-landing state.
- Do not infer descent, low altitude, surface proximity, or surface clearance from the episode-level landing phrase. Preserve one of those requirements only when the instruction requests it separately.
""".strip()
