"""AerialVLN benchmark protocol identity and task-context semantics.

This context clarifies the official episode-termination meaning of landing
language for the shared tool-loop agent. It does not change the raw dataset
instruction, Runtime tools, simulator control, or metric computation.
"""

from __future__ import annotations


AERIAL_VLN_PROTOCOL = "aerial_vln.uav_agent_native_actions.v2"

AERIAL_VLN_PARSER_PROTOCOL_CONTEXT = """AerialVLN episode protocol:
Task goal: Follow the natural-language route in its specified order and reach the requested final region.
Instruction interpretation: Preserve landmark-triggered actions and separately requested ascent, descent, approach and intermediate checkpoints. Seeing a landmark is different from reaching or passing it.
Information characteristics: Upcoming landmarks may not yet be visible. The instruction can contain explicit motions, landmark-dependent conditions and historical passage requirements; these retain their distinct meanings even when an earlier landmark is no longer in view.
Completion conditions:
- Interpret "land on X", "land in X", and "land by X" as reaching the target region associated with X and then ending the episode.
- Confirm the requested target region and position relation. This episode-level landing language does not require physical ground contact, an AirSim landed state, calling uav.land, or a landing-readiness claim.
- Do not infer descent, low altitude, surface proximity, or surface clearance from the episode-level landing phrase. Preserve those requirements when independently requested.
""".strip()
