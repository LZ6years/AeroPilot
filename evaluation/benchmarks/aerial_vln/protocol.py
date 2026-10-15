"""AerialVLN benchmark protocol identity and task-context semantics.

This caller-owned context preserves the whole instructed route and its final
position relation. It does not change the raw dataset
instruction, Runtime tools, simulator control, or metric computation.
"""

from __future__ import annotations


AERIAL_VLN_PROTOCOL = "aerial_vln.uav_agent_native_actions.v3"

AERIAL_VLN_PARSER_PROTOCOL_CONTEXT = """AerialVLN episode protocol:
Task goal: Follow the natural-language route in its specified order and reach the requested final region.
Instruction interpretation: Preserve landmark-triggered actions and separately requested ascent, descent, approach and intermediate checkpoints. Seeing a landmark is different from reaching or passing it.
Information characteristics: Upcoming landmarks may not yet be visible. The instruction can contain explicit motions, landmark-dependent conditions and historical passage requirements; these retain their distinct meanings even when an earlier landmark is no longer in view.
Completion conditions:
- Complete the episode only after fulfilling the whole instructed route and its final position relation. An intermediate arrival or stop does not complete the episode when explicit subsequent requirements remain.
Arrival precision:
- The scoring tolerance is a final horizontal distance of at most 20 metres. Treat this as an outer tolerance, rather than a preferred stopping distance. Aim to reach the instructed stopping region within roughly 10 metres when target identity, position relations and available distance evidence support that judgment.
- If distance measurements are unavailable, invalid or unsuitable for horizontal separation, do not invent a claim of being within 10 metres. Continue judging from target identity, spatial relations, actual motion and RGB observations.
- The roughly 10-metre aim does not override the instructed route, separately requested ascent/descent or final position relations.
""".strip()
