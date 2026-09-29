"""TravelUAV transfer-evaluation protocol identity and task context.

The protocol deliberately describes FlyAgent's front-camera, native-action
setting.  It must not be confused with the official five-view assistant-guided
TravelUAV leaderboard protocol.
"""

from __future__ import annotations


TRAVEL_UAV_PROTOCOL = (
    "travel_uav.flyagent_native_actions.front_view.no_assistant.v1"
)

TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT = """TravelUAV transfer-evaluation protocol:
Task goal: Find the described target using its initial direction, appearance and surrounding spatial relations.
Instruction interpretation: A relative yaw supplied in the initial instruction is anchored to the UAV orientation at instruction start; do not reinterpret it relative to each later heading. Surrounding-scene descriptions help identify the target and are not automatically an ordered route to traverse.
Information characteristics: An initial direction supplies neither target distance nor guaranteed immediate visibility; a search direction and a visible target are different information. The transfer setting uses available front-view observations, not an assumed multi-view assistant.
Completion conditions:
- Navigate until the described target object and its surrounding spatial relation are visually confirmed.
- A distant or ambiguous candidate sighting is not sufficient for completion; finish only after reaching a nearby, judgeable target region and then end the episode.
- Physical ground contact and calling uav.land are not required unless the instruction independently requests landing.
""".strip()
