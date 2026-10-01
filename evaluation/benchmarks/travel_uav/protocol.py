"""TravelUAV transfer-evaluation protocol identity and task context.

The protocol deliberately describes FlyAgent's front-camera, native-action
setting.  It must not be confused with the official five-view assistant-guided
TravelUAV leaderboard protocol.
"""

from __future__ import annotations


TRAVEL_UAV_PROTOCOL = (
    "travel_uav.flyagent_native_actions.front_view.no_assistant.v2"
)

TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT = """TravelUAV transfer-evaluation protocol:
Task goal: Find the described target using its initial direction, appearance and surrounding spatial relations.
Instruction interpretation: A relative yaw supplied in the initial instruction is a strong, persistent search-bearing prior anchored to the UAV orientation at instruction start; do not reinterpret it relative to each later heading. After aligning to it, preserve that world-relative search corridor while advancing. Do not accumulate reverse turns that cancel the initial yaw merely to inspect a same-category object near the original heading. Deviate only for safety, obstacle avoidance, or a candidate whose appearance, surrounding relations and bearing together support target identity. Surrounding-scene descriptions help identify the target and are not automatically an ordered route to traverse.
Information characteristics: An initial direction supplies neither target distance nor guaranteed immediate visibility; a search direction and a visible target are different information. A same-category detector match and a short visible-surface depth do not confirm the target when its bearing conflicts with the initial direction or its surrounding relations are not distinctive enough. After the target is confirmed from all available cues, stop unsupported large back-and-forth rotations and use the confirmed bearing, scene relations and valid visible-surface depth to approach. The transfer setting uses available front-view observations, not an assumed multi-view assistant.
Completion conditions:
- Navigate until the described target object and its surrounding spatial relation are visually confirmed.
- A distant or ambiguous candidate sighting is not sufficient for completion. Finish only when available visual and depth evidence supports that the UAV is within approximately 20 m of the confirmed target region.
- Physical ground contact and calling uav.land are not required unless the instruction independently requests landing.
""".strip()
