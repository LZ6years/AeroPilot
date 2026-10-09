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
Instruction interpretation: A relative yaw supplied in the initial instruction is an accurate target bearing measured from the UAV heading at instruction start. It defines a world-referenced ray from the instruction-start pose and must not be reinterpreted relative to each later heading. It does not provide target distance or prescribe that the UAV continuously move along that direction. Surrounding-scene descriptions help identify the target and are not automatically an ordered route to traverse.
Information characteristics: An initial direction supplies neither target distance nor guaranteed immediate visibility; a direction and a visible target are different information. Target appearance and surrounding spatial relations are identity evidence. A same-category detector match or a short visible-surface depth alone does not confirm the described target. The transfer setting uses available front-view observations, not an assumed multi-view assistant.
Completion conditions:
- Navigate until the described target object and its surrounding spatial relation are visually confirmed.
- A distant or ambiguous candidate sighting is not sufficient for completion. Finish only when available visual and depth evidence supports that the UAV is within approximately 20 m of the confirmed target region.
- Physical ground contact and calling uav.land are not required unless the instruction independently requests landing.
""".strip()
