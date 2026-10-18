"""TravelUAV transfer-evaluation protocol identity and task context.

The protocol deliberately describes FlyAgent's front-camera, native-action
setting.  It must not be confused with the official five-view assistant-guided
TravelUAV leaderboard protocol.
"""

from __future__ import annotations


TRAVEL_UAV_PROTOCOL = (
    "travel_uav.flyagent_native_actions.front_view.no_assistant.v3"
)

TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT = """TravelUAV transfer-evaluation protocol:
Task goal: Find the described target using its initial direction, appearance and surrounding spatial relations.
Instruction interpretation: A relative yaw supplied in the initial instruction is an accurate target bearing measured from the UAV heading at instruction start. It defines a world-referenced ray from the instruction-start pose and must not be reinterpreted relative to each later heading. It does not provide target distance or prescribe that the UAV continuously move along that direction. Surrounding-scene descriptions help identify the target and are not automatically an ordered route to traverse.
Information characteristics: An initial direction supplies neither target distance nor guaranteed immediate visibility; a direction and a visible target are different information. Target appearance and surrounding spatial relations are identity evidence. A same-category detector match or a short visible-surface depth alone does not confirm the described target. The transfer setting uses available front-view observations, not an assumed multi-view assistant.
Completion conditions:
- Ensure the described target is visually confirmed using its appearance, initial bearing and surrounding spatial relations. Surroundings are identification evidence that may be established across earlier and current observations; they are not a requirement to show every mentioned landmark in the final image.
- Stop beside the confirmed target when available evidence supports a distance of no more than 20 m (d <= 20 m, including the boundary). Distance means the three-dimensional straight-line separation between the UAV position and the target reference position: d = sqrt((x_uav - x_target)^2 + (y_uav - y_target)^2 + (z_uav - z_target)^2), in metres, including height difference. Estimate this relation from available observations and measurements. A camera-to-visible-surface depth sample is not automatically this reference-position distance, and a historical measurement describes its acquisition time rather than the current distance.
- Account for the camera position, the sampled surface's offset from the target reference position, and estimation uncertainty. A surface sample below 20 m alone does not establish arrival if compatible reference-position distances can still exceed 20 m. Resolve that uncertainty with applicable geometry or a supported closer approach before reporting completion; do not round the requirement to approximately 20 m.
- Once target identity and the stopping relation are supported, do not move farther to include background landmarks, centre the image or improve composition. Preserve any final spatial relation or route order explicitly requested by the instruction.
- Physical ground contact and calling uav.land are not required unless the instruction independently requests landing.
""".strip()
