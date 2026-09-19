"""TravelUAV transfer-evaluation protocol identity and parser context.

The protocol deliberately describes FlyAgent's front-camera, native-action
setting.  It must not be confused with the official five-view assistant-guided
TravelUAV leaderboard protocol.
"""

from __future__ import annotations


TRAVEL_UAV_PROTOCOL = (
    "travel_uav.flyagent_native_actions.front_view.no_assistant.v1"
)

TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT = """TravelUAV transfer-evaluation protocol:
- Navigate using the instruction and the available front-view observations until the described target object and its surrounding spatial relation are visually confirmed.
- A distant or ambiguous candidate sighting is not sufficient for completion; finish only after reaching a nearby, judgeable target region and then end the episode.
- Physical ground contact and calling uav.land are not required unless the instruction independently requests landing.
""".strip()
