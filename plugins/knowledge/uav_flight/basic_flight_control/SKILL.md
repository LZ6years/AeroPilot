---
name: basic_flight_control
description: "Applicable needs: choosing translation, yaw, ascent/descent or an angled flight path, especially before movement response is familiar; Decision questions: how far or how much to turn now, and how to adjust after actual motion and visual change; Contents: reference frames, short initial probes, velocity-duration scaling, vertical control and feedback calibration; Limits: does not establish landmark identity, target distance, route completion or landing."
summary: Select for Basic handling and movement-response learning; select when action scale or flight components are the active question.
---

# Basic Flight Control

## Application Boundary

Select horizontal movement, yaw, ascent/descent or angled flight. Choose
an observation position before speed and duration.

## Evidence Model

Yaw changes heading; translation changes position. Body directions rotate with
heading. Speed times duration estimates requested displacement, not actual motion
or target range. Pair the same action's before/after images and measured motion;
track the same cue to calibrate response.
Image growth is not metric range. Net displacement is not path length. Height
change is not clearance; missing clearance is unknown. Moving a fixed camera
changes viewpoint, not pitch.

## Decision Procedure

1. Bound travel by supported clear space along the whole movement path, not endpoint
   visibility. For hidden or misaligned passages, shorten travel or adjust position
   and heading to inspect. One depth sample or action success does not establish
   path clearance. Parameter limits are not defaults.
2. Choose distance or bearing purpose, then parameters. Preserve an explicit turn
   angle; otherwise use view overlap to inspect the needed bearing.
3. Compare requested and actual motion with that action's before/after images.
   Small change along a supported continuation can justify longer progress. Rapid
   growth, opening edges or nearby surfaces call for earlier observation.
   Recalibrate after heading, height or target changes. Missing images permit only
   retained motion and the present view, not a visual comparison.
4. Preserve instructed height and angled-path components. Shorten near the required
   relation or a surface. See [angled motion](references/angled_vertical_motion.md)
   and [handling examples](references/handling_examples.md) when needed.

Worked example: a direct path intersects the facade; inspect a connected passage
from known clear space before advancing toward the visible road.

## Completion And Stop

Check motion and view before another action. Use reasonable numeric tolerance.
Clearing an obstacle requires a supported above-obstacle
relation along the passage, not accumulated climb. Action success is not task completion.

## Recovery

Update estimates when motion differs. Unknown execution needs new evidence before
retrying. Use observation or abnormal recovery for uncertain visibility or
orientation; shorten corrections near surfaces.
