---
name: basic_flight_control
description: "Applicable needs: choosing translation, yaw, ascent/descent or an angled flight path, especially before movement response is familiar; Decision questions: how far or how much to turn now, and how to adjust after actual motion and visual change; Contents: reference frames, short initial probes, velocity-duration scaling, vertical control and feedback calibration; Limits: does not establish landmark identity, target distance, route completion or landing."
summary: Select for Basic handling and movement-response learning; select when action scale or flight components are the active question.
---

# Basic Flight Control

## Application Boundary

Use to select horizontal movement, yaw, ascent/descent or angled flight. Choose
an observation position before speed and duration.

## Evidence Model

Yaw changes heading; translation changes position. Body directions rotate with
heading. Speed times duration estimates requested displacement, not actual motion
or target distance. Pair available images before and after the same action with
its measured motion; track the same cue to calibrate the response.
Image growth is not metric range. Net displacement is not path length. Height
change is not clearance; missing clearance is unknown. Moving a fixed camera
changes viewpoint, not pitch.

## Decision Procedure

1. Bound travel by supported clear space along the whole movement path, not only
   visibility of its endpoint. When a nearby surface hides the continuation or
   the vehicle is not aligned with a passage, shorten travel or adjust position
   and heading to inspect it. A single surface-depth sample or action success
   does not establish clearance along the path. Parameter limits are not defaults.
2. Choose distance or bearing purpose, then parameters. Preserve an explicit turn
   angle; otherwise use view overlap to inspect the needed bearing.
3. Compare requested and actual motion with that action's before/after images.
   Small change along a supported continuation can justify longer progress. Rapid
   growth, opening road edges or nearby surfaces call for earlier observation.
   Recalibrate after heading, height or target changes. If either image is missing,
   use retained motion and the present view without claiming a visual comparison.
4. Preserve instructed height and angled-path components. Shorten near the required
   relation or a surface. See [angled motion](references/angled_vertical_motion.md)
   and [handling examples](references/handling_examples.md) when needed.

Worked example: a road is visible beyond a nearby building, but the straight path
toward it intersects the facade. Choose a reachable viewing position along known
clear space, adjust position or heading, and inspect the connected passage before
advancing. The visible road is not evidence that the direct path to it is clear.

## Completion And Stop

Check motion and view before another action. Use reasonable numeric tolerance.
Clearing an obstacle requires a supported above-obstacle
relation along the passage, not accumulated climb. Action success is not task completion.

## Recovery

Update the response estimate when motion differs. Unknown execution requires new
evidence before retrying. Use observation or abnormal recovery when visibility or
orientation is uncertain; shorten corrections near surfaces.
