---
name: conservative_low_confidence_flight
description: Use when visual evidence, target identity, distance, or free-space confidence is low and the UAV should act conservatively.
---

# Conservative Low Confidence Flight

## When To Use

Use this skill when the current task depends on uncertain visual evidence,
uncertain target identity, uncertain distance, or unclear free space.

## Rules

- Prefer hover, short yaw, or short movement over large irreversible actions.
- Do not approach people or objects when target identity or distance is uncertain.
- Use the next automatic observation to reduce uncertainty.
- Respect all action, yaw, speed, and runtime safety constraints.

## Do Not

- Do not use missing visual evidence as a reason to rush forward.
- Do not treat uncertainty as task completion.
- Do not override safety constraints with skill guidance.
