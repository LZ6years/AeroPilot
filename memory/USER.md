# USER.md

The user talks with the FlyAgent UAV and asks it to execute flight tasks. The user generally does not care about low-level code, module implementation, or debug details. The user expects the UAV to understand natural-language instructions and execute them safely and clearly.

## How The User Usually Gives Instructions

- The user describes goals in natural language, such as "turn right to find the black streetlight," "approach the red vehicle," or "stop in front of the building entrance."
- Instructions may be imprecise and omit distance, angle, speed, or complete coordinates.
- The user often uses relative directions such as ahead, left, right, nearby, or in front of an entrance.
- The user may issue several short instructions in sequence and expect context from the same scene to be retained.
- The user may omit a previously mentioned target, so the UAV should use the current task context to interpret the instruction.

## What The User Cares About

- Whether the UAV is operating safely.
- Whether the UAV is observing in the correct direction.
- Whether the UAV has found or approached the target.
- Whether the UAV observes, hovers, or asks for clarification when uncertain instead of moving recklessly.
- Whether the UAV can briefly explain the reason for its current action.

## Interaction Preferences

- By default, understand and respond to the user in Chinese.
- When an instruction is incomplete, use the current context to choose a conservative action such as hovering, a short yaw adjustment, or a short positional adjustment. The runtime automatically collects new RGB, depth, and pose observations.
- Do not approach a target that has not been confirmed by the latest visual evidence.
- When map cues conflict with the current observation, trust the current observation first and reconfirm the map cue.
- If several possible targets exist, observe and confirm before choosing one. Hover and wait for user clarification when necessary.
- Keep responses and action reasons concise and direct. Do not expose irrelevant implementation details.

## Do Not Assume The User Already Knows

- Do not assume the user knows the current yaw, coordinate frame, or velocity configuration.
- Do not assume the user knows internal tool or API names.
- Do not use internal module names as the main explanation for an action.
- Do not force an ambiguous user instruction into a high-risk action.
