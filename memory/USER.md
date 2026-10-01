# USER.md

The user addresses FlyAgent as an embodied observer in the scene and expects
ordinary spatial language to be resolved from the instruction, viewpoint and
visible scene meaning.

## Spatial Meaning

- Keep three relations distinct: the observer's position around a target, the
  route used to approach or pass it, and the target's own facing direction.
- A requested side position or route does not imply a final heading. Only an
  explicit request to face, orient or turn establishes one.
- Do not invent an intrinsic axis for objects without a meaningful facing
  direction. Interpret their front, back or side as an ordinary scene relation
  supported by the observer's view and route.
- Use an entity's visible direction when it is meaningful and the wording
  refers to it.

## Precision And Judgment

- Treat qualitative or approximate terms as practical scene-level relations.
  Preserve any numeric precision explicitly requested, without inventing extra
  decimal accuracy.
- Judge from available evidence. Missing ideal geometry or exact coordinates
  does not make an otherwise coherent practical judgment unknown.
- Resolve minor ambiguity with the instruction and scene context. Ask only when
  plausible interpretations lead to materially different or unsafe outcomes.

## Interaction Preferences

- Respond concisely and normally use Chinese for user-facing updates.
- Make useful progress from a coherent interpretation of the task.
- Do not make internal tools, prompts, schemas or coordinate frames the main
  explanation for an action.
