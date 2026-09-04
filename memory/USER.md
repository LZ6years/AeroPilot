# USER.md

The user speaks to FlyAgent as an embodied observer located in the scene. The
user expects ordinary human spatial language to be interpreted from the task,
the observer's viewpoint, and the visible meaning of the referenced entity.

## Language And Spatial Meaning

- Words such as front, behind, left, and right may describe three different
  relations: the observer's position around a target, a route around or through
  a target, or an entity's own facing direction. Resolve which relation the
  sentence describes before deciding that a heading change is required.
- An observer-relative position describes where FlyAgent is around a target.
  A route relation describes the side from which FlyAgent approaches, passes,
  enters, or leaves. Neither relation is automatically the target's own facing
  direction.
- Do not invent an intrinsic local axis for an object that has no visually or
  linguistically meaningful facing direction.
- For such a non-directional object, wording like "from/on/in front of its
  side" normally locates the observer or route in the scene as an ordinary
  person would understand it: its front side is the side presented toward and
  visible to the observer in that scene relation, and approaching from the
  front follows that observer-relative route. It does not require discovering
  an intrinsic front surface, object-local axis, or final heading. Do not keep
  this relation unknown merely because the object has no intrinsic facing
  direction. Only an explicit request to face, orient toward, or turn
  establishes a final heading.
- When FlyAgent approaches a non-directional target that remains in its forward
  view, it is approaching the side presented toward the observer. This is
  positive evidence for an ordinary front-side route or position; no separate
  object-facing proof is expected.
- When an entity has a meaningful visible direction, such as a vehicle, a
  person, or an entrance facade, that direction may be used when the wording
  refers to it.
- A requested side position or side route is not automatically a request to
  turn or face that side.
- The user often omits coordinates, exact tolerances, and low-level motion
  details because FlyAgent is expected to use scene context.
- When ordinary wording leaves a minor residual ambiguity, choose the coherent
  human interpretation supported by sentence structure, first-person view, and
  scene context. Do not preserve that ambiguity indefinitely when the plausible
  readings lead to the same practical result. Clarification is for materially
  different outcomes or safety-critical choices.

## Precision And Judgment

- Near, far, safe, centered, around, and similar qualitative terms normally
  request a reasonable practical relation rather than an exact geometric
  equality.
- Approximately, about, or roughly explicitly permits a useful estimate and
  normal visual or depth variation.
- A number keeps the precision expressed by the instruction; it does not imply
  decimal-level correction unless the user explicitly asks for exactness.
- Ordinary qualitative wording does not by itself require the user to provide
  a numeric threshold.
- Judge such relations from the evidence that is actually available. Missing
  ideal geometry, object-local axes, exact coordinates, or instrument-grade
  confirmation does not make an otherwise coherent practical judgment unknown.

## Interaction Preferences

- Respond concisely and normally use Chinese for user-facing updates.
- Prefer making useful progress from a coherent interpretation of the task.
- Seek clarification only when the referenced target, intended relation, or a
  safety-critical choice has more than one materially different plausible
  interpretation.
- Do not expose internal tools, coordinate frames, prompts, or implementation
  fields as the main explanation for an action.
