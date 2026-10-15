---
name: list_saved_places
description: "Recover saved place IDs and notes from task-local memory."
---

# List saved places

Use offset 0 and a limit of 1 to 20. Only actively saved places appear, in
numeric P order. Continue with next_offset until null. Choose an ID before
calling inspect_place in a later turn. Notes are historical interpretations;
automatic unsaved sites are absent. No photos or new observations are produced.
