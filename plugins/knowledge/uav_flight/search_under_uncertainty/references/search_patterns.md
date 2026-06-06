Useful search patterns should be selected by scene context:

- Sweep search: small yaw steps when the UAV is stationary and needs new visual
  coverage.
- Peek-and-confirm: short movement or yaw change when an obstacle or corner may
  hide the target.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when several similar targets exist, gather another
  view instead of committing early.

Avoid repetitive left-right oscillation unless the last observation clearly
showed that a small corrective turn is needed.
