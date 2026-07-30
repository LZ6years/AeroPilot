"""Define the public process-ownership choice for simulator scene control.

MapPackage answers where a simulator lives and which protocol it speaks.
SimulationExecutionMode answers whether this run may start and stop a simulator
process. Simulator SDK/RPC access is always real; offline configuration
inspection is provided by commands that do not call ``start_scene``.
"""

from __future__ import annotations

from enum import Enum


class SimulationExecutionMode(str, Enum):
    """Describe whether one SimulationServer may own simulator processes."""

    ATTACH = "attach"
    MANAGED = "managed"

    @classmethod
    def parse(
        cls, value: "SimulationExecutionMode | str"
    ) -> "SimulationExecutionMode":
        """Normalize public string input and reject ambiguous values early."""

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as exc:
            available = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported simulation execution mode {value!r}; "
                f"available: {available}"
            ) from exc

    @property
    def may_start_process(self) -> bool:
        """Return whether this run may create an owned UE process."""

        return self is self.MANAGED
