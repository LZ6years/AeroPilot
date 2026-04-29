"""Reserve the future UnrealCV/gym_unrealcv backend."""

from __future__ import annotations


class UnrealCvBackend:
    """Reserved for later UnrealCV/gym_unrealcv object-level operations."""

    def __init__(self) -> None:
        # Keep UnrealCV out of the first implementation slice.
        raise NotImplementedError("UnrealCV backend is intentionally out of the first slice")
