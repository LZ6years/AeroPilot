"""Reserve the future networked EnvClient implementation."""

from __future__ import annotations


class RemoteEnvClient:
    """Placeholder for a future networked EnvClient implementation."""

    def __init__(self, endpoint: str) -> None:
        # Store the endpoint that a future RPC client will call.
        self.endpoint = endpoint

    def health_check(self) -> bool:
        # Make the missing RPC implementation explicit for now.
        raise NotImplementedError("RemoteEnvClient is reserved for a later RPC phase")
