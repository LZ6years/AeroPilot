"""Construct stateless model clients without leaking providers into callers."""
from __future__ import annotations

from src.llm.client import DefaultModelClient
from src.llm.contracts import ModelClient, ProviderSettings


def create_model_client(settings: ProviderSettings) -> ModelClient:
    """Create the configured transport behind FlyAgent's stable client seam."""

    if settings.provider == "openai_compatible":
        return DefaultModelClient(settings)
    raise ValueError(f"Unsupported model provider: {settings.provider}")
