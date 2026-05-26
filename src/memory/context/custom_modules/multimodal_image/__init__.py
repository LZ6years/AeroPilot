"""Phase-aware custom image context module package."""

from __future__ import annotations

from .client import (
    build_visual_caption_payload,
    call_visual_caption_model,
    caption_model_config,
)
from .constants import (
    BANNED_ACTION_TERMS,
    DEFAULT_VISUAL_DESCRIPTION_MODEL,
    SCHEMA_VERSION,
    STRATEGY,
)
from .description import normalize_visual_description, visual_description_text
from .module import MultimodalImageContextModule

__all__ = [
    "BANNED_ACTION_TERMS",
    "DEFAULT_VISUAL_DESCRIPTION_MODEL",
    "SCHEMA_VERSION",
    "STRATEGY",
    "MultimodalImageContextModule",
    "build_visual_caption_payload",
    "call_visual_caption_model",
    "caption_model_config",
    "normalize_visual_description",
    "visual_description_text",
]
