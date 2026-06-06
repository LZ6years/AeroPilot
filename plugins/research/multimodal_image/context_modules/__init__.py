"""Context modules contributed by the multimodal image research plugin."""

from plugins.research.multimodal_image.context_modules.client import (
    build_visual_caption_payload,
    call_visual_caption_model,
    caption_model_config,
)
from plugins.research.multimodal_image.context_modules.constants import (
    BANNED_ACTION_TERMS,
    DEFAULT_VISUAL_DESCRIPTION_MODEL,
    SCHEMA_VERSION,
    STRATEGY,
)
from plugins.research.multimodal_image.context_modules.description import (
    normalize_visual_description,
    visual_description_text,
)
from plugins.research.multimodal_image.context_modules.image import PluginContextModule

MultimodalImageContextModule = PluginContextModule

__all__ = [
    "BANNED_ACTION_TERMS",
    "DEFAULT_VISUAL_DESCRIPTION_MODEL",
    "SCHEMA_VERSION",
    "STRATEGY",
    "MultimodalImageContextModule",
    "PluginContextModule",
    "build_visual_caption_payload",
    "call_visual_caption_model",
    "caption_model_config",
    "normalize_visual_description",
    "visual_description_text",
]
