"""Research perception helpers for custom_map preprocessing.

These modules generate run-memory artifacts before the custom_map context module
reads memory. They are intentionally separate from context building so model
downloads, GPU use, and perception failures do not affect default planner
context construction.
"""

from .closed_set_segmentation import ClosedSetSegmentationConfig, ClosedSetSegmentationResult, run_closed_set_segmentation
from .huggingface_backend import (
    ADE_MASK2FORMER_LARGE,
    ADE_ONEFORMER_DINAT_LARGE,
    ADE_ONEFORMER_SWIN_LARGE,
    ADE_SEGFORMER_B5,
    CITYSCAPES_MASK2FORMER_LARGE,
    CITYSCAPES_ONEFORMER_DINAT_LARGE,
    CITYSCAPES_ONEFORMER_SWIN_LARGE,
    CITYSCAPES_SEGFORMER_B2,
    MAPILLARY_MASK2FORMER_LARGE,
    HuggingFaceSegmentationBackend,
)
from .segformer_backend import DEFAULT_SEGFORMER_MODEL, SegFormerBackend

__all__ = [
    "CITYSCAPES_MASK2FORMER_LARGE",
    "ADE_MASK2FORMER_LARGE",
    "ADE_ONEFORMER_DINAT_LARGE",
    "ADE_ONEFORMER_SWIN_LARGE",
    "ADE_SEGFORMER_B5",
    "CITYSCAPES_ONEFORMER_DINAT_LARGE",
    "CITYSCAPES_ONEFORMER_SWIN_LARGE",
    "CITYSCAPES_SEGFORMER_B2",
    "ClosedSetSegmentationConfig",
    "ClosedSetSegmentationResult",
    "DEFAULT_SEGFORMER_MODEL",
    "HuggingFaceSegmentationBackend",
    "MAPILLARY_MASK2FORMER_LARGE",
    "SegFormerBackend",
    "run_closed_set_segmentation",
]
