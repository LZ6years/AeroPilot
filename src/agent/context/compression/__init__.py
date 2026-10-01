"""Public contracts and model implementation for history compression."""

from src.agent.context.compression.contracts import (
    CompactionCallResult,
    CompressionCandidate,
    CompressedHistorySection,
    CompressionDraft,
    CompressionRequest,
    CompressionValidationResult,
    CompressionWindow,
    EvidenceCatalogEntry,
    HistoryCompressionConfig,
    HistoryCompactor,
    HistoryMediaEntry,
    HistoryProjectionState,
)
from src.agent.context.compression.coordinator import HistoryCompactionCoordinator
from src.agent.context.compression.model_compactor import ModelHistoryCompactor

__all__ = [
    "CompactionCallResult",
    "CompressionCandidate",
    "CompressedHistorySection",
    "CompressionDraft",
    "CompressionRequest",
    "CompressionValidationResult",
    "CompressionWindow",
    "EvidenceCatalogEntry",
    "HistoryCompressionConfig",
    "HistoryCompactionCoordinator",
    "HistoryCompactor",
    "HistoryMediaEntry",
    "HistoryProjectionState",
    "ModelHistoryCompactor",
]
