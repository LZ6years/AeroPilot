"""Open-vocabulary perception components for standalone STMR probes."""

from .grounding_dino_backend import DetectionResult, GroundingDinoBackend
from .dynamic_targets import (
    DynamicTargetFilterConfig,
    apply_dynamic_target_policy,
    build_dynamic_candidate,
    should_write_candidate,
)
from .landmark_extractor import Landmark, extract_landmarks
from .model_landmark_extractor import (
    ModelLandmarkExtractionResult,
    ModelLandmarkExtractor,
    ModelLandmarkExtractorError,
    parse_landmarks_response,
)
from .open_vocab_writer import OpenVocabMaskResult, write_open_vocab_mask
from .sam_backend import SamBackend, SamMaskResult
from .tfidf_filter import TfidfMatch, filter_detections_by_tfidf, tfidf_similarity

__all__ = [
    "DetectionResult",
    "DynamicTargetFilterConfig",
    "GroundingDinoBackend",
    "Landmark",
    "ModelLandmarkExtractionResult",
    "ModelLandmarkExtractor",
    "ModelLandmarkExtractorError",
    "OpenVocabMaskResult",
    "SamBackend",
    "SamMaskResult",
    "TfidfMatch",
    "apply_dynamic_target_policy",
    "build_dynamic_candidate",
    "extract_landmarks",
    "filter_detections_by_tfidf",
    "parse_landmarks_response",
    "should_write_candidate",
    "tfidf_similarity",
    "write_open_vocab_mask",
]
