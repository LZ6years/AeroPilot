"""Open-vocabulary perception components for standalone STMR probes."""

from plugins.research.stmr_map.profiling import StageProfiler, artifact_size_bytes

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
from .session import StmrFramePerception, StmrPerceptionSession
from .tfidf_filter import TfidfMatch, filter_detections_by_tfidf, tfidf_similarity
from .urban_priors import (
    DEFAULT_UAV_PRIOR_NAME,
    MAX_LANDMARK_QUERIES,
    PRIOR_SETS,
    URBAN_UAV_PRIOR_V1_NAME,
    assemble_landmark_queries,
    default_uav_prior,
    normalize_landmark_query,
    urban_uav_prior_v1,
)

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
    "StmrFramePerception",
    "StmrPerceptionSession",
    "StageProfiler",
    "TfidfMatch",
    "DEFAULT_UAV_PRIOR_NAME",
    "MAX_LANDMARK_QUERIES",
    "PRIOR_SETS",
    "URBAN_UAV_PRIOR_V1_NAME",
    "apply_dynamic_target_policy",
    "assemble_landmark_queries",
    "artifact_size_bytes",
    "build_dynamic_candidate",
    "default_uav_prior",
    "extract_landmarks",
    "filter_detections_by_tfidf",
    "parse_landmarks_response",
    "should_write_candidate",
    "tfidf_similarity",
    "normalize_landmark_query",
    "urban_uav_prior_v1",
    "write_open_vocab_mask",
]
