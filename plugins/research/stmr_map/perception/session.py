"""Reusable GroundingDINO/SAM perception session for STMR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from plugins.research.stmr_map.profiling import StageProfiler, artifact_size_bytes

from .dynamic_targets import DynamicTargetFilterConfig, apply_dynamic_target_policy, should_write_candidate
from .grounding_dino_backend import DEFAULT_GROUNDING_DINO_MODEL, DetectionResult, GroundingDinoBackend
from .landmark_extractor import Landmark
from .open_vocab_writer import OpenVocabMaskResult, write_open_vocab_mask
from .sam_backend import DEFAULT_SAM_MODEL, SamBackend, SamMaskResult
from .tfidf_filter import TfidfMatch, filter_detections_by_tfidf
from .urban_priors import DEFAULT_UAV_PRIOR_NAME, assemble_landmark_queries


@dataclass
class StmrFramePerception:
    """Detector, text-filter, and mask outputs for one RGB frame."""

    detections: list[DetectionResult]
    matches: list[TfidfMatch]
    masks: list[SamMaskResult]


class StmrPerceptionSession:
    """Own reusable model instances and the open-vocabulary policy stages."""

    def __init__(
        self,
        *,
        grounding_model: str = DEFAULT_GROUNDING_DINO_MODEL,
        sam_model: str = DEFAULT_SAM_MODEL,
        device: str = "auto",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        tfidf_threshold: float = 0.8,
        profiler: StageProfiler | None = None,
        grounding_backend: Any | None = None,
        sam_backend: Any | None = None,
    ) -> None:
        self.profiler = profiler
        self.tfidf_threshold = float(tfidf_threshold)
        self.grounding = grounding_backend or GroundingDinoBackend(
            grounding_model,
            device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            profiler=profiler,
        )
        self.sam = sam_backend or SamBackend(sam_model, device, profiler=profiler)

    def warmup(
        self,
        *,
        profiler: StageProfiler | None = None,
        rgb_path: str | Path | None = None,
        landmark_sets: list[list[Landmark]] | None = None,
    ) -> None:
        """Load both models and optionally warm representative query shapes."""

        active = profiler or self.profiler
        self.grounding.warmup(profiler=active)
        self.sam.warmup(profiler=active)
        if rgb_path is None:
            return
        for index, landmarks in enumerate(landmark_sets or []):
            self.infer_frame(
                rgb_path,
                landmarks,
                observation_id=f"warmup_{index:02d}",
                profiler=active,
            )

    def assemble_queries(
        self,
        task_landmarks: list[Landmark | str],
        *,
        prior: str = DEFAULT_UAV_PRIOR_NAME,
        max_queries: int = 16,
        profiler: StageProfiler | None = None,
    ) -> list[Landmark]:
        active = profiler or self.profiler
        if active is None:
            return assemble_landmark_queries(task_landmarks, prior=prior, max_queries=max_queries)
        with active.stage(
            "landmark_query_assembly",
            metadata={"prior": prior, "task_landmark_count": len(task_landmarks)},
        ) as stage:
            landmarks = assemble_landmark_queries(task_landmarks, prior=prior, max_queries=max_queries)
            stage.update(query_count=len(landmarks), queries=[item.label for item in landmarks])
            return landmarks

    def infer_frame(
        self,
        rgb_path: str | Path,
        landmarks: list[Landmark],
        *,
        observation_id: str | None = None,
        profiler: StageProfiler | None = None,
    ) -> StmrFramePerception:
        """Ground queries, filter phrases, and segment retained boxes."""

        active = profiler or self.profiler
        if active is not None and observation_id:
            active = active.child(observation_id=observation_id)
        prompts = [item.label for item in landmarks]
        detections = self.grounding.detect(rgb_path, prompts, profiler=active)
        if active is None:
            matches = filter_detections_by_tfidf(detections, landmarks, threshold=self.tfidf_threshold)
            kept_detections, kept_matches = _retained_detections(detections, matches)
        else:
            with active.stage(
                "tfidf_and_semantic_filter",
                metadata={"detection_count": len(detections), "query_count": len(prompts)},
            ) as stage:
                matches = filter_detections_by_tfidf(detections, landmarks, threshold=self.tfidf_threshold)
                kept_detections, kept_matches = _retained_detections(detections, matches)
                stage.update(
                    kept_detection_count=len(kept_detections),
                    rejected_detection_count=len(detections) - len(kept_detections),
                )
        masks = self.sam.segment(rgb_path, kept_detections, profiler=active)
        return StmrFramePerception(kept_detections, kept_matches, masks)

    def build_dynamic_candidates(
        self,
        builder: Callable[[], list[dict[str, Any]]],
        *,
        observation_id: str | None = None,
        profiler: StageProfiler | None = None,
    ) -> list[dict[str, Any]]:
        active = profiler or self.profiler
        if active is None:
            return builder()
        metadata = {"observation_id": observation_id} if observation_id else None
        with active.stage("depth_and_dynamic_candidate_stats", metadata=metadata) as stage:
            candidates = builder()
            stage.update(
                candidate_count=len(candidates),
                dynamic_candidate_count=sum(bool(item.get("is_dynamic")) for item in candidates),
            )
            return candidates

    def apply_dynamic_policy(
        self,
        frame_items: list[dict[str, Any]],
        *,
        config: DynamicTargetFilterConfig,
        profiler: StageProfiler | None = None,
    ) -> dict[str, Any]:
        active = profiler or self.profiler
        if active is None:
            return apply_dynamic_target_policy(frame_items, config=config)
        with active.stage("dynamic_track_policy", metadata={"frame_count": len(frame_items)}) as stage:
            summary = apply_dynamic_target_policy(frame_items, config=config)
            stage.update(
                candidate_count=summary.get("candidate_count", 0),
                rejected_count=summary.get("rejected_count", 0),
            )
            return summary

    def write_frame_artifacts(
        self,
        run_dir: str | Path,
        frame: dict[str, Any],
        masks: list[SamMaskResult],
        matches: list[TfidfMatch],
        *,
        candidates: list[dict[str, Any]],
        dynamic_filter_enabled: bool,
        rejected_masks: list[dict[str, Any]] | None = None,
        kept_non_primary_masks: list[dict[str, Any]] | None = None,
        primary_dynamic_track: dict[str, Any] | None = None,
        profiler: StageProfiler | None = None,
    ) -> OpenVocabMaskResult:
        """Paint only policy-approved masks and append the memory frame."""

        paint_masks: list[SamMaskResult] = []
        paint_matches: list[TfidfMatch] = []
        for candidate, mask_result, match in zip(candidates, masks, matches):
            if not should_write_candidate(candidate, dynamic_filter_enabled=dynamic_filter_enabled):
                continue
            paint_masks.append(mask_result)
            paint_matches.append(
                TfidfMatch(
                    detection_index=len(paint_matches),
                    phrase=match.phrase,
                    landmark=match.landmark,
                    similarity=match.similarity,
                    keep=match.keep,
                    score=match.score,
                )
            )

        active = profiler or self.profiler
        metadata = {
            "observation_id": str(frame.get("observation_id") or ""),
            "input_mask_count": len(masks),
            "painted_mask_count": len(paint_masks),
        }
        if active is None:
            return write_open_vocab_mask(
                run_dir,
                frame["path"],
                frame["observation_id"],
                paint_masks,
                paint_matches,
                force=True,
                dynamic_target_candidates=candidates,
                rejected_masks=rejected_masks,
                kept_non_primary_masks=kept_non_primary_masks,
                primary_dynamic_track=primary_dynamic_track,
            )
        with active.stage("mask_overlay_and_memory_write", metadata=metadata) as stage:
            result = write_open_vocab_mask(
                run_dir,
                frame["path"],
                frame["observation_id"],
                paint_masks,
                paint_matches,
                force=True,
                dynamic_target_candidates=candidates,
                rejected_masks=rejected_masks,
                kept_non_primary_masks=kept_non_primary_masks,
                primary_dynamic_track=primary_dynamic_track,
            )
            stage.update(
                mask_artifact_bytes=artifact_size_bytes(result.mask_path),
                overlay_artifact_bytes=artifact_size_bytes(result.overlay_path),
                boxes_artifact_bytes=artifact_size_bytes(result.boxes_path),
            )
            return result


def _retained_detections(
    detections: list[DetectionResult],
    matches: list[TfidfMatch],
) -> tuple[list[DetectionResult], list[TfidfMatch]]:
    kept = [item for item in matches if item.keep]
    kept_detections = [detections[item.detection_index] for item in kept]
    kept_matches = [
        TfidfMatch(
            detection_index=index,
            phrase=item.phrase,
            landmark=item.landmark,
            similarity=item.similarity,
            keep=item.keep,
            score=item.score,
        )
        for index, item in enumerate(kept)
    ]
    return kept_detections, kept_matches


__all__ = ["StmrFramePerception", "StmrPerceptionSession"]
