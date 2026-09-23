"""Deterministic tests for the replaceable visual-grounding plugin pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from plugins.capabilities.visual_grounding.locate.scripts.contracts import (
    CandidateMask,
    GroundedCandidate,
    VisualGroundingSettings,
)
from plugins.capabilities.visual_grounding.locate.scripts.service import (
    VisualGroundingService,
)


class FakeGrounder:
    def __init__(self, candidates: list[GroundedCandidate]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[Path, str]] = []

    def locate(self, rgb_path: Path, query: str) -> list[GroundedCandidate]:
        self.calls.append((rgb_path, query))
        return list(self.candidates)


class FakeSegmenter:
    def __init__(self, masks: list[CandidateMask] | None = None, *, error: Exception | None = None) -> None:
        self.masks = masks or []
        self.error = error
        self.calls = 0

    def segment(self, rgb_path: Path, candidates: list[GroundedCandidate]) -> list[CandidateMask]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.masks)


def test_memory_only_rgb_depth_are_consumed_without_backend_artifacts(tmp_path):
    import base64
    from io import BytesIO
    rgb = BytesIO()
    Image.new("RGB", (20, 20), (80, 100, 120)).save(rgb, format="PNG")
    depth = BytesIO()
    np.save(depth, np.full((20, 20), 7.0, dtype=np.float32))
    observation = {"observation_id": "obs_memory", "frames": {
        "rgb": {"base64": base64.b64encode(rgb.getvalue()).decode()},
        "depth": {"base64": base64.b64encode(depth.getvalue()).decode(), "depth_unit": "meter"},
    }}

    class ReadingGrounder(FakeGrounder):
        def locate(self, rgb_path, query):
            with Image.open(rgb_path) as image:
                assert image.size == (20, 20)
            return super().locate(rgb_path, query)

    service = VisualGroundingService(ReadingGrounder([_candidate()]), None, _settings())
    result = service.locate(observation=observation, observation_id="obs_memory",
        query="person", include_distance=True, run_dir=None, tool_call_id="fixture")
    assert result["data"]["candidates"][0]["distance_m"] == 7.0
    assert result["data"]["candidates"][0]["distance_status"] == "ok"
    assert not list(tmp_path.iterdir())


def _settings(**overrides) -> VisualGroundingSettings:
    values = {
        "grounding_backend": "fake",
        "grounding_model": "fake-grounder",
        "segmentation_backend": "fake",
        "segmentation_model": "fake-segmenter",
        "device": "cpu",
        "box_threshold": 0.25,
        "text_threshold": 0.25,
        "max_candidates": 5,
        "mask_erosion_ratio": 0.0,
        "box_inset_ratio": 0.2,
        "minimum_valid_depth_pixels": 9,
        "minimum_valid_depth_coverage": 0.5,
        "maximum_coherent_mad_m": 0.25,
    }
    values.update(overrides)
    return VisualGroundingSettings(**values)


def _observation(tmp_path, depth: np.ndarray | None = None) -> dict:
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (20, 20), (80, 100, 120)).save(rgb_path)
    frames = {"rgb": {"id": "frame_rgb", "path": str(rgb_path)}}
    if depth is not None:
        depth_path = tmp_path / "depth.npy"
        np.save(depth_path, depth.astype(np.float32))
        frames["depth"] = {
            "id": "frame_depth",
            "path": str(depth_path),
            "depth_unit": "meter",
        }
    return {"observation_id": "obs_0001", "frames": frames}


def _candidate() -> GroundedCandidate:
    return GroundedCandidate((4.0, 3.0, 16.0, 18.0), "person", 0.91)


def test_configured_threshold_filters_before_evidence_and_depth(tmp_path):
    candidates = [GroundedCandidate((4, 3, 16, 18), str(score), score)
                  for score in (0.399, 0.4, 0.401)]
    service = VisualGroundingService(FakeGrounder(candidates), None,
                                     _settings(box_threshold=0.4))
    result = service.locate(observation=_observation(tmp_path), observation_id="obs_0001",
                            query="object", include_distance=False,
                            run_dir=str(tmp_path), tool_call_id="threshold")
    assert [c["confidence"] for c in result["data"]["candidates"]] == [0.401, 0.4]
    assert result["data"]["audit_details"]["filtered_candidate_count"] == 1
    assert len(result["model_inputs"]) == 1


def test_all_low_candidates_do_not_segment_or_emit_empty_image(tmp_path):
    segmenter = FakeSegmenter(error=AssertionError("must not segment"))
    service = VisualGroundingService(
        FakeGrounder([GroundedCandidate((1, 1, 10, 10), "weak", 0.399)]),
        segmenter, _settings(box_threshold=0.4))
    result = service.locate(observation=_observation(tmp_path), observation_id="obs_0001",
                            query="object", include_distance=True,
                            run_dir=str(tmp_path), tool_call_id="empty")
    assert result["data"]["candidates"] == []
    assert not result["artifact_refs"] and not result["model_inputs"]
    assert segmenter.calls == 0


def test_empty_detection_handler_reports_uncertainty(monkeypatch, tmp_path):
    from plugins.capabilities.visual_grounding.locate.scripts import handler
    service = VisualGroundingService(FakeGrounder([]), None, _settings())
    monkeypatch.setattr(handler, "service_from_settings", lambda _: service)
    result = handler.handle({"args": {"observation_id": "obs_0001", "query": "object"},
                             "runtime_context": {"latest_observation": _observation(tmp_path)}})
    assert "remains uncertain" in result.message
    assert result.data["candidates"] == []


def test_detection_only_never_invokes_segmentation_or_depth(tmp_path):
    grounder = FakeGrounder([_candidate()])
    segmenter = FakeSegmenter(error=AssertionError("segmentation should not run"))
    service = VisualGroundingService(grounder, segmenter, _settings())

    result = service.locate(
        observation=_observation(tmp_path),
        observation_id="obs_0001",
        query="穿红衣服的人",
        include_distance=False,
        run_dir=str(tmp_path),
        tool_call_id="call_1",
    )

    assert segmenter.calls == 0
    assert grounder.calls[0][1] == "穿红衣服的人"
    assert result["data"]["candidates"][0]["distance_status"] == "not_requested"
    assert result["data"]["candidates"][0]["distance_m"] is None
    assert len(result["artifact_refs"]) == 1
    assert len(result["model_inputs"]) == 1


def test_distance_uses_segmented_surface_robust_median(tmp_path):
    depth = np.full((20, 20), 9.0, dtype=np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 7:13] = True
    depth[mask] = 2.86
    depth[7, 9] = 90.0
    depth[7, 10] = np.nan
    segmenter = FakeSegmenter([CandidateMask(0, mask, 0.95)])
    service = VisualGroundingService(
        FakeGrounder([_candidate()]), segmenter, _settings()
    )

    result = service.locate(
        observation=_observation(tmp_path, depth),
        observation_id="obs_0001",
        query="person",
        include_distance=True,
        run_dir=str(tmp_path),
        tool_call_id="call_2",
    )

    candidate = result["data"]["candidates"][0]
    audit = result["data"]["audit_details"]["candidates"][0]
    assert segmenter.calls == 1
    assert candidate["distance_m"] == 2.9
    assert candidate["distance_status"] == "ok"
    assert audit["sampling_source"] == "eroded_mask"
    assert audit["inlier_pixel_count"] < audit["valid_pixel_count"]


def test_segmentation_failure_falls_back_to_inset_detector_box(tmp_path):
    depth = np.full((20, 20), 3.4, dtype=np.float32)
    service = VisualGroundingService(
        FakeGrounder([_candidate()]),
        FakeSegmenter(error=RuntimeError("SAM unavailable")),
        _settings(),
    )

    result = service.locate(
        observation=_observation(tmp_path, depth),
        observation_id="obs_0001",
        query="person",
        include_distance=True,
        run_dir=None,
        tool_call_id=None,
    )

    candidate = result["data"]["candidates"][0]
    audit = result["data"]["audit_details"]["candidates"][0]
    assert candidate["distance_m"] == 3.4
    assert candidate["distance_status"] == "ok"
    assert audit["sampling_source"] == "box_interior"
    assert "SAM unavailable" in result["data"]["audit_details"]["segmentation_error"]


def test_multiple_depth_surfaces_return_unavailable_instead_of_a_number(tmp_path):
    depth = np.full((20, 20), 9.0, dtype=np.float32)
    depth[6:14, 6:10] = 2.0
    depth[6:14, 10:14] = 6.0
    mask = np.zeros((20, 20), dtype=bool)
    mask[6:14, 6:14] = True
    service = VisualGroundingService(
        FakeGrounder([_candidate()]),
        FakeSegmenter([CandidateMask(0, mask)]),
        _settings(),
    )

    result = service.locate(
        observation=_observation(tmp_path, depth),
        observation_id="obs_0001",
        query="person",
        include_distance=True,
        run_dir=None,
        tool_call_id=None,
    )

    candidate = result["data"]["candidates"][0]
    assert candidate["distance_m"] is None
    assert candidate["distance_status"] == "inconsistent_depth_surfaces"


def test_empty_candidates_and_stale_observation_are_explicit(tmp_path):
    service = VisualGroundingService(FakeGrounder([]), None, _settings())
    empty = service.locate(
        observation=_observation(tmp_path),
        observation_id="obs_0001",
        query="a red vehicle",
        include_distance=False,
        run_dir=None,
        tool_call_id=None,
    )
    assert empty["data"]["candidates"] == []

    try:
        service.locate(
            observation=_observation(tmp_path),
            observation_id="obs_old",
            query="a red vehicle",
            include_distance=False,
            run_dir=None,
            tool_call_id=None,
        )
    except ValueError as exc:
        assert "latest observation_id" in str(exc)
    else:
        raise AssertionError("stale observation must be rejected")
