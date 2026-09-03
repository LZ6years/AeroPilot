"""Test first-person acceptance helpers without launching Unreal Engine.

The production Backend only captures generic segmentation. These tests lock
the developer probe's separate responsibility: a deployment that explicitly
requests first-person validation must reject frames containing the controlled
Actor's mask color.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.perception.raw_observation import RawObservation
from tests.manual.unrealcv_local_actor_probe import (
    _parse_object_color,
    _required_agl,
    _visual_quality,
)


def _observation(tmp_path: Path, segmentation: np.ndarray) -> RawObservation:
    """Create non-degenerate artifacts matching the manual probe contract."""

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    segmentation_path = tmp_path / "segmentation.png"

    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(25, dtype=np.uint8).reshape(5, 5) * 8
    rgb[:, :, 1] = 80
    depth = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(depth_path, depth)
    Image.fromarray(segmentation.astype(np.uint8), mode="RGB").save(
        segmentation_path
    )

    return RawObservation(
        scene_instance_id="scene-1",
        agent_id="drone-1",
        modalities=["rgb", "depth", "segmentation", "pose"],
        frames={
            "rgb": {"artifact_path": str(rgb_path)},
            "depth": {"artifact_path": str(depth_path)},
            "segmentation": {"artifact_path": str(segmentation_path)},
        },
    )


def test_first_person_probe_accepts_hidden_controlled_actor(tmp_path: Path) -> None:
    segmentation = np.full((5, 5, 3), (10, 20, 30), dtype=np.uint8)

    quality = _visual_quality(
        _observation(tmp_path, segmentation),
        require=True,
        self_actor_color=(200, 100, 50),
        require_self_hidden=True,
    )

    assert quality["segmentation"]["controlled_actor_pixels"] == 0
    assert quality["segmentation"]["controlled_actor_hidden"] is True


def test_first_person_probe_rejects_visible_controlled_actor(tmp_path: Path) -> None:
    actor_color = (200, 100, 50)
    segmentation = np.full((5, 5, 3), (10, 20, 30), dtype=np.uint8)
    segmentation[2:, 1:3] = actor_color

    with pytest.raises(AssertionError, match="Controlled Actor is still visible"):
        _visual_quality(
            _observation(tmp_path, segmentation),
            require=True,
            self_actor_color=actor_color,
            require_self_hidden=True,
        )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("23 127 47", (23, 127, 47)),
        ("(R=23,G=127,B=47,A=255)", (23, 127, 47)),
    ],
)
def test_object_color_parser_accepts_unrealcv_encodings(
    response: str,
    expected: tuple[int, int, int],
) -> None:
    assert _parse_object_color(response, "BP_Drone01_2") == expected


def test_agl_probe_helper_requires_finite_optional_pose_value() -> None:
    assert _required_agl({"agl_m": 1.25}) == 1.25
    with pytest.raises(AssertionError, match="landed_origin_z_m"):
        _required_agl({"z": 2.0})
    with pytest.raises(AssertionError, match="finite"):
        _required_agl({"agl_m": float("nan")})
