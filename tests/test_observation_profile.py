"""Test run-level observation profiles independently of MapPackage."""

from __future__ import annotations

import json

import pytest

from src.env_client.observation_profile import (
    ObservationProfile,
    load_observation_profile,
)


def test_default_profile_uses_generic_scene_and_perspective_depth():
    profile = ObservationProfile()

    assert profile.modalities == ("rgb", "depth", "pose")
    assert profile.capture_sources["rgb"] == "scene"
    assert profile.capture_sources["depth"] == "depth_perspective"


def test_profile_loader_keeps_capture_selection_outside_map_package(tmp_path):
    path = tmp_path / "observation.json"
    path.write_text(
        json.dumps(
            {
                "modalities": ["rgb", "depth", "pose"],
                "capture_sources": {
                    "rgb": "scene",
                    "depth": "depth_vis",
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_observation_profile(path)

    assert profile.capture_sources["depth"] == "depth_vis"
    assert profile.to_dict()["modalities"] == ["rgb", "depth", "pose"]


def test_profile_rejects_duplicate_modalities():
    with pytest.raises(ValueError, match="must be unique"):
        ObservationProfile.from_dict({"modalities": ["rgb", "rgb"]})
