"""Test the user-facing EnvClient demo's offline configuration entry points."""

from __future__ import annotations

import pytest

from demos.simulator_env_client_demo import (
    DEFAULT_CONFIG,
    DEFAULT_OBSERVATION_PROFILE,
    _visual_quality,
    build_parser,
    list_maps,
)
from src.perception.raw_observation import RawObservation


FOUR_MAPS = {
    "demo_local_airsim_cv",
    "demo_remote_airsim_cv",
    "demo_local_unrealcv",
    "demo_remote_unrealcv",
}


def test_demo_example_config_lists_all_four_routing_pairs():
    rows = list_maps(DEFAULT_CONFIG)

    assert {row["map_id"] for row in rows} == FOUR_MAPS
    assert {
        (row["connection"], row["protocol"]) for row in rows
    } == {
        ("local", "airsim"),
        ("remote", "airsim"),
        ("local", "unrealcv"),
        ("remote", "unrealcv"),
    }

def test_demo_defaults_to_managed_and_zero_motion():
    args = build_parser().parse_args([])

    assert args.execution == "managed"
    assert args.move_forward_m == 0.0
    assert args.observation_profile == DEFAULT_OBSERVATION_PROFILE


def test_demo_rejects_removed_dry_run_execution_mode():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--execution", "dry-run"])


def test_strict_visual_quality_rejects_constant_artifacts(tmp_path):
    import numpy as np
    from PIL import Image

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(
        rgb_path
    )
    np.save(depth_path, np.ones((4, 4), dtype=np.float32))
    observation = RawObservation(
        "scene_1",
        "uav_1",
        ["rgb", "depth"],
        frames={
            "rgb": {"artifact_path": str(rgb_path)},
            "depth": {"artifact_path": str(depth_path)},
        },
    )

    quality = _visual_quality(observation, ["rgb", "depth"])

    assert not quality["ok"]
    assert not quality["frames"]["rgb"]["ok"]
    assert not quality["frames"]["depth"]["ok"]
