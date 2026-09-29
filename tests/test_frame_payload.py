"""File and in-memory frame parity across capture, context, and depth tools."""

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.airsim.capture import AirSimCapture
from src.agent.context.media import MediaHydrator
from src.memory.context.media_refs import image_media_ref
from src.perception.frame_payload import load_rgb_frame, load_depth_frame
from plugins.capabilities.visual_grounding.locate.scripts.depth import load_metric_depth


def capture(tmp_path, source, *, save, raw=b"", floats=None, width=2, height=2):
    component = AirSimCapture("front", "drone", tmp_path, save)
    response = SimpleNamespace(width=width, height=height, image_data_uint8=raw, image_data_float=floats or [])
    return component._frame_metadata("rgb" if source == "scene" else "depth", source, component._capture_spec(source), response)


def test_capture_memory_and_file_have_identical_pixels_and_depth(tmp_path):
    results = []
    for save in (False, True):
        root = tmp_path / str(save)
        rgb = capture(root, "scene", save=save, raw=bytes(range(12)))
        depth = capture(root, "depth_perspective", save=save, floats=[1, 2, 3, 4])
        assert root.exists() is save
        if save:
            ref = image_media_ref(
                rgb["artifact_path"],
                observation_id="obs_1",
                frame_id="frame_1",
                role="current_observation",
            )
            assert MediaHydrator().hydrate(ref).data_url.startswith(
                "data:image/png;base64,"
            )
        np.testing.assert_array_equal(load_depth_frame(depth, require_metric=True), load_metric_depth(depth))
        results.append((load_rgb_frame(rgb).tobytes(), load_depth_frame(depth), rgb["base64"], depth["base64"]))
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2:] == results[1][2:]


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("source,raw,floats,width", [
    ("scene", b"", None, 2), ("scene", b"bad", None, 2),
    ("scene", bytes(12), None, 0), ("depth_perspective", b"", [1], 2),
    ("depth_vis", b"not-png", None, 2),
])
def test_invalid_capture_fails_without_artifact_dependency(tmp_path, save, source, raw, floats, width):
    with pytest.raises((ValueError, OSError)):
        capture(tmp_path, source, save=save, raw=raw, floats=floats, width=width)


def test_depth_vis_cannot_be_consumed_as_meters(tmp_path):
    frame = capture(tmp_path, "depth_vis", save=False, floats=[0, .25, .5, 1])
    assert frame["depth_unit"] == "normalized"
    with pytest.raises(ValueError, match="meters"):
        load_metric_depth(frame)


@pytest.mark.parametrize("payload", ["bad%%%", base64.b64encode(b"not an image").decode()])
def test_bad_payload_is_explicit_failure(payload):
    with pytest.raises((ValueError, OSError)):
        load_rgb_frame({"base64": payload})


def test_nested_metadata_and_missing_file_fall_back_to_valid_payload(tmp_path):
    frame = capture(tmp_path, "scene", save=False, raw=bytes(range(12)))
    assert load_rgb_frame({"path": "missing.png", "metadata": frame}).size == (2, 2)
