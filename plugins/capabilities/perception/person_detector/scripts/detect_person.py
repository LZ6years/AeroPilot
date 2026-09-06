"""Mock person detector handler for the callable skill demo."""

from __future__ import annotations


def handle(payload: dict) -> dict:
    args = dict(payload.get("args", {}))
    runtime_context = dict(payload.get("runtime_context", {}))
    latest_observation = dict(runtime_context.get("latest_observation", {}))
    frames = latest_observation.get("frames") if isinstance(latest_observation, dict) else {}
    rgb = frames.get("rgb") if isinstance(frames, dict) else None
    has_rgb = isinstance(rgb, dict)
    observation_id = str(latest_observation.get("observation_id") or "")
    requested_observation_id = str(args.get("observation_id") or "")
    if requested_observation_id != observation_id:
        raise ValueError(
            "person detector only accepts the latest observation_id; "
            f"requested={requested_observation_id!r}, latest={observation_id!r}"
        )
    min_confidence = float(args.get("min_confidence", 0.5))
    confidence = max(min_confidence, 0.51) if has_rgb else 0.0
    return {
        "mock": True,
        "image_source": args.get("image_source", "latest_rgb"),
        "observation_id": observation_id,
        "frame_id": rgb.get("id") if has_rgb else None,
        "has_rgb": has_rgb,
        "rgb_metadata": dict(rgb) if has_rgb else None,
        "detections": [
            {
                "class": "person",
                "confidence": round(confidence, 3),
                "bbox": [0.42, 0.34, 0.58, 0.72],
                "center": [0.5, 0.53],
                "source": "mock_person_detector",
            }
        ]
        if has_rgb
        else [],
    }
