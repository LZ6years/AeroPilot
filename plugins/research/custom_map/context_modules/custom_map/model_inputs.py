"""Encode rendered custom map artifacts for planner model inputs."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


def custom_map_model_input(path: Path) -> dict[str, Any] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image",
        "input_id": "custom_map_1",
        "role": "custom_map",
        "usage": "Use as compressed spatial evidence for path, pose, and semantic map context.",
        "mime_type": "image/png",
        "encoding": "data_url",
        "data": f"data:image/png;base64,{encoded}",
    }

