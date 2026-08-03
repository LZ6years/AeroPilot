"""Typed configuration for AerialVLN native-action evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AerialVlnEvalConfig:
    dataset_root: str
    split: str = "val_seen"
    output_dir: str = "evaluation/benchmarks/aerial_vln/outputs"
    run_root: str = "runs"
    max_cases: int | None = None
    scene_filter: list[str] | None = None
    map_config_path: str = "configs/maps/airvln_maps.json"
    scene_map: dict[str, str] = field(
        default_factory=lambda: {"1": "airvln_env_1"}
    )
    observation_profile_path: str | None = None
    execution_mode: str = "dry"
    headless: bool = True
    save_artifacts: bool = False
    generate_html_review: bool = True
    enable_action_planner_debug: bool = False
    env_path: str = ".env"
    parser_mode: str = "model"
    shutdown_policy: str = "land"
    runtime_loop_guard_windows: int = 40
    initialization_min_settle_sec: float = 0.0
    action_config_overrides: dict[str, float] = field(default_factory=dict)
    safety_overrides: dict[str, float] = field(default_factory=dict)
    upstream_reference_commit: str = "local_airvln_reference"
    dataset_version: str = "unspecified"

    @classmethod
    def from_file(cls, path: str | Path) -> "AerialVlnEvalConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("AerialVLN config root must be an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AerialVlnEvalConfig":
        dataset_root = str(data.get("dataset_root", "")).strip()
        if not dataset_root:
            raise ValueError("AerialVLN config requires dataset_root")
        execution_mode = str(data.get("execution_mode", "dry")).lower()
        if execution_mode not in {"dry", "attach", "managed"}:
            raise ValueError(
                "AerialVLN execution_mode must be dry, attach, or managed"
            )
        raw_scene_map = data.get("scene_map", {"1": "airvln_env_1"})
        if not isinstance(raw_scene_map, dict) or not raw_scene_map:
            raise ValueError("AerialVLN scene_map must be a non-empty object")
        scene_map = {
            str(scene_id): str(map_id)
            for scene_id, map_id in raw_scene_map.items()
            if str(scene_id).strip() and str(map_id).strip()
        }
        if not scene_map:
            raise ValueError("AerialVLN scene_map cannot be empty")
        scene_filter = data.get("scene_filter")
        return cls(
            dataset_root=dataset_root,
            split=str(data.get("split", "val_seen")),
            output_dir=str(
                data.get(
                    "output_dir",
                    "evaluation/benchmarks/aerial_vln/outputs",
                )
            ),
            run_root=str(data.get("run_root", "runs")),
            max_cases=_optional_int(data.get("max_cases")),
            scene_filter=(
                [str(item) for item in scene_filter]
                if isinstance(scene_filter, list)
                else None
            ),
            map_config_path=str(
                data.get("map_config_path", "configs/maps/airvln_maps.json")
            ),
            scene_map=scene_map,
            observation_profile_path=_optional_text(
                data.get("observation_profile_path")
            ),
            execution_mode=execution_mode,
            headless=bool(data.get("headless", True)),
            save_artifacts=bool(data.get("save_artifacts", False)),
            generate_html_review=bool(
                data.get("generate_html_review", True)
            ),
            enable_action_planner_debug=bool(
                data.get("enable_action_planner_debug", False)
            ),
            env_path=str(data.get("env_path", ".env")),
            parser_mode=str(data.get("parser_mode", "model")),
            shutdown_policy=str(data.get("shutdown_policy", "land")),
            runtime_loop_guard_windows=int(
                data.get("runtime_loop_guard_windows", 40)
            ),
            initialization_min_settle_sec=_non_negative_float(
                data.get("initialization_min_settle_sec", 0.0),
                field_name="initialization_min_settle_sec",
            ),
            action_config_overrides=_float_dict(
                data.get("action_config_overrides")
            ),
            safety_overrides=_float_dict(data.get("safety_overrides")),
            upstream_reference_commit=str(
                data.get(
                    "upstream_reference_commit",
                    "local_airvln_reference",
                )
            ),
            dataset_version=str(
                data.get("dataset_version", "unspecified")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}


def _non_negative_float(value: Any, *, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"AerialVLN {field_name} must be finite and non-negative"
        )
    return parsed
