"""Typed configuration for AerialVLN native-action evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DATASET_FAMILIES = {"aerialvln", "aerialvln-s"}
DIFFICULTIES = ("low", "medium", "high")
AERIAL_VLN_S_SPLITS = ("val_seen", "val_unseen")
AERIAL_VLN_SPLIT_LABELS = {
    "val_seen": "Validation Seen",
    "val_unseen": "Validation Unseen",
}


@dataclass(frozen=True)
class AerialVlnEvalConfig:
    dataset_root: str
    split: str = "val_seen"
    dataset_family: str = "aerialvln"
    expected_split_sha256: str | None = None
    difficulties: tuple[str, ...] = DIFFICULTIES
    limit_per_difficulty: int | None = None
    output_dir: str = "evaluation/benchmarks/aerial_vln/outputs"
    run_root: str = "runs"
    max_cases: int | None = None
    scene_filter: list[str] | None = None
    map_config_path: str = "configs/maps/airvln_maps.json"
    scene_map: dict[str, str] = field(
        default_factory=lambda: {"1": "airvln_env_1"}
    )
    observation_profile_path: str | None = None
    execution_mode: str = field(kw_only=True)
    abort_on_infrastructure_error: bool = True
    headless: bool = True
    save_artifacts: bool = False
    generate_html_review: bool = True
    enable_action_planner_debug: bool = False
    # STMR is benchmark opt-in. Shadow builds report-only spatial memory after
    # an episode; planner additionally exposes the compact map to the model.
    stmr_mode: str = "off"
    stmr_prior: str = "default_uav_prior"
    stmr_max_queries: int = 16
    stmr_device: str = "auto"
    env_path: str = ".env"
    shutdown_policy: str = "land"
    # Long multi-landmark routes need more observation/action windows than the
    # shared Runtime default. This remains benchmark-runner configuration and
    # does not change TaskRuntime or UAV-Flow defaults.
    runtime_loop_guard_windows: int = 300
    initialization_min_settle_sec: float = 0.0
    action_config_overrides: dict[str, float] = field(default_factory=dict)
    safety_overrides: dict[str, float] = field(default_factory=dict)
    upstream_reference_commit: str = "local_airvln_reference"
    dataset_version: str = "unspecified"

    def __post_init__(self) -> None:
        allowed = {"attach", "managed"}
        if self.execution_mode not in allowed:
            raise ValueError("execution_mode must be explicitly set to " + "/".join(sorted(allowed)) + "; dry execution was removed; use --validate-only for offline checks")

    @classmethod
    def from_file(cls, path: str | Path) -> "AerialVlnEvalConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("AerialVLN config root must be an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AerialVlnEvalConfig":
        if data.get("execution_mode") == "dry":
            raise ValueError("dry execution was removed; use --validate-only for offline checks or explicitly configure a real execution_mode")
        if "parser_mode" in data:
            raise ValueError("parser_mode was removed: submit raw instructions to the unified tool loop")
        dataset_root = str(data.get("dataset_root", "")).strip()
        if not dataset_root:
            raise ValueError("AerialVLN config requires dataset_root")
        dataset_family = str(
            data.get("dataset_family", "aerialvln")
        ).strip().lower()
        if dataset_family not in DATASET_FAMILIES:
            raise ValueError(
                "AerialVLN dataset_family must be aerialvln or aerialvln-s"
            )
        split = str(data.get("split", "val_seen")).strip()
        if not split:
            raise ValueError("AerialVLN split cannot be empty")
        execution_mode = str(data.get("execution_mode", "")).lower()
        if execution_mode not in {"attach", "managed"}:
            raise ValueError(
                "AerialVLN execution_mode must be explicitly set to attach or managed"
            )
        raw_scene_map = data.get("scene_map", {"1": "airvln_env_1"})
        if not isinstance(raw_scene_map, dict):
            raise ValueError("AerialVLN scene_map must be an object")
        scene_map = {
            str(scene_id): str(map_id)
            for scene_id, map_id in raw_scene_map.items()
            if str(scene_id).strip() and str(map_id).strip()
        }
        if not scene_map and dataset_family != "aerialvln-s":
            raise ValueError("AerialVLN scene_map cannot be empty")
        scene_filter = data.get("scene_filter")
        parsed_scene_filter = (
            [str(item) for item in scene_filter]
            if isinstance(scene_filter, list)
            else None
        )
        max_cases = _optional_int(data.get("max_cases"))
        if dataset_family == "aerialvln-s":
            if split not in AERIAL_VLN_S_SPLITS:
                raise ValueError(
                    "AerialVLN-S split must be val_seen or val_unseen"
                )
            if max_cases is not None:
                raise ValueError(
                    "AerialVLN-S does not support max_cases; use "
                    "limit_per_difficulty"
                )
            if parsed_scene_filter is not None:
                raise ValueError(
                    "AerialVLN-S classifies the complete selected split and "
                    "does not support scene_filter"
                )
        return cls(
            dataset_root=dataset_root,
            split=split,
            dataset_family=dataset_family,
            expected_split_sha256=_optional_sha256(
                data.get("expected_split_sha256")
            ),
            difficulties=_difficulties(data.get("difficulties")),
            limit_per_difficulty=_positive_optional_int(
                data.get("limit_per_difficulty"),
                field_name="limit_per_difficulty",
            ),
            output_dir=str(
                data.get(
                    "output_dir",
                    "evaluation/benchmarks/aerial_vln/outputs",
                )
            ),
            run_root=str(data.get("run_root", "runs")),
            max_cases=max_cases,
            scene_filter=parsed_scene_filter,
            map_config_path=str(
                data.get("map_config_path", "configs/maps/airvln_maps.json")
            ),
            scene_map=scene_map,
            observation_profile_path=_optional_text(
                data.get("observation_profile_path")
            ),
            execution_mode=execution_mode,
            abort_on_infrastructure_error=bool(
                data.get("abort_on_infrastructure_error", True)
            ),
            headless=bool(data.get("headless", True)),
            save_artifacts=bool(data.get("save_artifacts", False)),
            generate_html_review=bool(
                data.get("generate_html_review", True)
            ),
            enable_action_planner_debug=bool(
                data.get("enable_action_planner_debug", False)
            ),
            stmr_mode=_choice(
                data.get("stmr_mode", "off"),
                {"off", "shadow", "planner"},
                field_name="stmr_mode",
            ),
            stmr_prior=str(
                data.get("stmr_prior", "default_uav_prior")
            ),
            stmr_max_queries=_positive_int(
                data.get("stmr_max_queries", 16),
                field_name="stmr_max_queries",
            ),
            stmr_device=_choice(
                data.get("stmr_device", "auto"),
                {"auto", "cuda", "cpu"},
                field_name="stmr_device",
            ),
            env_path=str(data.get("env_path", ".env")),
            shutdown_policy=str(data.get("shutdown_policy", "land")),
            runtime_loop_guard_windows=int(
                data.get("runtime_loop_guard_windows", 300)
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


def aerial_vln_split_label(split: str) -> str:
    """Return a report label without restricting legacy dataset splits."""

    normalized = str(split).strip()
    return AERIAL_VLN_SPLIT_LABELS.get(normalized, normalized)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_sha256(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "AerialVLN expected_split_sha256 must contain 64 hex characters"
        )
    return normalized


def _difficulties(value: Any) -> tuple[str, ...]:
    if value is None:
        return DIFFICULTIES
    if not isinstance(value, (list, tuple)):
        raise ValueError("AerialVLN difficulties must be a list")
    parsed = tuple(str(item).strip().lower() for item in value)
    if not parsed:
        raise ValueError("AerialVLN difficulties cannot be empty")
    unknown = sorted(set(parsed).difference(DIFFICULTIES))
    if unknown:
        raise ValueError(
            "Unknown AerialVLN difficulties: " + ", ".join(unknown)
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError("AerialVLN difficulties must be unique")
    return parsed


def _positive_optional_int(
    value: Any, *, field_name: str
) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(
            f"AerialVLN {field_name} must be positive or null"
        )
    return parsed


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


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"AerialVLN {field_name} must be positive")
    return parsed


def _choice(
    value: Any, choices: set[str], *, field_name: str
) -> str:
    parsed = str(value).strip().lower()
    if parsed not in choices:
        raise ValueError(
            f"AerialVLN {field_name} must be one of: "
            + ", ".join(sorted(choices))
        )
    return parsed
