"""Typed configuration for TravelUAV FlyAgent transfer evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evaluation.benchmarks.travel_uav.selection import (
    TRAVEL_UAV_DIFFICULTIES,
    TRAVEL_UAV_SPLITS,
)


@dataclass(frozen=True)
class TravelUavEvalConfig:
    """All benchmark-owned choices; simulator mechanics remain shared."""

    index_path: str
    split: str = "test_seen"
    expected_index_sha256: str | None = None
    difficulties: tuple[str, ...] = TRAVEL_UAV_DIFFICULTIES
    limit_per_difficulty: int | None = 3
    case_ids: tuple[str, ...] | None = None
    output_dir: str = "evaluation/benchmarks/travel_uav/outputs"
    run_root: str = "runs"
    map_config_path: str = "configs/maps/travel_uav_remote.example.json"
    scene_map: dict[str, str] = field(default_factory=dict)
    observation_profile_path: str | None = None
    execution_mode: str = field(kw_only=True)
    abort_on_infrastructure_error: bool = True
    headless: bool = True
    save_artifacts: bool = False
    generate_html_review: bool = True
    enable_action_planner_debug: bool = False
    env_path: str = ".env"
    context_config_path: str = "configs/context.json"
    shutdown_policy: str = "hover"
    runtime_loop_guard_windows: int = 300
    initialization_min_settle_sec: float = 0.0
    action_config_overrides: dict[str, float] = field(default_factory=dict)
    safety_overrides: dict[str, float] = field(default_factory=dict)
    dataset_version: str = "traveluav-official"
    upstream_reference_commit: str = "prince687028/TravelUAV"

    def __post_init__(self) -> None:
        allowed = {"managed"}
        if self.execution_mode not in allowed:
            raise ValueError("execution_mode must be explicitly set to " + "/".join(sorted(allowed)) + "; dry execution was removed; use --validate-only for offline checks")

    @classmethod
    def from_file(cls, path: str | Path) -> "TravelUavEvalConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("TravelUAV config root must be an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TravelUavEvalConfig":
        if data.get("execution_mode") == "dry":
            raise ValueError("dry execution was removed; use --validate-only for offline checks or explicitly configure a real execution_mode")
        if "parser_mode" in data:
            raise ValueError("parser_mode was removed: submit raw instructions to the unified tool loop")
        index_path = str(data.get("index_path", "")).strip()
        if not index_path:
            raise ValueError("TravelUAV config requires index_path")
        split = str(data.get("split", "test_seen")).strip().lower()
        if split not in TRAVEL_UAV_SPLITS:
            raise ValueError(
                "TravelUAV split must be test_seen, test_unseen_map, or "
                "test_unseen_object"
            )
        execution_mode = str(data.get("execution_mode", "")).strip().lower()
        if execution_mode not in {"managed"}:
            raise ValueError(
                "TravelUAV v1 execution_mode must be explicitly set to managed; attach "
                "cannot guarantee dynamic-target cleanup"
            )
        raw_scene_map = data.get("scene_map", {})
        if not isinstance(raw_scene_map, dict):
            raise ValueError("TravelUAV scene_map must be an object")
        scene_map = {
            str(key): str(value)
            for key, value in raw_scene_map.items()
            if str(key).strip() and str(value).strip()
        }
        raw_case_ids = data.get("case_ids")
        case_ids = None
        if raw_case_ids is not None:
            if not isinstance(raw_case_ids, (list, tuple)):
                raise ValueError("TravelUAV case_ids must be a list or null")
            case_ids = tuple(str(item).strip() for item in raw_case_ids)
            if any(not item for item in case_ids) or len(set(case_ids)) != len(case_ids):
                raise ValueError("TravelUAV case_ids must be non-empty and unique")
        raw_difficulties = data.get("difficulties", TRAVEL_UAV_DIFFICULTIES)
        if not isinstance(raw_difficulties, (list, tuple)):
            raise ValueError("TravelUAV difficulties must be a list")
        difficulties = tuple(str(item).strip().lower() for item in raw_difficulties)
        if (
            not difficulties
            or len(set(difficulties)) != len(difficulties)
            or set(difficulties).difference(TRAVEL_UAV_DIFFICULTIES)
        ):
            raise ValueError("TravelUAV difficulties must be unique easy/hard values")
        raw_limit = data.get("limit_per_difficulty", 3)
        limit = None if raw_limit is None else int(raw_limit)
        if limit is not None and limit <= 0:
            raise ValueError("TravelUAV limit_per_difficulty must be positive or null")
        return cls(
            index_path=index_path,
            split=split,
            expected_index_sha256=_optional_sha(data.get("expected_index_sha256")),
            difficulties=difficulties,
            limit_per_difficulty=limit,
            case_ids=case_ids,
            output_dir=str(data.get("output_dir", cls.output_dir)),
            run_root=str(data.get("run_root", "runs")),
            map_config_path=str(
                data.get("map_config_path", "configs/maps/travel_uav_remote.example.json")
            ),
            scene_map=scene_map,
            observation_profile_path=_optional_text(data.get("observation_profile_path")),
            execution_mode=execution_mode,
            abort_on_infrastructure_error=bool(data.get("abort_on_infrastructure_error", True)),
            headless=bool(data.get("headless", True)),
            save_artifacts=bool(data.get("save_artifacts", False)),
            generate_html_review=bool(data.get("generate_html_review", True)),
            enable_action_planner_debug=bool(data.get("enable_action_planner_debug", False)),
            env_path=str(data.get("env_path", ".env")),
            context_config_path=str(data.get("context_config_path", "configs/context.json")),
            shutdown_policy=str(data.get("shutdown_policy", "hover")),
            runtime_loop_guard_windows=_positive_int(
                data.get("runtime_loop_guard_windows", 300), "runtime_loop_guard_windows"
            ),
            initialization_min_settle_sec=_non_negative_float(
                data.get("initialization_min_settle_sec", 0.0),
                "initialization_min_settle_sec",
            ),
            action_config_overrides=_float_dict(data.get("action_config_overrides")),
            safety_overrides=_float_dict(data.get("safety_overrides")),
            dataset_version=str(data.get("dataset_version", "traveluav-official")),
            upstream_reference_commit=str(
                data.get("upstream_reference_commit", "prince687028/TravelUAV")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_sha(value: Any) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = text.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("TravelUAV expected_index_sha256 must contain 64 hex characters")
    return normalized


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"TravelUAV {label} must be positive")
    return parsed


def _non_negative_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"TravelUAV {label} must be finite and non-negative")
    return parsed


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): float(item) for key, item in value.items()}
