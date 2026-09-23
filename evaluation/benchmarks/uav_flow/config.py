"""Configuration for UAV-Flow benchmark evaluation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_UAV_FLOW_REPO = ".external_refs/UAV-Flow/UAV-Flow-Eval"
DEFAULT_OUTPUT_DIR = "evaluation/benchmarks/uav_flow/outputs"
DEFAULT_MAP_CONFIG = "configs/maps/unrealcv_actor_control.example.json"
DEFAULT_MAP_ID = "downtownwest_local_unrealcv"
LEGACY_SCENE_FIELDS = {
    "env_id",
    "real_unrealcv",
    "reuse_real_unrealcv_env",
    "time_dilation",
    "resolution",
    "seed",
}


@dataclass(frozen=True)
class UavFlowEvalConfig:
    """Benchmark runner configuration.

    The config is separate from core runtime configs because UAV-Flow is an
    external benchmark harness, not a default runtime environment.
    """

    uav_flow_repo_path: str = DEFAULT_UAV_FLOW_REPO
    output_dir: str = DEFAULT_OUTPUT_DIR
    run_root: str = "runs"
    map_config_path: str = DEFAULT_MAP_CONFIG
    map_id: str = DEFAULT_MAP_ID
    execution_mode: str = field(kw_only=True)
    observation_profile_path: str | None = None
    headless: bool = True
    save_artifacts: bool = False
    generate_html_review: bool = True
    instruction_field: str = "instruction_unified"
    case_ids: tuple[str, ...] | None = None
    classes: list[str] | None = None
    limit_per_class: int | None = 1
    enable_sr_judge: bool = False
    env_path: str = ".env"
    shutdown_policy: str = "hover"
    action_config_overrides: dict[str, float] = field(default_factory=dict)
    safety_overrides: dict[str, float] = field(default_factory=dict)
    runtime_loop_guard_windows: int = 300
    enable_visual_diagnostics: bool = False
    visual_diagnostics_max_frames: int = 12
    enable_action_planner_debug: bool = False
    planner_depth_input: str = "off"
    stmr_mode: str = "off"
    stmr_prior: str = "default_uav_prior"
    stmr_max_queries: int = 16
    stmr_device: str = "auto"
    max_cases: int | None = None
    test_jsons_path_override: str | None = None
    classified_instr_path_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"attach", "managed"}
        if self.execution_mode not in allowed:
            raise ValueError("execution_mode must be explicitly set to " + "/".join(sorted(allowed)) + "; dry execution was removed; use offline case loading/report utilities for offline checks")

    @classmethod
    def from_file(cls, path: str | Path) -> "UavFlowEvalConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UavFlowEvalConfig":
        if data.get("execution_mode") == "dry":
            raise ValueError("dry execution was removed; use offline case loading/report utilities for offline checks or explicitly configure a real execution_mode")
        if "parser_mode" in data:
            raise ValueError("parser_mode was removed: submit raw instructions to the unified tool loop")
        legacy = sorted(LEGACY_SCENE_FIELDS.intersection(data))
        if legacy:
            raise ValueError(
                "Legacy UAV-Flow gym scene fields are no longer supported: "
                + ", ".join(legacy)
                + "; use map_config_path, map_id, execution_mode, "
                "observation_profile_path, headless, and save_artifacts"
            )
        case_ids = _case_ids(data.get("case_ids"))
        classes = data.get("classes")
        if case_ids is not None:
            # Exact selection is intentionally unambiguous: class and count
            # selectors cannot silently truncate or reorder requested cases.
            classes = None
        execution_mode = _choice(
            data.get("execution_mode", ""),
            {"attach", "managed"},
            "execution_mode",
        )
        return cls(
            uav_flow_repo_path=str(data.get("uav_flow_repo_path", cls.uav_flow_repo_path)),
            output_dir=str(data.get("output_dir", cls.output_dir)),
            run_root=str(data.get("run_root", cls.run_root)),
            map_config_path=str(
                data.get("map_config_path", cls.map_config_path)
            ),
            map_id=str(data.get("map_id", cls.map_id)),
            execution_mode=execution_mode,
            observation_profile_path=_optional_str(
                data.get("observation_profile_path")
            ),
            headless=bool(data.get("headless", True)),
            save_artifacts=bool(data.get("save_artifacts", False)),
            generate_html_review=bool(
                data.get("generate_html_review", True)
            ),
            instruction_field=str(data.get("instruction_field", cls.instruction_field)),
            case_ids=case_ids,
            classes=[str(item) for item in classes] if isinstance(classes, list) else None,
            limit_per_class=(
                None
                if case_ids is not None
                else _optional_int(
                    data.get("limit_per_class", cls.limit_per_class)
                )
            ),
            enable_sr_judge=bool(data.get("enable_sr_judge", False)),
            env_path=str(data.get("env_path", cls.env_path)),
            shutdown_policy=str(data.get("shutdown_policy", cls.shutdown_policy)),
            action_config_overrides=_float_dict(data.get("action_config_overrides", {})),
            safety_overrides=_float_dict(data.get("safety_overrides", data.get("safety_config_overrides", {}))),
            runtime_loop_guard_windows=int(data.get("runtime_loop_guard_windows", cls.runtime_loop_guard_windows)),
            enable_visual_diagnostics=bool(data.get("enable_visual_diagnostics", False)),
            visual_diagnostics_max_frames=int(data.get("visual_diagnostics_max_frames", cls.visual_diagnostics_max_frames)),
            enable_action_planner_debug=bool(data.get("enable_action_planner_debug", False)),
            planner_depth_input=_choice(
                data.get("planner_depth_input", "off"),
                {"off", "now"},
                "planner_depth_input",
            ),
            stmr_mode=_choice(data.get("stmr_mode", "off"), {"off", "shadow", "planner"}, "stmr_mode"),
            stmr_prior=str(data.get("stmr_prior", "default_uav_prior")),
            stmr_max_queries=max(1, int(data.get("stmr_max_queries", 16))),
            stmr_device=_choice(data.get("stmr_device", "auto"), {"auto", "cuda", "cpu"}, "stmr_device"),
            max_cases=(
                None
                if case_ids is not None
                else _optional_int(data.get("max_cases"))
            ),
            test_jsons_path_override=_optional_str(
                data.get("test_jsons_path", data.get("test_jsons_path_override"))
            ),
            classified_instr_path_override=_optional_str(
                data.get("classified_instr_path", data.get("classified_instr_path_override"))
            ),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def test_jsons_path(self) -> Path:
        if self.test_jsons_path_override:
            return Path(self.test_jsons_path_override)
        return Path(self.uav_flow_repo_path) / "test_jsons"

    @property
    def classified_instr_path(self) -> Path:
        if self.classified_instr_path_override:
            return Path(self.classified_instr_path_override)
        return Path(self.uav_flow_repo_path) / "classified_instr.json"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _case_ids(value: Any) -> tuple[str, ...] | None:
    """Validate exact dataset selectors before any simulator is started."""

    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("case_ids must be a list of dataset case stems")
    normalized = tuple(str(item).strip() for item in value)
    if not normalized or any(not item for item in normalized):
        raise ValueError("case_ids must contain at least one non-empty case stem")
    invalid = [
        item
        for item in normalized
        if item.endswith(".json") or "/" in item or "\\" in item
    ]
    if invalid:
        raise ValueError(
            "case_ids must use file-name stems without paths or .json: "
            + ", ".join(invalid)
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("case_ids must be unique")
    return normalized


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _choice(value: Any, allowed: set[str], field_name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {field_name}: {value!r}; expected one of: {choices}")
    return normalized
