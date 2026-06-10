"""Configuration for UAV-Flow benchmark evaluation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_UAV_FLOW_REPO = r"C:\Users\HP\Desktop\UAV-Flow-main\UAV-Flow-main\UAV-Flow-Eval"
DEFAULT_ENV_ID = "UnrealTrack-DowntownWest-ContinuousColor-v0"
DEFAULT_OUTPUT_DIR = "evaluation/benchmarks/uav_flow/outputs"


@dataclass(frozen=True)
class UavFlowEvalConfig:
    """Benchmark runner configuration.

    The config is separate from core runtime configs because UAV-Flow is an
    external benchmark harness, not a default runtime environment.
    """

    uav_flow_repo_path: str = DEFAULT_UAV_FLOW_REPO
    env_id: str = DEFAULT_ENV_ID
    output_dir: str = DEFAULT_OUTPUT_DIR
    run_root: str = "runs"
    instruction_field: str = "instruction_unified"
    classes: list[str] | None = None
    limit_per_class: int | None = 1
    enable_sr_judge: bool = False
    real_unrealcv: bool = False
    reuse_real_unrealcv_env: bool = True
    time_dilation: int = 10
    resolution: list[int] = field(default_factory=lambda: [256, 256])
    seed: int = 0
    env_path: str = ".env"
    parser_mode: str = "model"
    shutdown_policy: str = "hover"
    action_config_overrides: dict[str, float] = field(default_factory=dict)
    safety_overrides: dict[str, float] = field(default_factory=dict)
    runtime_loop_guard_windows: int = 100
    enable_visual_diagnostics: bool = False
    visual_diagnostics_max_frames: int = 12
    enable_action_planner_debug: bool = False
    max_cases: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "UavFlowEvalConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UavFlowEvalConfig":
        classes = data.get("classes")
        return cls(
            uav_flow_repo_path=str(data.get("uav_flow_repo_path", cls.uav_flow_repo_path)),
            env_id=str(data.get("env_id", cls.env_id)),
            output_dir=str(data.get("output_dir", cls.output_dir)),
            run_root=str(data.get("run_root", cls.run_root)),
            instruction_field=str(data.get("instruction_field", cls.instruction_field)),
            classes=[str(item) for item in classes] if isinstance(classes, list) else None,
            limit_per_class=_optional_int(data.get("limit_per_class", cls.limit_per_class)),
            enable_sr_judge=bool(data.get("enable_sr_judge", False)),
            real_unrealcv=bool(data.get("real_unrealcv", False)),
            reuse_real_unrealcv_env=bool(data.get("reuse_real_unrealcv_env", True)),
            time_dilation=int(data.get("time_dilation", cls.time_dilation)),
            resolution=_resolution(data.get("resolution", [256, 256])),
            seed=int(data.get("seed", cls.seed)),
            env_path=str(data.get("env_path", cls.env_path)),
            parser_mode=str(data.get("parser_mode", cls.parser_mode)),
            shutdown_policy=str(data.get("shutdown_policy", cls.shutdown_policy)),
            action_config_overrides=_float_dict(data.get("action_config_overrides", {})),
            safety_overrides=_float_dict(data.get("safety_overrides", data.get("safety_config_overrides", {}))),
            runtime_loop_guard_windows=int(data.get("runtime_loop_guard_windows", cls.runtime_loop_guard_windows)),
            enable_visual_diagnostics=bool(data.get("enable_visual_diagnostics", False)),
            visual_diagnostics_max_frames=int(data.get("visual_diagnostics_max_frames", cls.visual_diagnostics_max_frames)),
            enable_action_planner_debug=bool(data.get("enable_action_planner_debug", False)),
            max_cases=_optional_int(data.get("max_cases")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def test_jsons_path(self) -> Path:
        return Path(self.uav_flow_repo_path) / "test_jsons"

    @property
    def classified_instr_path(self) -> Path:
        return Path(self.uav_flow_repo_path) / "classified_instr.json"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolution(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        return [256, 256]
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return [256, 256]
    return [width, height] if width > 0 and height > 0 else [256, 256]


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
