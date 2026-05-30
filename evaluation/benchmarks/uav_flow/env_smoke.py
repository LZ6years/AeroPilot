"""Low-cost UAV-Flow environment smoke checks that do not call planner models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config


def run_env_smoke(
    config: UavFlowEvalConfig,
    *,
    class_name: str | None = None,
    output_root: str | Path = "tests/manual_outputs/uav_flow_env_smoke",
    real_unrealcv: bool | None = None,
) -> dict[str, Any]:
    """Run one case through EnvClient observation/action without model calls."""

    classes = [class_name] if class_name else config.classes
    cases = load_cases(
        config.uav_flow_repo_path,
        classes=classes,
        limit_per_class=1,
        instruction_field=config.instruction_field,
        test_jsons_path=config.test_jsons_path,
        classified_instr_path=config.classified_instr_path,
    )
    if not cases:
        raise ValueError("No UAV-Flow case available for env smoke")
    case = cases[0]
    case_dir = Path(output_root) / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    action_config = _benchmark_action_config(config)
    client = UavFlowBenchmarkEnvClient(
        case,
        run_dir=case_dir,
        env_id=config.env_id,
        uav_flow_repo_path=config.uav_flow_repo_path,
        real_unrealcv=config.real_unrealcv if real_unrealcv is None else real_unrealcv,
        time_dilation=config.time_dilation,
        resolution=config.resolution,
        seed=config.seed,
        action_config=action_config,
    )
    scene = client.start_scene("uav_flow_downtownwest", case.case_id)
    before = client.get_observation(scene.scene_instance_id, "uav_flow_drone", ["rgb", "pose"]).to_dict()
    turn = client.execute_action(scene.scene_instance_id, "uav_flow_drone", "turn_relative", {"yaw_deg": 10}).to_dict()
    move = client.execute_action(
        scene.scene_instance_id,
        "uav_flow_drone",
        "move_body",
        {"direction": "forward", "duration": 0.25},
    ).to_dict()
    after = client.get_observation(scene.scene_instance_id, "uav_flow_drone", ["rgb", "pose"]).to_dict()
    client.stop_scene(scene.scene_instance_id)
    summary = {
        "case_id": case.case_id,
        "class_name": case.class_name,
        "instruction": case.selected_instruction,
        "real_unrealcv": client.real_unrealcv,
        "scene": scene.to_dict(),
        "before_observation": before,
        "turn_result": turn,
        "move_result": move,
        "after_observation": after,
        "pose_history": client.pose_history,
        "action_config": action_config.to_dict(),
    }
    (case_dir / "env_smoke.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "console.txt").write_text(_console(summary), encoding="utf-8")
    return {"case_dir": str(case_dir), "summary": summary}


def _console(summary: dict[str, Any]) -> str:
    before_pose = summary.get("before_observation", {}).get("data", {}).get("pose", {})
    after_pose = summary.get("after_observation", {}).get("data", {}).get("pose", {})
    return "\n".join(
        [
            "uav_flow_env_smoke: ok",
            f"case_id={summary.get('case_id')}",
            f"class={summary.get('class_name')}",
            f"real_unrealcv={summary.get('real_unrealcv')}",
            f"before_pose={before_pose}",
            f"after_pose={after_pose}",
            f"turn_ok={summary.get('turn_result', {}).get('ok')}",
            f"move_ok={summary.get('move_result', {}).get('ok')}",
        ]
    ) + "\n"


def _benchmark_action_config(config: UavFlowEvalConfig) -> UavActionConfig:
    data = load_uav_action_config().to_dict()
    data.update(config.action_config_overrides)
    return UavActionConfig.from_dict(data)
