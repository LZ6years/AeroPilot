"""Tests for offline real-run evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from demos.real_engine_instruction_demo import instructions_from_task_suite
from tools.real_run_evaluator import evaluate_run, extract_action_sequence, load_task_suite, write_eval_outputs


MOJIBAKE_PATTERNS = ("鍚", "绫", "鐒", "鎮", "锟", "?0", "?m")


FORWARD_TURN_FORWARD = "向前飞行5米后向右转40度，最后再向前走2米"
FORWARD_LEFT_FORWARD = "向前飞行3米后向左转30度，然后继续向前飞行3米并悬停"
STRAFE_LEFT_FORWARD_RIGHT = "向左平移2米后向前飞行3米，再向右转30度并悬停"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events), encoding="utf-8")


def make_artifacts(run_dir: Path, count: int) -> list[tuple[Path, Path]]:
    paths = []
    for index in range(1, count + 1):
        rgb = run_dir / "artifacts" / "rgb" / f"frame_{index:06d}.png"
        depth = run_dir / "artifacts" / "depth" / f"frame_{index:06d}.npy"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        paths.append((rgb, depth))
    return paths


def observation_event(rgb: Path, depth: Path) -> dict:
    return {
        "event": "automatic_observation_updated",
        "payload": {
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": -2.0}},
            "frames": {
                "rgb": {"artifact_path": str(rgb), "width": 224, "height": 224},
                "depth": {"artifact_path": str(depth), "width": 256, "height": 256},
            },
        },
    }


def phase_window_event(phase: str, measurements: dict, facts: dict | None = None) -> dict:
    return {
        "event": "phase_window_recorded",
        "payload": {
            "phase": phase,
            "action_window": {
                "intent": f"advance {phase}",
                "scope": "current_phase_only",
                "requires_intermediate_observation": False,
                "expected_effect": f"{phase} should show measurable progress",
            },
            "phase_window_count": 1,
            "phase_execution_summary": {
                "phase": phase,
                "window_count": 1,
                "max_phase_window_count": 5,
                "phase_local_measurements": measurements,
                "cumulative_tool_facts": facts or {},
            },
        },
    }


def step_event(phase: str, tool: str, args: dict) -> dict:
    return {
        "event": "tool_call_completed",
        "payload": {
            "phase": phase,
            "tool": tool,
            "args": args,
            "result": {"ok": True},
        },
    }


def task_event(instruction: str) -> dict:
    return {"event": "task_parsed", "payload": {"instruction": instruction}}


def phase_applied_complete() -> dict:
    return {"event": "phase_transition_applied", "payload": {"decision": "complete_task"}}


def agent_message_event(message_type: str, content: str = "测试消息") -> dict:
    return {
        "event": "agent_message",
        "payload": {
            "source": "phase_judge" if message_type == "final" else "action_planner",
            "phase": "test_phase",
            "message": {
                "type": message_type,
                "content": content,
                "evidence_refs": [],
                "confidence": "medium",
                "requires_user_reply": False,
            },
        },
    }


def agent_message_pair() -> list[dict]:
    return [
        agent_message_event("progress", "我正在执行当前阶段。"),
        agent_message_event("final", "任务完成，我已悬停。"),
    ]


def write_completed_run_boilerplate(run_dir: Path) -> None:
    write_json(run_dir / "latest_state.json", {"completed": True, "connected": True, "airborne": True})
    write_json(run_dir / "contexts" / "latest_context.json", {"phase_execution_summary": {}})


def test_evaluator_extracts_compound_task_metrics_and_artifacts(tmp_path):
    run_dir = tmp_path / "run_compound"
    artifacts = make_artifacts(run_dir, 3)
    events = [
        task_event(FORWARD_TURN_FORWARD),
        step_event("move_forward", "uav.move_body", {"direction": "forward", "duration": 2.5}),
        observation_event(*artifacts[0]),
        phase_window_event("move_forward", {"forward_displacement_m": 5.1}, {"move_body_duration_by_direction_success": {"forward": 2.5}}),
        step_event("rotate_right", "uav.turn_relative", {"yaw_deg": 40.0}),
        observation_event(*artifacts[1]),
        phase_window_event("rotate_right", {"yaw_delta_deg": 40.0}, {"turn_relative_yaw_deg_success_sum": 40.0}),
        step_event("move_forward_again", "uav.move_body", {"direction": "forward", "duration": 1.0}),
        observation_event(*artifacts[2]),
        phase_window_event("move_forward_again", {"forward_displacement_m": 2.0}),
        *agent_message_pair(),
        phase_applied_complete(),
    ]
    write_events(run_dir / "events.jsonl", events)
    write_completed_run_boilerplate(run_dir)

    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))
    result = evaluate_run(run_dir, suite)
    metrics = result["metrics"]

    assert metrics["task"]["id"] == "single_forward_5m_turn_right_40_forward_2m"
    assert metrics["overall"]["passed"] is True
    assert metrics["actions"]["success_rate"] == 1.0
    assert metrics["action_windows"]["total"] == 3
    assert metrics["action_windows"]["passed"] is True
    assert [item["actual"] for item in metrics["motion"]["metrics"]] == [5.1, 40.0, 2.0]
    assert metrics["artifacts"]["counts"] == {"rgb": 3, "depth": 3, "other": 0}


def test_evaluator_uses_phase_aliases_and_sequential_fallback(tmp_path):
    run_dir = tmp_path / "alias_run"
    artifacts = make_artifacts(run_dir, 3)
    write_events(
        run_dir / "events.jsonl",
        [
            task_event(STRAFE_LEFT_FORWARD_RIGHT),
            step_event("strafe_left", "uav.move_body", {"direction": "left", "duration": 1.0}),
            observation_event(*artifacts[0]),
            phase_window_event("strafe_left", {"right_displacement_m": -2.1}),
            step_event("move_forward_then_hover", "uav.move_body", {"direction": "forward", "duration": 1.5}),
            observation_event(*artifacts[1]),
            phase_window_event("move_forward_then_hover", {"forward_displacement_m": 3.0}),
            step_event("rotate_right_and_stabilize", "uav.turn_relative", {"yaw_deg": 30.0}),
            observation_event(*artifacts[2]),
            phase_window_event("rotate_right_and_stabilize", {"yaw_delta_deg": 30.0}, {"turn_relative_yaw_deg_success_sum": 30.0}),
            *agent_message_pair(),
            phase_applied_complete(),
        ],
    )
    write_completed_run_boilerplate(run_dir)

    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))
    result = evaluate_run(run_dir, suite, task_id="single_strafe_left_2m_forward_3m_turn_right_30")

    assert result["metrics"]["phases"]["passed"] is True
    assert result["metrics"]["motion"]["passed"] is True
    assert [item["match_status"] for item in result["metrics"]["motion"]["metrics"]] == [
        "phase_name_or_alias",
        "phase_name_or_alias",
        "phase_name_or_alias",
    ]


def test_standard_task_suite_has_readable_chinese_instructions():
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))
    instructions = []
    for task in suite["single_instruction_tasks"]:
        instructions.append(task["instruction"])
    for task in suite["continuous_instruction_tasks"]:
        instructions.extend(task["instructions"])
    joined = "\n".join(instructions)

    assert FORWARD_TURN_FORWARD in joined
    assert not any(pattern in joined for pattern in MOJIBAKE_PATTERNS)


def test_evaluator_handles_missing_run_files_without_crashing(tmp_path):
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))

    result = evaluate_run(run_dir, suite)

    assert result["metrics"]["overall"]["passed"] is False
    assert "basic" in result["metrics"]["overall"]["failed_sections"]
    assert result["metrics"]["actions"]["total"] == 0


def test_evaluator_writes_metrics_report_and_csv(tmp_path):
    run_dir = tmp_path / "minimal_compound_run"
    artifacts = make_artifacts(run_dir, 3)
    write_events(
        run_dir / "events.jsonl",
        [
            task_event(FORWARD_LEFT_FORWARD),
            step_event("move_forward", "uav.move_body", {"direction": "forward", "duration": 1.5}),
            observation_event(*artifacts[0]),
            phase_window_event("move_forward", {"forward_displacement_m": 3.0}),
            step_event("rotate_left", "uav.turn_relative", {"yaw_deg": -30.0}),
            observation_event(*artifacts[1]),
            phase_window_event("rotate_left", {"yaw_delta_deg": -30.0}, {"turn_relative_yaw_deg_success_sum": -30.0}),
            step_event("move_forward_again", "uav.move_body", {"direction": "forward", "duration": 1.5}),
            observation_event(*artifacts[2]),
            phase_window_event("move_forward_again", {"forward_displacement_m": 3.0}),
            *agent_message_pair(),
            phase_applied_complete(),
        ],
    )
    write_completed_run_boilerplate(run_dir)
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))
    result = evaluate_run(run_dir, suite, task_id="single_forward_3m_turn_left_30_forward_3m")

    write_eval_outputs(result)

    assert (run_dir / "eval" / "metrics.json").exists()
    assert (run_dir / "eval" / "report.md").exists()
    assert (run_dir / "eval" / "summary.csv").exists()


def test_continuous_evaluator_reports_each_instruction(tmp_path):
    run_dir = tmp_path / "continuous_run"
    artifacts = make_artifacts(run_dir, 2)
    write_events(
        run_dir / "events.jsonl",
        [
            task_event(FORWARD_TURN_FORWARD),
            step_event("move_forward", "uav.move_body", {"direction": "forward", "duration": 2.5}),
            observation_event(*artifacts[0]),
            phase_window_event("move_forward", {"forward_displacement_m": 5.0}),
            *agent_message_pair(),
            phase_applied_complete(),
            task_event(FORWARD_LEFT_FORWARD),
            step_event("rotate_left", "uav.turn_relative", {"yaw_deg": -30.0}),
            observation_event(*artifacts[1]),
            phase_window_event("rotate_left", {"yaw_delta_deg": -30.0}),
            *agent_message_pair(),
            phase_applied_complete(),
        ],
    )
    write_completed_run_boilerplate(run_dir)
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))

    result = evaluate_run(run_dir, suite, task_id="continuous_compound_navigation_set")
    instruction_results = result["metrics"]["continuous"]["instruction_results"]

    assert len(instruction_results) == 2
    assert all(item["passed"] for item in instruction_results)
    assert instruction_results[0]["action_total"] == 1
    assert instruction_results[1]["observed_phases"] == ["rotate_left"]


def test_batch_single_run_is_evaluated_per_instruction(tmp_path):
    run_dir = tmp_path / "batch_single_run"
    artifacts = make_artifacts(run_dir, 2)
    write_json(
        run_dir / "demo" / "summary.json",
        {"instructions": [{"instruction": FORWARD_TURN_FORWARD, "completed": True}, {"instruction": FORWARD_LEFT_FORWARD, "completed": True}]},
    )
    write_events(
        run_dir / "events.jsonl",
        [
            task_event(FORWARD_TURN_FORWARD),
            step_event("move_forward", "uav.move_body", {"direction": "forward", "duration": 2.5}),
            observation_event(*artifacts[0]),
            phase_window_event("move_forward", {"forward_displacement_m": 5.0}),
            *agent_message_pair(),
            phase_applied_complete(),
            task_event(FORWARD_LEFT_FORWARD),
            step_event("move_forward", "uav.move_body", {"direction": "forward", "duration": 1.5}),
            observation_event(*artifacts[1]),
            phase_window_event("move_forward", {"forward_displacement_m": 3.0}),
            *agent_message_pair(),
            phase_applied_complete(),
        ],
    )
    write_completed_run_boilerplate(run_dir)
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))

    result = evaluate_run(run_dir, suite)

    assert result["metrics"]["task"]["id"] == "batch_single"
    assert result["metrics"]["batch"]["is_batch"] is True
    assert result["metrics"]["batch"]["instruction_count"] == 2
    assert len(result["metrics"]["batch"]["instruction_results"]) == 2


def test_evaluator_flags_duplicate_artifact_paths(tmp_path):
    run_dir = tmp_path / "duplicate_artifacts"
    artifacts = make_artifacts(run_dir, 1)
    write_events(
        run_dir / "events.jsonl",
        [
            task_event(FORWARD_TURN_FORWARD),
            observation_event(*artifacts[0]),
            observation_event(*artifacts[0]),
            *agent_message_pair(),
            phase_applied_complete(),
        ],
    )
    write_completed_run_boilerplate(run_dir)
    suite = load_task_suite(Path("configs/eval/standard_tasks.json"))

    result = evaluate_run(run_dir, suite, task_id="single_forward_5m_turn_right_40_forward_2m")

    assert result["metrics"]["artifacts"]["passed"] is False
    assert len(result["metrics"]["artifacts"]["duplicate_paths"]) == 2


def test_task_suite_instruction_loader_supports_single_and_continuous(tmp_path):
    suite = {
        "single_instruction_tasks": [{"id": "one", "instruction": FORWARD_TURN_FORWARD}],
        "continuous_instruction_tasks": [
            {
                "id": "seq",
                "instructions": [FORWARD_TURN_FORWARD, FORWARD_LEFT_FORWARD],
            }
        ],
    }
    path = tmp_path / "suite.json"
    write_json(path, suite)

    assert instructions_from_task_suite(path, "single", "one") == [FORWARD_TURN_FORWARD]
    assert instructions_from_task_suite(path, "continuous", "seq") == [FORWARD_TURN_FORWARD, FORWARD_LEFT_FORWARD]


def test_extract_action_sequence_ignores_non_action_events():
    events = [
        {"event": "noise", "payload": {}},
        {"event": "tool_call_completed", "payload": {"phase": "p", "tool": "uav.hover", "args": {}, "result": {"ok": True}}},
    ]

    assert extract_action_sequence(events, {}) == [{"phase": "p", "tool": "uav.hover", "args": {}, "ok": True}]
