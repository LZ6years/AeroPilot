"""Contract tests for simulator-neutral benchmark execution."""

from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.benchmarks.core import (
    BenchmarkAuditLog,
    BenchmarkAuditReader,
    BenchmarkCaseReview,
    BenchmarkEpisodeExecutor,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    EpisodeAuditCursor,
    FrameAssetResolver,
    EpisodeTraceReader,
    PreparedBenchmarkEpisode,
    ResolvedFrameAsset,
    ReviewSection,
    TraceCursor,
    write_audit_json,
)
from src.env_client import ObservationProfile, SceneCapabilities, SceneInstance
from src.runtime.configs import TaskExecutionOptions
from src.runtime.context.checkpoints import UnsupportedRunSchema


def test_readable_provider_request_keeps_latest_strategy_and_escapes_text(tmp_path):
    from evaluation.benchmarks.core.review import _provider_round
    old = {"current_phase": {"name": "old"}, "phase_strategy": {"stages": ["OLD"], "active_stage": 0}}
    latest = {"current_phase": {"name": "new"}}
    messages = [{"role": "user", "content": [{"type": "text", "text": json.dumps(c)}]} for c in (old, latest)]
    messages.append({"role": "assistant", "content": "<script>evil</script>", "tool_calls": [{"id": "x", "function": {"name": "inspect", "arguments": '{"query":"<img>"}'}}]})
    original = json.dumps(messages)
    rendered = _provider_round({"request": {"messages": messages}, "canonical_turn": {"tool_calls": [{"tool_name": "runtime.commit_phase_turn", "args": {"phase_strategy": {"stages": ["PROPOSED"]}}}]}, "runtime_outcome": {"status": "arguments_rejected"}}, tmp_path, [])
    assert "No strategy in latest request checkpoint" in rendered
    assert "PROPOSED" in rendered
    assert "No recorded application" in rendered
    assert "Readable request messages" in rendered
    assert "&lt;script&gt;" in rendered and "<script>evil" not in rendered
    assert json.dumps(messages) == original


def test_provider_round_applied_strategy_requires_event_evidence(tmp_path):
    from evaluation.benchmarks.core.review import _provider_round
    rendered = _provider_round({"runtime_outcome": {"status": "commit_accepted"}}, tmp_path, [],
                               applied_strategy={"events": [{"update_kind": "advance"}], "snapshot": {"active_stage": 1}})
    assert "Recorded strategy update (not proof of successful motion)" in rendered
    assert "advance" in rendered


def test_state_checkpoints_repeat_safe_image_links_and_report_missing_assets(tmp_path):
    from evaluation.benchmarks.core.review import _agent_tool_transcript
    image_path = tmp_path / "review_assets" / "same.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"fixture")
    records = [{"kind": "state_checkpoint", "payload": {"observation_id": "obs1", "context": {"current_phase": {"name": "p"}}}} for _ in range(2)]
    audit = SimpleNamespace(agent_transcript=records)
    rendered = _agent_tool_transcript(audit, (), tmp_path, {"request": [
        {"observation_id": "obs1", "timeline_role": "TOOL_EVIDENCE", "path": str(tmp_path / "candidate.png")},
        {"observation_id": "obs1", "path": str(image_path)},
    ]})
    assert rendered.count('<img ') == 2
    assert rendered.count('src="review_assets/same.png"') == 2
    assert "Original RGB unavailable" in rendered
    missing = _agent_tool_transcript(audit, (), tmp_path, {})
    assert missing.count("Exact provider image unavailable") == 2


def test_rejection_card_distinguishes_affected_call_and_legacy_message(tmp_path):
    from evaluation.benchmarks.core.review import _agent_tool_transcript
    payload = {"tool_call_id": "commit", "tool_name": "runtime.commit_phase_turn", "status": "rejected", "message": "bad <yaw>", "data": {"rejection": {"role": "affected", "source_tool_call_id": "turn", "parameter_path": "args.yaw_deg"}}}
    audit = SimpleNamespace(agent_transcript=[{"kind": "tool_result", "payload": payload}])
    rendered = _agent_tool_transcript(audit, (), tmp_path, {})
    assert "Not executed because another call" in rendered
    assert "args.yaw_deg" in rendered and "bad &lt;yaw&gt;" in rendered
    payload["data"] = {}
    legacy = _agent_tool_transcript(audit, (), tmp_path, {})
    assert "Structured attribution unavailable" in legacy


def test_prepared_episode_requires_profile_options_alignment():
    with pytest.raises(ValueError, match="modalities must match"):
        make_episode(
            profile=ObservationProfile.from_dict(
                {"modalities": ["rgb", "pose"]}
            ),
            options=TaskExecutionOptions(),
        )


def test_unified_capture_reader_obeys_frozen_event_boundary(tmp_path):
    write_trace_fixture(tmp_path, window_event="physical_window_recorded")
    folder = tmp_path / "debug" / "tool_loop"
    folder.mkdir(parents=True)
    for index in (1, 2):
        decision = f"decision_{index:04d}"
        payload = {"loop_schema": "unified_tool_loop_v2", "decision_id": decision,
                   "decision_index": index, "checkpoint_observation_id": "obs_0001",
                   "model_context": {"task_plan": None}, "model_rounds": []}
        (folder / f"{decision}.json").write_text(json.dumps(payload), encoding="utf-8")
    def capture(index):
        decision = f"decision_{index:04d}"
        append_jsonl(tmp_path / "events.jsonl", {
            "timestamp": "2026-01-01T00:00:02+00:00", "event": "model_turn_recorded",
            "payload": {"decision_id": decision, "debug_ref": f"debug/tool_loop/{decision}.json"}})
    capture(1)
    cursor = EpisodeAuditCursor.freeze(tmp_path)
    capture(2)
    audit = BenchmarkAuditReader().read(cursor)
    assert [r["decision_id"] for r in audit.planner_debug] == ["decision_0001"]
    assert not audit.diagnostics


@pytest.mark.parametrize("ref", ["../outside.json", "missing.json"])
def test_unified_capture_reports_unsafe_or_missing_artifact(tmp_path, ref):
    from evaluation.benchmarks.core.audit import _tool_loop_debug_records
    records, diagnostics = _tool_loop_debug_records(tmp_path, [{"event": "model_turn_recorded",
        "payload": {"decision_id": "d1", "debug_ref": ref}}])
    assert records == [] and len(diagnostics) == 1


def test_unified_review_displays_checkpoint_plan_without_future_backfill(tmp_path):
    from evaluation.benchmarks.core.review import _phase_decision_review, _initial_plan_review
    first = {"loop_schema": "unified_tool_loop_v2", "decision_id": "d1", "decision_index": 1,
             "checkpoint_observation_id": "obs1", "model_context": {"task_plan": None},
             "model_rounds": []}
    second = {**first, "decision_id": "d2", "decision_index": 2,
              "model_context": {"task_plan": {"revision": 1, "steps": [
                  {"id": "a", "description": "Reach <gate>", "status": "in_progress",
                   "evidence_refs": []}]}}}
    audit = SimpleNamespace(planner_debug=[first, second], agent_transcript=[], events=[])
    rendered = _phase_decision_review(audit, (), tmp_path, {})
    first_card, second_card = rendered.split('id="decision-d2"')
    assert "No plan created" in first_card and "Reach" not in first_card
    assert "Reach &lt;gate&gt;" in second_card
    assert "Task execution timeline" in rendered
    assert "Strategy at checkpoint" not in rendered and "phase outcome" not in rendered
    assert "Optional task plan" in _initial_plan_review(audit)


def test_trace_reader_excludes_preflight_and_shutdown_and_keeps_stationary_action(
    tmp_path,
):
    write_trace_fixture(tmp_path)
    cursor = TraceCursor.freeze(tmp_path)
    append_jsonl(
        tmp_path / "memory" / "actions.jsonl",
        {
            "id": "act_0003",
            "timestamp": "2026-01-01T00:00:05+00:00",
            "tool": "uav.land",
            "result": {"ok": True},
        },
    )

    trace = EpisodeTraceReader().read(cursor)

    assert trace.action_count == 1
    assert [sample.source for sample in trace.samples] == [
        "task_initial",
        "action_result",
    ]
    assert trace.samples[1].tool == "uav.turn_relative"
    assert trace.samples[1].pose["x"] == 1.0
    assert trace.samples[1].action_args == {"yaw_deg": 15.0}
    assert trace.physical_windows[0].observation_after_pose["yaw_deg"] == (
        15.25
    )
    assert trace.trace_integrity == "ok"
    assert trace.physical_call_count == 1
    assert trace.physical_window_count == 1
    assert trace.motion_call_count == 1
    assert trace.require_single_coordinate_frame("unrealcv_world_m") == (
        "unrealcv_world_m"
    )


def test_trace_reader_falls_back_to_observation_pose(tmp_path):
    write_trace_fixture(tmp_path, action_pose=False)

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    assert trace.action_count == 1
    assert trace.samples[-1].source == "window_observation_fallback"
    assert trace.samples[-1].pose["yaw_deg"] == 15.0


def test_collision_trace_keeps_actual_pose_without_control_error(tmp_path):
    from evaluation.benchmarks.uav_flow.metric_suite import (
        _backend_action_tracking,
    )
    from src.uav.config import UavActionConfig

    write_trace_fixture(tmp_path, window_event="physical_window_recorded")
    events_path = tmp_path / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    window = events[-1]["payload"]
    window["status"] = "collision"
    window["ordered_steps"][0]["status"] = "collision"
    events_path.write_text(
        "\n".join(json.dumps(item) for item in events) + "\n",
        encoding="utf-8",
    )

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))
    tracking = _backend_action_tracking(
        trace,
        action_config=UavActionConfig(),
        completed=False,
    )

    assert trace.trace_integrity == "ok"
    assert trace.collision_call_count == trace.collision_window_count == 1
    assert trace.motion_call_count == 1
    assert trace.samples[-1].pose["x"] == 1.0
    assert tracking["collision_call_count"] == 1
    assert tracking["collision_window_count"] == 1
    assert tracking["failed_window_count"] == 0
    assert tracking["failed_or_skipped_step_count"] == 0
    row = tracking["windows"][0]
    assert row["comparison_status"] == "collision_not_scored"
    assert row["observed_delta_m"] == {"dx": 1.0, "dy": 0.0, "dz": 0.0}
    assert row["position_tracking_error_m"] is None
    assert row["yaw_tracking_error_deg"] is None


def test_review_uses_collision_label_instead_of_failed():
    from evaluation.benchmarks.core.review import _execution_cell

    rendered = _execution_cell(
        {
            "tracking": {"status": "collision"},
            "result": {
                "ok": False,
                "message": "Collision detected",
                "execution_state": {"outcome": "collision"},
            },
        }
    )

    assert "collision-label" in rendered
    assert "collision" in rendered
    assert "failed" not in rendered


@pytest.mark.parametrize("event_name", ["physical_window_recorded", "phase_window_recorded"])
def test_physical_window_event_names_share_explicit_trace_and_action_refs(tmp_path, event_name):
    """A phase-free window retains the same trajectory and frozen action join."""
    write_trace_fixture(tmp_path, include_read_only=True, window_event=event_name)
    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))
    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(tmp_path))
    assert trace.trace_integrity == "ok"
    assert trace.physical_window_count == 1
    assert trace.physical_call_count == trace.motion_call_count == 1
    assert trace.physical_windows[0].observation_before_id == "obs_0001"
    assert trace.physical_windows[0].observation_after_id == "obs_0002"
    assert [item["tool"] for item in audit.actions] == ["uav.turn_relative"]
    if event_name == "physical_window_recorded":
        assert trace.physical_windows[0].phase_key is None


def test_trace_reader_keeps_initial_pose_when_no_formal_actions(tmp_path):
    write_trace_fixture(tmp_path, include_action=False)

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    assert trace.action_count == 0
    assert len(trace.samples) == 1
    assert trace.samples[0].source == "task_initial"


def test_trace_reader_excludes_runtime_error_hover_after_last_observation(
    tmp_path,
):
    write_trace_fixture(tmp_path)
    append_jsonl(
        tmp_path / "memory" / "actions.jsonl",
        {
            "id": "act_error_hover",
            "timestamp": "2026-01-01T00:00:04+00:00",
            "tool": "uav.hover",
            "result": {"ok": True},
        },
    )

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    assert trace.action_count == 1
    assert [sample.tool for sample in trace.samples[1:]] == [
        "uav.turn_relative"
    ]


def test_trace_reader_never_promotes_read_only_action_to_trajectory(tmp_path):
    write_trace_fixture(tmp_path, include_read_only=True)

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    assert trace.trace_integrity == "ok"
    assert trace.action_count == 1
    assert trace.physical_call_count == 1
    assert [sample.tool for sample in trace.samples[1:]] == [
        "uav.turn_relative"
    ]
    assert trace.physical_windows[0].ordered_steps[0].action_ref == (
        "memory/actions.jsonl#3"
    )


def test_trace_reader_marks_legacy_timestamps_unsupported(tmp_path):
    write_trace_fixture(tmp_path)
    (tmp_path / "events.jsonl").unlink()

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    assert trace.trace_integrity == "unsupported_legacy_trace_schema"
    assert trace.action_count == 0
    assert len(trace.samples) == 1


def test_trace_reader_rejects_mixed_coordinate_frames(tmp_path):
    write_trace_fixture(tmp_path)
    poses_path = tmp_path / "memory" / "poses.jsonl"
    poses = [
        json.loads(line)
        for line in poses_path.read_text(encoding="utf-8").splitlines()
    ]
    poses[-1]["coordinate_frame"] = "airsim_ned_canonical_m"
    poses_path.write_text(
        "\n".join(json.dumps(item) for item in poses) + "\n",
        encoding="utf-8",
    )

    trace = EpisodeTraceReader().read(TraceCursor.freeze(tmp_path))

    with pytest.raises(ValueError, match="requires one explicit"):
        trace.require_single_coordinate_frame()


def test_executor_orders_prepare_runtime_freeze_shutdown_cleanup_then_metrics(
    tmp_path,
):
    calls: list[str] = []
    runtime = FakeRuntime(tmp_path, calls)
    scenario = FakeScenario(calls)
    metrics = FakeMetrics(calls)

    result = BenchmarkEpisodeExecutor().execute(
        make_episode(),
        runtime=runtime,
        scenario_controller=scenario,
        metric_suite=metrics,
        run_dir=tmp_path,
    )

    assert calls == [
        "ensure",
        "scenario_prepare",
        "run",
        "finish",
        "scenario_cleanup",
        "metrics",
    ]
    assert result.outcome.completed is True
    assert result.trace.action_count == 1
    assert result.metrics == {"sample_count": 2}
    manifest = json.loads(
        (tmp_path / "benchmark" / "episode_planner_safe.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["metadata"] == {"weather": "clear"}
    assert "evaluation_payload" not in manifest
    assert "scenario_payload" not in manifest


def test_executor_stops_scene_without_state_when_scenario_prepare_fails(tmp_path):
    calls: list[str] = []
    runtime = FakeRuntime(tmp_path, calls)
    scenario = FakeScenario(calls, fail_prepare=True)

    result = BenchmarkEpisodeExecutor().execute(
        make_episode(),
        runtime=runtime,
        scenario_controller=scenario,
        metric_suite=FakeMetrics(calls),
        run_dir=tmp_path,
    )

    assert calls == [
        "ensure",
        "scenario_prepare",
        "finish",
        "scenario_cleanup",
        "metrics",
    ]
    assert runtime.finished_state is None
    assert result.outcome.error == {
        "error_type": "RuntimeError",
        "error": "prepare failed",
    }


def test_metric_failure_happens_after_scene_cleanup(tmp_path):
    calls: list[str] = []
    runtime = FakeRuntime(tmp_path, calls)

    result = BenchmarkEpisodeExecutor().execute(
        make_episode(),
        runtime=runtime,
        scenario_controller=FakeScenario(calls),
        metric_suite=FakeMetrics(calls, fail=True),
        run_dir=tmp_path,
    )

    assert calls.index("scenario_cleanup") < calls.index("metrics")
    assert result.metric_error == {
        "error_type": "ValueError",
        "error": "metric failed",
    }


def test_audit_cursor_bounds_frames_actions_transcript_and_shutdown(tmp_path):
    write_audit_fixture(tmp_path)
    cursor = EpisodeAuditCursor.freeze(tmp_path)
    append_jsonl(
        tmp_path / "memory" / "agent_transcript.jsonl",
        {
            "id": "turn_after_freeze",
            "timestamp": "2026-01-01T00:00:04+00:00",
            "instruction_id": "inst_1",
            "kind": "protocol_feedback",
            "payload": {"message": "must not enter frozen audit"},
        },
    )
    append_jsonl(
        tmp_path / "memory" / "actions.jsonl",
        {
            "id": "act_shutdown",
            "timestamp": "2026-01-01T00:00:09+00:00",
            "tool": "uav.land",
            "result": {"ok": True},
        },
    )
    append_jsonl(
        tmp_path / "memory" / "frames.jsonl",
        {
            "id": "frame_shutdown",
            "timestamp": "2026-01-01T00:00:09+00:00",
            "observation_id": "obs_shutdown",
            "modality": "rgb",
            "metadata": {},
        },
    )

    audit = BenchmarkAuditReader().read(cursor)

    assert [item["tool"] for item in audit.actions] == [
        "uav.turn_relative"
    ]
    assert [item["id"] for item in audit.frames] == ["frame_0001"]
    assert [item["kind"] for item in audit.agent_transcript] == [
        "user_instruction",
        "state_checkpoint",
    ]
    assert audit.observation_anchors["task_initial"]["id"] == "obs_0001"


def test_frame_asset_resolver_supports_artifact_path_path_and_base64(
    tmp_path,
):
    run_dir = tmp_path / "run"
    case_dir = tmp_path / "case"
    png = _png_bytes()
    artifact = run_dir / "artifacts" / "rgb" / "artifact.png"
    top_path = run_dir / "artifacts" / "rgb" / "top.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(png)
    top_path.write_bytes(png)
    audit = BenchmarkAuditLog(
        run_dir=str(run_dir),
        started_at=None,
        ended_at="2026-01-01T00:00:00+00:00",
        task_initial_observation_ref=None,
        observations=(),
        frames=(
            {
                "id": "frame_artifact",
                "observation_id": "obs_1",
                "modality": "rgb",
                "metadata": {"artifact_path": str(artifact)},
            },
            {
                "id": "frame_path",
                "observation_id": "obs_2",
                "modality": "rgb",
                "path": str(top_path),
                "metadata": {},
            },
            {
                "id": "frame_base64",
                "observation_id": "obs_3",
                "modality": "rgb",
                "metadata": {
                    "base64": base64.b64encode(png).decode("ascii")
                },
            },
            {
                "id": "frame_broken",
                "observation_id": "obs_4",
                "modality": "rgb",
                "metadata": {"base64": "not-base64"},
            },
        ),
        poses=(),
        actions=(),
        events=(),
        debug_events=(),
        planner_debug=(),
    )

    assets = FrameAssetResolver().materialize(audit, case_dir)

    assert [item.source for item in assets[:3]] == [
        "artifact_path",
        "path",
        "base64",
    ]
    assert all(
        Path(item.display_path).is_file() for item in assets[:3]
    )
    assert assets[3].unavailable_reason == "invalid_base64"


def test_core_review_writes_portable_case_index_and_audit(tmp_path):
    audit = BenchmarkAuditLog(
        run_dir=str(tmp_path / "run"),
        started_at=None,
        ended_at="2026-01-01T00:00:00+00:00",
        task_initial_observation_ref=None,
        observations=(),
        frames=(),
        poses=(),
        actions=(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "tool": "uav.hover",
                "args": {},
                "result": {"ok": True},
            },
        ),
        events=(),
        debug_events=(),
        planner_debug=(),
    )
    case_dir = tmp_path / "eval" / "cases" / "case_1"
    image_path = case_dir / "review_assets" / "frames" / "frame.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_png_bytes())
    depth_path = case_dir / "review_assets" / "frames" / "depth.npy"
    depth_path.write_bytes(b"\x93NUMPY")
    frame_asset = ResolvedFrameAsset(
        frame_id="frame_1",
        observation_id="obs_1",
        modality="rgb",
        timestamp="2026-01-01T00:00:00+00:00",
        display_path=str(image_path),
        raw_path=str(image_path),
        source="artifact_path",
        unavailable_reason=None,
    )
    depth_asset = ResolvedFrameAsset(
        frame_id="frame_depth",
        observation_id="obs_1",
        modality="depth",
        timestamp="2026-01-01T00:00:00+00:00",
        display_path=None,
        raw_path=str(depth_path),
        source="base64",
        unavailable_reason="no_visual_payload",
    )
    audit_path = write_audit_json(
        case_dir / "audit.json", audit, (frame_asset, depth_asset)
    )
    summary = BenchmarkSummaryReview(
        benchmark_name="Test",
        protocol_variant="test.v1",
        metrics={"success": 1.0},
        cases=(
            {
                "case_id": "case_1",
                "status": "ok",
                "metrics": {"success": True},
            },
        ),
    )
    case_review = BenchmarkCaseReview(
        case_id="case_1",
        benchmark_name="Test",
        protocol_variant="test.v1",
        instruction="Hold",
        outcome={"status": "ok"},
        metrics={"success": True},
        audit=audit,
        frame_assets=(frame_asset, depth_asset),
        sections=(
            ReviewSection("Notes", "text", "portable review"),
        ),
    )
    paths = BenchmarkReviewRenderer(detailed_audit=True).write(
        tmp_path / "eval",
        summary,
        (case_review,),
    )

    assert Path(audit_path).exists()
    assert Path(paths["html_review"]).exists()
    case_html = (case_dir / "audit_review.html").read_text(encoding="utf-8")
    audit_text = Path(audit_path).read_text(encoding="utf-8")
    assert "../../review_assets/benchmark_review.css" in case_html
    assert "review_assets/frames/frame.png" in case_html
    assert "review_assets/frames/depth.npy" not in case_html
    assert "download raw asset" not in case_html
    assert "frame_depth" in audit_text
    assert "depth.npy" in audit_text
    assert '"agent_transcript": []' in audit_text
    assert "portable review" in case_html
    assert "uav.hover" in case_html
    assert "No agent tool transcript was recorded for this episode." in case_html
    assert "No audit diagnostics were recorded." in case_html

    BenchmarkReviewRenderer(detailed_audit=True).write(
        tmp_path / "eval",
        summary,
        (
            BenchmarkCaseReview(
                **{
                    **case_review.__dict__,
                    "gallery_modalities": ("rgb", "depth"),
                }
            ),
        ),
    )
    opted_in_html = (case_dir / "audit_review.html").read_text(encoding="utf-8")
    assert "review_assets/frames/depth.npy" in opted_in_html
    assert "download raw asset" in opted_in_html


def test_core_review_renders_agent_tool_transcript_causally(tmp_path):
    case_dir = tmp_path / "eval" / "cases" / "transcript"
    image_1 = case_dir / "review_assets" / "frames" / "obs_1.png"
    image_2 = case_dir / "review_assets" / "frames" / "obs_2.png"
    image_1.parent.mkdir(parents=True)
    image_1.write_bytes(_png_bytes())
    image_2.write_bytes(_png_bytes())
    assets = (
        ResolvedFrameAsset(
            frame_id="frame_1",
            observation_id="obs_1",
            modality="rgb",
            timestamp="2026-01-01T00:00:01+00:00",
            display_path=str(image_1),
            raw_path=str(image_1),
            source="artifact_path",
            unavailable_reason=None,
        ),
        ResolvedFrameAsset(
            frame_id="frame_2",
            observation_id="obs_2",
            modality="rgb",
            timestamp="2026-01-01T00:00:05+00:00",
            display_path=str(image_2),
            raw_path=str(image_2),
            source="artifact_path",
            unavailable_reason=None,
        ),
    )
    transcript = (
        _transcript_event(
            "user_instruction",
            {"instruction": "Find <person> and turn safely."},
            1,
        ),
        _transcript_event(
            "state_checkpoint",
            {
                "observation_id": "obs_1",
                "context": {
                    "current_phase": {"name": "inspect"},
                    "fragments": [],
                },
            },
            2,
        ),
        _transcript_event(
            "assistant_turn",
            {
                "turn_id": "assistant_detect",
                "public_intent": "Confirm <candidate> before moving.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_detect_bad",
                        "tool_name": "skill.person_detector.detect",
                        "args": {"observation_id": "stale"},
                    }
                ],
            },
            3,
        ),
            _transcript_event(
                "tool_result",
            {
                "tool_call_id": "call_detect_bad",
                "tool_name": "skill.person_detector.detect",
                "status": "rejected",
                "message": "stale observation",
                "requested_args": {"observation_id": "stale"},
                "executed_args": {},
                "safety_interventions": [],
                "artifact_refs": [],
                "data": {},
            },
            4,
        ),
        _transcript_event(
            "protocol_feedback",
            {"message": "Use the latest observation_id."},
            5,
        ),
        _transcript_event(
            "assistant_turn",
            {
                "turn_id": "assistant_detect_fixed",
                "public_intent": "Retry against the current frame.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_detect",
                        "tool_name": "skill.person_detector.detect",
                        "args": {"observation_id": "obs_1"},
                    }
                ],
            },
            6,
        ),
        _transcript_event(
            "tool_result",
            {
                "tool_call_id": "call_detect",
                "tool_name": "skill.person_detector.detect",
                "status": "ok",
                "message": "person detected",
                "requested_args": {"observation_id": "obs_1"},
                "executed_args": {"observation_id": "obs_1"},
                "safety_interventions": [],
                "artifact_refs": ["frame_1", "../../not-linked.txt"],
                "data": {"confidence": 0.91},
            },
                7,
            ),
            _transcript_event(
                "assistant_turn",
                {
                    "turn_id": "assistant_depth",
                    "public_intent": "Measure the current visible surface.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_depth",
                            "tool_name": "skill.metric_depth.sample_region",
                            "args": {
                                "observation_id": "obs_1",
                                "x_min": 0.45,
                                "y_min": 0.4,
                                "x_max": 0.55,
                                "y_max": 0.58,
                            },
                        }
                    ],
                },
                7,
            ),
            _transcript_event(
                "tool_result",
                {
                    "tool_call_id": "call_depth",
                    "tool_name": "skill.metric_depth.sample_region",
                    "status": "ok",
                    "message": "Approximate camera-to-visible-surface distance.",
                    "requested_args": {
                        "observation_id": "obs_1",
                        "x_min": 0.45,
                        "y_min": 0.4,
                        "x_max": 0.55,
                        "y_max": 0.58,
                    },
                    "executed_args": {
                        "observation_id": "obs_1",
                        "x_min": 0.45,
                        "y_min": 0.4,
                        "x_max": 0.55,
                        "y_max": 0.58,
                    },
                    "safety_interventions": [],
                    "artifact_refs": [],
                    "data": {
                        "observation_id": "obs_1",
                        "distance_m": 2.9,
                        "audit_details": {
                            "frame_id": "depth_1",
                            "roi_pixels": [115, 102, 141, 149],
                            "valid_pixel_count": 412,
                            "inlier_pixel_count": 396,
                            "raw_median_m": 2.86,
                            "mad_m": 0.07,
                        },
                    },
                },
                7,
            ),
            _transcript_event(
                "assistant_turn",
                {
                    "turn_id": "assistant_grounding",
                    "public_intent": "Locate the requested person and measure only this candidate.",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_grounding",
                            "tool_name": "skill.visual_grounding.measure_visible_depth",
                            "args": {
                                "observation_id": "obs_1",
                                "query": "the person to approach",
                                "include_distance": True,
                            },
                        }
                    ],
                },
                7,
            ),
            _transcript_event(
                "tool_result",
                {
                    "tool_call_id": "call_grounding",
                    "tool_name": "skill.visual_grounding.measure_visible_depth",
                    "status": "ok",
                    "message": "Located one candidate with robust metric depth.",
                    "requested_args": {
                        "observation_id": "obs_1",
                        "query": "the person to approach",
                        "include_distance": True,
                    },
                    "executed_args": {
                        "observation_id": "obs_1",
                        "query": "the person to approach",
                        "include_distance": True,
                    },
                    "safety_interventions": [],
                    "artifact_refs": ["frame_1"],
                    "data": {
                        "observation_id": "obs_1",
                        "query": "the person to approach",
                        "include_distance": True,
                        "candidates": [
                            {
                                "candidate_id": "candidate_01",
                                "label": "person",
                                "confidence": 0.91,
                                "box_normalized": [0.45, 0.3, 0.55, 0.75],
                                "distance_m": 2.9,
                                "distance_status": "ok",
                            }
                        ],
                        "audit_details": {
                            "grounding_backend": "grounding_dino",
                            "segmentation_backend": "sam",
                            "candidates": [
                                {
                                    "candidate_id": "candidate_01",
                                    "sampling_source": "eroded_mask",
                                    "mad_m": 0.06,
                                }
                            ],
                        },
                    },
                },
                7,
            ),
            _transcript_event(
                "assistant_turn",
            {
                "turn_id": "assistant_move",
                "public_intent": "Turn and advance in one action window.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_commit",
                        "tool_name": "runtime.commit_phase_turn",
                        "args": {"phase": "inspect"},
                    },
                    {
                        "tool_call_id": "call_turn",
                        "tool_name": "uav.turn_relative",
                        "args": {"yaw_deg": 25.0},
                    },
                    {
                        "tool_call_id": "call_move",
                        "tool_name": "uav.move_body",
                        "args": {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.0},
                    },
                ],
            },
            8,
        ),
        _tool_result_event("call_commit", "runtime.commit_phase_turn", 9),
        _transcript_event(
            "tool_result",
            {
                "tool_call_id": "call_turn",
                "tool_name": "uav.turn_relative",
                "status": "ok",
                "message": "turn complete",
                "requested_args": {"yaw_deg": 25.0},
                "executed_args": {"yaw_deg": 20.0},
                "safety_interventions": [
                    {"field": "yaw_deg", "requested": 25.0, "executed": 20.0}
                ],
                "artifact_refs": [],
                "data": {},
            },
            10,
        ),
        _tool_result_event("call_move", "uav.move_body", 11),
        _transcript_event(
            "state_checkpoint",
            {
                "observation_id": "obs_2",
                "context": {
                    "current_phase": {"name": "inspect"},
                    "completed": False,
                    "fragments": [
                        {
                            "name": "image",
                            "content": {
                                "selected_frames": [
                                    {
                                        "role": "phase_start_before_last_action_reference",
                                        "observation_id": "obs_1",
                                        "frame_id": "frame_1",
                                    },
                                    {
                                        "role": "current_observation",
                                        "observation_id": "obs_2",
                                        "frame_id": "frame_2",
                                    },
                                ]
                            },
                        }
                    ],
                },
            },
            12,
        ),
        _transcript_event(
            "assistant_turn",
            {
                "turn_id": "assistant_complete",
                "public_intent": "The requested state is now satisfied.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_complete",
                        "tool_name": "runtime.commit_phase_turn",
                        "args": {"phase": "inspect"},
                    }
                ],
            },
            13,
        ),
        _tool_result_event(
            "call_complete", "runtime.commit_phase_turn", 14
        ),
    )
    audit = BenchmarkAuditLog(
        run_dir=str(tmp_path / "run"),
        started_at=None,
        ended_at="2026-01-01T00:00:14+00:00",
        task_initial_observation_ref=None,
        observations=(),
        frames=(),
        poses=(),
        actions=(),
        events=(),
        debug_events=(),
        planner_debug=(),
        agent_transcript=transcript,
    )
    _write_case_review(tmp_path / "eval", audit, assets)

    html = (case_dir / "audit_review.html").read_text(encoding="utf-8")
    assert "Agent Tool Transcript" in html
    assert "Find &lt;person&gt; and turn safely." in html
    assert "Provider public intent" in html
    assert "Confirm &lt;candidate&gt; before moving." in html
    assert "call_detect_bad" in html and "rejected" in html
    assert "Use the latest observation_id." in html
    assert "call_commit" in html and "call_turn" in html and "call_move" in html
    assert "requested_args" in html and "executed_args" in html
    assert "safety_interventions" in html
    assert "skill.metric_depth.sample_region" in html
    assert "distance_m" in html and "2.9" in html
    assert "Internal depth calculation audit" in html and "mad_m" in html
    assert "Requested metric-depth ROI" in html
    assert "Manual ROI review required." in html
    assert "semantic target intended by the model" in html
    assert "depth-roi-box" in html
    assert "skill.visual_grounding.measure_visible_depth" in html
    assert "candidate_01" in html
    assert "Internal grounding and depth audit" in html
    assert "Manual candidate review required." in html
    assert "review_assets/frames/obs_1.png" in html
    assert "review_assets/frames/obs_2.png" in html
    assert "HISTORY" in html
    assert "CURRENT" in html
    assert '<a href="../../not-linked.txt"' not in html
    assert "Transcript audit issues" not in html


def test_core_review_reports_unclosed_duplicate_and_orphan_tool_records(
    tmp_path,
):
    transcript = (
        _transcript_event(
            "assistant_turn",
            {
                "public_intent": "First",
                "tool_calls": [
                    {
                        "tool_call_id": "duplicate",
                        "tool_name": "uav.hover",
                        "args": {},
                    },
                    {
                        "tool_call_id": "open",
                        "tool_name": "uav.hover",
                        "args": {},
                    },
                ],
            },
            1,
        ),
        _transcript_event(
            "assistant_turn",
            {
                "public_intent": "Second",
                "tool_calls": [
                    {
                        "tool_call_id": "duplicate",
                        "tool_name": "uav.turn_relative",
                        "args": {"yaw_deg": 5},
                    }
                ],
            },
            2,
        ),
        _tool_result_event("orphan", "uav.hover", 3),
    )
    audit = BenchmarkAuditLog(
        run_dir=str(tmp_path / "run"),
        started_at=None,
        ended_at="2026-01-01T00:00:03+00:00",
        task_initial_observation_ref=None,
        observations=(),
        frames=(),
        poses=(),
        actions=(),
        events=(),
        debug_events=(),
        planner_debug=(),
        agent_transcript=transcript,
    )
    _write_case_review(tmp_path / "eval", audit, ())

    html = (
        tmp_path / "eval" / "cases" / "transcript" / "audit_review.html"
    ).read_text(encoding="utf-8")
    assert "duplicate ToolCall id: duplicate" in html
    assert "ToolCall has no terminal ToolResult: open" in html
    assert "orphan ToolResult: orphan" in html


def test_core_review_labels_fallback_intent_and_provider_rejection(tmp_path):
    transcript = (
        _transcript_event(
            "provider_turn_rejected",
            {
                "error": "unknown provider tool: unsafe_<tool>",
                "finish_reason": "tool_calls",
                "content_present": False,
                "content": None,
                "tool_calls": [
                    {
                        "tool_call_id": "bad_<id>",
                        "wire_name": "unsafe_<tool>",
                        "arguments": {"payload": "<script>bad()</script>"},
                        "arguments_status": "parsed",
                    }
                ],
            },
            1,
        ),
        _transcript_event(
            "protocol_feedback",
            {"message": "Use an exposed tool."},
            2,
        ),
        _transcript_event(
            "assistant_turn",
            {
                "turn_id": "assistant_fixed",
                "public_intent": "Commit phase decision for inspect.",
                "public_intent_source": "tool_arguments_fallback",
                "tool_calls": [
                    {
                        "tool_call_id": "commit_fixed",
                        "tool_name": "runtime.commit_phase_turn",
                        "args": {"phase": "inspect"},
                    }
                ],
            },
            3,
        ),
        _tool_result_event(
            "commit_fixed", "runtime.commit_phase_turn", 4
        ),
        _transcript_event(
            "task_terminal",
            {
                "completed": False,
                "stopped": True,
                "stopped_reason": "agent_reported_blocked:insufficient_evidence",
                "stop_source": "runtime.report_blocked",
                "blocking_condition": "insufficient_evidence",
                "evidence_refs": ["obs_1"],
                "suggested_user_action": "Provide &lt;landmark&gt;.",
            },
            5,
        ),
    )
    audit = BenchmarkAuditLog(
        run_dir=str(tmp_path / "run"),
        started_at=None,
        ended_at="2026-01-01T00:00:04+00:00",
        task_initial_observation_ref=None,
        observations=(),
        frames=(),
        poses=(),
        actions=(),
        events=(),
        debug_events=(),
        planner_debug=(),
        agent_transcript=transcript,
    )

    _write_case_review(tmp_path / "eval", audit, ())

    html = (
        tmp_path / "eval" / "cases" / "transcript" / "audit_review.html"
    ).read_text(encoding="utf-8")
    assert "Provider Turn Rejected" in html
    assert "Not executed; audit only." in html
    assert "Intent reconstructed from Tool arguments" in html
    assert "unsafe_&lt;tool&gt;" in html
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "ToolCall has no terminal ToolResult: bad_&lt;id&gt;" not in html
    assert "Instruction stopped incomplete." in html
    assert "agent_reported_blocked:insufficient_evidence" in html
    assert "Provide &amp;lt;landmark&amp;gt;." in html


def test_core_review_correlates_completion_and_renders_planner_window(
    tmp_path,
):
    run_dir = tmp_path / "run"
    write_audit_fixture(run_dir)
    debug_dir = run_dir / "debug" / "action_planner"
    debug_dir.mkdir(parents=True)
    (debug_dir / "phase_window_001.json").write_text(
        json.dumps(
            {
                "phase": "align",
                "phase_key": "1:align",
                "window_index": 1,
                "model_context": {
                    "active_instruction": "Turn to the person",
                    "current_phase": {
                        "name": "align",
                        "done_when": ["the person is visible"],
                    },
                    "phase_progress_evidence": {
                        "motion_progress": {
                            "yaw_only_so_far": True,
                            "has_translation": False,
                        }
                    },
                    "fragments": [],
                },
                "raw_plan": {
                    "steps": [
                        {
                            "tool": "uav.turn_relative",
                            "args": {"yaw_deg": 15.0},
                        }
                    ]
                },
                "safe_plan": {
                    "steps": [
                        {
                            "tool": "uav.turn_relative",
                            "args": {"yaw_deg": 15.0},
                        }
                    ]
                },
                "coupled_phase_decision": {
                    "decision": "continue_current_phase",
                    "reason": "target not visible",
                },
                "coupled_plan": {
                    "completion_assessment": {
                        "done_when": [
                            {
                                "index": 0,
                                "status": "unknown",
                                "evidence": "person not visible in NOW",
                            }
                        ]
                    }
                },
                "safety_decision": {
                    "plan_changed": False,
                    "image_url": "data:image/png;base64,secret",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": "2026-01-01T00:00:03+00:00",
            "event": "phase_completion_assessed",
            "payload": {
                "phase": "align",
                "phase_key": "1:align",
                "window_index": 1,
                "assessment": {
                    "done_when": [
                        {
                            "index": 0,
                            "status": "unknown",
                            "evidence": "person not visible in NOW",
                        }
                    ]
                },
                "semantic_complete": False,
                "metric_complete": True,
                "derived_decision": "continue_current_phase",
                "fallback": False,
            },
        },
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": "2026-01-01T00:00:03+00:00",
            "event": "phase_transition_continued",
            "payload": {
                "phase": "align",
                "decision": "continue_current_phase",
                "transition_source": "coupled_planner",
                "reason": "target not visible",
            },
        },
    )

    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(run_dir))
    assert audit.planner_debug[0]["_phase_completion_result"][
        "semantic_complete"
    ] is False

    eval_dir = tmp_path / "eval"
    BenchmarkReviewRenderer(detailed_audit=True).write(
        eval_dir,
        BenchmarkSummaryReview(
            benchmark_name="Test",
            protocol_variant="test.v1",
            metrics={},
            cases=(
                {
                    "case_id": "planner",
                    "status": "ok",
                    "metrics": {},
                },
            ),
        ),
        (
            BenchmarkCaseReview(
                case_id="planner",
                benchmark_name="Test",
                protocol_variant="test.v1",
                instruction="Turn to the person",
                outcome={"status": "ok"},
                metrics={},
                audit=audit,
                frame_assets=(),
            ),
        ),
    )

    case_html = (
        eval_dir / "cases" / "planner" / "audit_review.html"
    ).read_text(encoding="utf-8")
    assert "No RGB frame assets available." in case_html
    assert "Completion Assessment" in case_html
    assert "the person is visible" in case_html
    assert "person not visible in NOW" in case_html
    assert "continue_current_phase" in case_html
    assert "Runtime Applied Phase State" in case_html
    assert "coupled_planner" in case_html
    assert "Saved Planner Context (legacy; not an exact provider request)" in case_html
    assert "provider_request_media_not_captured" in case_html
    assert "yaw-only so far" in case_html
    assert "data:image/png;base64" not in case_html
    assert "&lt;binary payload omitted&gt;" in case_html


def test_core_review_renders_exact_successful_model_images_and_steps(tmp_path):
    run_dir = tmp_path / "run"
    write_audit_fixture(run_dir)
    debug_dir = run_dir / "debug" / "action_planner"
    phase_windows_dir = run_dir / "debug" / "phase_windows"
    debug_dir.mkdir(parents=True)
    phase_windows_dir.mkdir(parents=True)
    image_data = "data:image/png;base64," + base64.b64encode(
        _png_bytes()
    ).decode("ascii")

    def planner_context() -> dict:
        return {
            "active_instruction": "Pass the trees on the left",
            "current_phase": {"name": "pass", "done_when": []},
            "fragments": [
                {
                    "name": "image",
                    "model_inputs": [
                        {
                            "type": "image",
                            "input_id": "past",
                            "role": "phase_start_reference",
                            "usage": "PAST image.",
                            "encoding": "data_url",
                            "data": image_data,
                        },
                        {
                            "type": "image",
                            "input_id": "now",
                            "role": "current_observation",
                            "usage": "NOW image.",
                            "encoding": "data_url",
                            "data": image_data,
                        },
                        {
                            "type": "image",
                            "input_id": "overlay",
                            "role": "current_rgb_depth_overlay",
                            "usage": "NOW RGB-depth overlay.",
                            "encoding": "data_url",
                            "data": image_data,
                        },
                        {
                            "type": "image",
                            "input_id": "missing",
                            "role": "auxiliary_view",
                            "usage": "Auxiliary image.",
                            "encoding": "data_url",
                            "data": "data:image/png;base64,not-base64",
                        },
                    ],
                }
            ],
        }

    for window_index, recovery in (
        (
            1,
            {
                "attempts": [
                    {"attempt": 1, "image_mode": "all", "status": "error"},
                    {
                        "attempt": 2,
                        "image_mode": "current_only",
                        "status": "ok",
                    },
                ]
            },
        ),
        (
            2,
            {"attempts": [{"attempt": 1, "image_mode": "all", "status": "ok"}]},
        ),
    ):
        (debug_dir / f"phase_window_{window_index:03d}.json").write_text(
            json.dumps(
                {
                    "phase": "pass",
                    "phase_key": "0:pass",
                    "window_index": window_index,
                    "planner_context": planner_context(),
                    "model_context": {
                        "active_instruction": "Pass the trees on the left",
                        "current_phase": {"name": "pass", "done_when": []},
                        "fragments": [],
                    },
                    "raw_plan": {
                        "steps": [
                            {
                                "tool": "uav.move_body",
                                "args": {"vy": -1.0},
                                "reason": "Establish the left side.",
                                "modalities": ["rgb", "depth"],
                            }
                        ]
                    },
                    "safe_plan": {
                        "steps": [
                            {
                                "tool": "uav.move_body",
                                "args": {"vy": -0.5},
                                "reason": "Safety-limited left movement.",
                                "modalities": ["rgb", "depth"],
                            }
                        ]
                    },
                    "coupled_phase_decision": {
                        "decision": "continue_current_phase",
                        "reason": "continue",
                    },
                    "model_request_recovery": recovery,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (phase_windows_dir / f"pass_window_{window_index:03d}.json").write_text(
            json.dumps(
                {
                    "executed_steps": [
                        {
                            "phase": "pass",
                            "step_index": 0,
                            "step": {
                                "tool": "uav.move_body",
                                "args": {"vy": -0.5},
                                "reason": "Safety-limited left movement.",
                                "modalities": ["rgb", "depth"],
                            },
                            "result": {
                                "ok": window_index == 1,
                                "message": (
                                    "moved left"
                                    if window_index == 1
                                    else "blocked by obstacle"
                                ),
                            },
                            "stabilization": {
                                "status": "completed",
                                "reason": "velocity settled",
                            },
                        }
                    ],
                    "safety_interventions": ["limited lateral velocity"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "timestamp": f"2026-01-01T00:00:0{window_index + 3}+00:00",
                "event": "phase_window_recorded",
                "payload": {
                    "phase": "pass",
                    "phase_key": "0:pass",
                    "window_index": window_index,
                    "debug_ref": (
                        f"debug/phase_windows/pass_window_{window_index:03d}.json"
                    ),
                },
            },
        )

    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(run_dir))
    assert all(
        record.get("_phase_window_execution", {}).get("executed_steps")
        for record in audit.planner_debug
    )

    eval_dir = tmp_path / "eval"
    case_review = BenchmarkCaseReview(
        case_id="planner",
        benchmark_name="Test",
        protocol_variant="test.v1",
        instruction="Pass the trees on the left",
        outcome={"status": "ok"},
        metrics={},
        audit=audit,
        frame_assets=(),
    )
    renderer = BenchmarkReviewRenderer(detailed_audit=True)
    renderer.write(
        eval_dir,
        BenchmarkSummaryReview(
            benchmark_name="Test",
            protocol_variant="test.v1",
            metrics={},
            cases=({"case_id": "planner", "status": "ok", "metrics": {}},),
        ),
        (case_review,),
    )

    case_dir = eval_dir / "cases" / "planner"
    case_html = (case_dir / "audit_review.html").read_text(encoding="utf-8")
    assets = sorted((case_dir / "review_assets" / "planner_inputs").glob("*.png"))
    assert assets == []
    assert "provider_request_media_not_captured" in case_html
    assert "Steps" in case_html
    assert "accepted ToolCall" in case_html
    assert "safety-reviewed call" in case_html
    assert "actual execution" in case_html
    assert "Safety-limited left movement." in case_html
    assert "success" in case_html
    assert "failed" in case_html
    assert "stabilization: completed (velocity settled)" in case_html
    assert "data:image/png;base64" not in case_html
    renderer.write_case(eval_dir, case_review)
    assert sorted((case_dir / "review_assets" / "planner_inputs").glob("*.png")) == assets


def test_core_review_renders_provider_rounds_and_keeps_terminal_decision_separate(
    tmp_path,
):
    run_dir = tmp_path / "run"
    write_audit_fixture(run_dir)
    planner_dir = run_dir / "debug" / "action_planner"
    planner_dir.mkdir(parents=True)
    (run_dir / "debug" / "provider_media").mkdir(parents=True)
    (run_dir / "debug" / "phase_windows").mkdir(parents=True)
    image_bytes = _png_bytes()
    digest = hashlib.sha256(image_bytes).hexdigest()
    media_ref = f"debug/provider_media/{digest}.png"
    (run_dir / media_ref).write_bytes(image_bytes)
    request = {
        "model": "stub-vl",
        "messages": [
            {"role": "system", "content": "exact system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "STATE CHECKPOINT obs_0001"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"[provider_media sha256={digest}]"
                        },
                    },
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "runtime_commit_phase_turn"},
            }
        ],
        "tool_choice": "auto",
        "temperature": 0,
    }
    response = {
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "commit_1",
                    "type": "function",
                    "function": {
                        "name": "runtime_commit_phase_turn",
                        "arguments": "{\"phase\":\"align\"}",
                    },
                }
            ],
        },
        "usage": {"prompt_tokens": 10},
    }
    common = {
        "schema_version": "agent_decision_debug_v2",
        "phase": "align",
        "phase_key": "1:align",
        "model_context": {"current_phase": {"name": "align"}},
        "planner_context": {},
        "safety_decision": {"interventions": [], "plan_changed": False},
    }
    decision_1 = {
        **common,
        "decision_id": "decision_0001",
        "decision_index": 1,
        "checkpoint_observation_id": "obs_0001",
        "completed_window_count": 0,
        "physical_window_index": 1,
        "window_index": 1,
        "model_rounds": [
            {
                "round_index": 1,
                "request": request,
                "request_media": [
                    {
                        "sha256": digest,
                        "mime_type": "image/png",
                        "suffix": ".png",
                        "request_path": [
                            "messages", 1, "content", 1, "image_url", "url"
                        ],
                        "timeline_role": "CURRENT",
                        "observation_id": "obs_0001",
                        "frame_id": "frame_0001",
                        "scene_size": "256x256",
                        "asset_ref": media_ref,
                    }
                ],
                "response": response,
                "canonical_turn": {
                    "public_intent": "hover once",
                    "public_intent_source": "tool_arguments_fallback",
                },
                "runtime_outcome": {
                    "status": "commit_accepted",
                    "feedback": None,
                },
            }
        ],
        "raw_plan": {"steps": [{"tool": "uav.hover", "args": {}}]},
        "safe_plan": {"steps": [{"tool": "uav.hover", "args": {}}]},
        "coupled_phase_decision": {
            "decision": "continue_current_phase",
            "reason": "observe once",
        },
    }
    decision_2 = {
        **common,
        "decision_id": "decision_0002",
        "decision_index": 2,
        "checkpoint_observation_id": "obs_0002",
        "completed_window_count": 1,
        "physical_window_index": None,
        "window_index": None,
        "model_rounds": [
            {
                "round_index": 1,
                "request": {
                    **request,
                    "messages": [{"role": "system", "content": "terminal"}],
                },
                "request_media": [],
                "response": response,
                "runtime_outcome": {
                    "status": "commit_accepted",
                    "feedback": None,
                },
            }
        ],
        "raw_plan": {"steps": []},
        "safe_plan": {"steps": []},
        "coupled_phase_decision": {
            "decision": "complete_task",
            "reason": "complete",
        },
    }
    for payload in (decision_1, decision_2):
        (planner_dir / f"align_{payload['decision_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    window_ref = "debug/phase_windows/align_window_001.json"
    (run_dir / window_ref).write_text(
        json.dumps(
            {
                "decision_id": "decision_0001",
                "executed_steps": [
                    {
                        "step_index": 0,
                        "step": {"tool": "uav.hover", "args": {}},
                        "result": {"ok": True, "message": "hovered"},
                    }
                ],
                "safety_interventions": [],
            }
        ),
        encoding="utf-8",
    )
    for decision_id, window_index in (
        ("decision_0001", 1),
        ("decision_0002", None),
    ):
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "timestamp": "2026-01-01T00:00:06+00:00",
                "event": "phase_completion_assessed",
                "payload": {
                    "decision_id": decision_id,
                    "phase": "align",
                    "phase_key": "1:align",
                    "window_index": window_index,
                },
            },
        )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": "2026-01-01T00:00:06+00:00",
            "event": "phase_window_recorded",
            "payload": {
                "decision_id": "decision_0001",
                "phase": "align",
                "phase_key": "1:align",
                "window_index": 1,
                "debug_ref": window_ref,
            },
        },
    )

    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(run_dir))
    assert audit.planner_debug[0].get("_phase_window_execution")
    assert "_phase_window_execution" not in audit.planner_debug[1]
    eval_dir = tmp_path / "eval"
    case = BenchmarkCaseReview(
        case_id="decision",
        benchmark_name="Test",
        protocol_variant="test.v1",
        instruction="Turn",
        outcome={"status": "ok"},
        metrics={},
        audit=audit,
        frame_assets=(),
    )
    renderer = BenchmarkReviewRenderer(detailed_audit=True)
    renderer.write_case(eval_dir, case)
    renderer.write_case(eval_dir, case)
    page = (eval_dir / "cases" / "decision" / "audit_review.html").read_text(
        encoding="utf-8"
    )

    assert (
        "Decision 1 · obs_0001 · 1 model round · continue_current_phase · "
        "physical window 1"
    ) in page
    assert (
        "Decision 2 · obs_0002 · 1 model round · complete_task · "
        "no physical window"
    ) in page
    assert "Actual Provider Request" in page
    assert "exact system prompt" in page
    assert "runtime_commit_phase_turn" in page
    assert "tool_arguments_fallback" in page
    assert "#1 · CURRENT" in page
    assert "observation_id=obs_0001" in page
    assert "scene_size=256x256" in page
    assert "accepted ToolCall" in page
    assert "data:image" not in page
    copied = list(
        (
            eval_dir
            / "cases"
            / "decision"
            / "review_assets"
            / "provider_inputs"
        ).glob("*.png")
    )
    assert len(copied) == 1


def test_legacy_tool_transcript_is_rejected(tmp_path):
    run_dir = tmp_path / "legacy_run"
    write_audit_fixture(run_dir)
    transcript_path = run_dir / "memory" / "agent_transcript.jsonl"
    records = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in records:
        record.pop("schema_version", None)
    transcript_path.write_text(
        "\n".join(json.dumps(item) for item in records), encoding="utf-8"
    )

    with pytest.raises(UnsupportedRunSchema, match="agent_transcript_v3"):
        BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(run_dir))


def make_episode(
    *,
    profile: ObservationProfile | None = None,
    options: TaskExecutionOptions | None = None,
) -> PreparedBenchmarkEpisode[dict, dict]:
    selected_profile = profile or ObservationProfile()
    return PreparedBenchmarkEpisode(
        benchmark_name="test",
        case_id="case_1",
        protocol_variant="test.native.v1",
        instruction="Move forward",
        map_id="map",
        scene_id="scene",
        agent_id="drone",
        observation_profile=selected_profile,
        execution_options=options
        or TaskExecutionOptions.from_observation_profile(selected_profile),
        planner_safe_metadata={"weather": "clear"},
        scenario_payload={"object": "person"},
        evaluation_payload={"goal": [100, 200]},
    )


class FakeRuntime:
    def __init__(self, run_dir: Path, calls: list[str]) -> None:
        self.run_dir = run_dir
        self.calls = calls
        self.finished_state = "unset"
        self.scene = SceneInstance(
            "scene_instance",
            "map",
            "scene",
            "local",
            "127.0.0.1",
            9000,
            None,
            None,
            "running",
            capabilities=SceneCapabilities(
                "unrealcv",
                "actor_pose",
                ("connect", "hover", "set_pose", "move_body"),
                ("rgb", "depth", "pose"),
                False,
                False,
            ),
        )

    def ensure_reused_scene(self, map_id, scene_id):
        self.calls.append("ensure")
        return self.scene

    def run_task(self, *args, **kwargs):
        self.calls.append("run")
        write_trace_fixture(self.run_dir)
        return SimpleNamespace(
            completed=True,
            stopped=False,
            stopped_reason=None,
            stop_source=None,
        )

    def finish_reused_scene(self, map_id, state, **kwargs):
        self.calls.append("finish")
        self.finished_state = state
        append_jsonl(
            self.run_dir / "memory" / "actions.jsonl",
            {
                "id": "act_shutdown",
                "timestamp": "2026-01-01T00:00:09+00:00",
                "tool": "uav.land",
                "result": {"ok": True},
            },
        )


class FakeScenario:
    def __init__(self, calls: list[str], fail_prepare: bool = False) -> None:
        self.calls = calls
        self.fail_prepare = fail_prepare

    def prepare_episode(self, scene, payload):
        self.calls.append("scenario_prepare")
        if self.fail_prepare:
            raise RuntimeError("prepare failed")

    def cleanup_episode(self, scene, payload):
        self.calls.append("scenario_cleanup")


class FakeMetrics:
    def __init__(self, calls: list[str], fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def evaluate(self, payload, trace, outcome):
        self.calls.append("metrics")
        if self.fail:
            raise ValueError("metric failed")
        return {"sample_count": len(trace.samples)}


def write_trace_fixture(
    run_dir: Path,
    *,
    include_action: bool = True,
    action_pose: bool = True,
    include_read_only: bool = False,
    window_event: str = "phase_window_recorded",
) -> None:
    memory_dir = run_dir / "memory"
    state_dir = run_dir / "state"
    memory_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "event": "instruction_started",
            "payload": {"physical_trace_schema": "physical_window_v2"},
        },
    )
    if include_read_only:
        append_jsonl(
            memory_dir / "actions.jsonl",
            {
                "id": "act_depth",
                "timestamp": "2026-01-01T00:00:01.5+00:00",
                "tool": "skill.metric_depth.sample_region",
                "args": {"observation_id": "obs_0001"},
                "result": {"ok": True},
            },
        )
    append_jsonl(
        memory_dir / "actions.jsonl",
        {
            "id": "act_0001",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "tool": "uav.connect",
            "result": {"ok": True},
        },
    )
    append_jsonl(
        memory_dir / "poses.jsonl",
        {
            "id": "pose_0001",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 3.0,
                "yaw_deg": 0.0,
                "coordinate_frame": "unrealcv_world_m",
            },
            "coordinate_frame": "unrealcv_world_m",
            "source": "observation",
            "observation_id": "obs_0001",
        },
    )
    append_jsonl(
        memory_dir / "observations.jsonl",
        {
            "id": "obs_0001",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "source": "preflight",
            "pose_ref": {"ref": "memory/poses.jsonl#1"},
        },
    )
    if include_action:
        append_jsonl(
            memory_dir / "actions.jsonl",
            {
                "id": "act_0002",
                "timestamp": "2026-01-01T00:00:02+00:00",
                "tool": "uav.turn_relative",
                "args": {"yaw_deg": 15.0},
                "result": {"ok": True},
            },
        )
        if action_pose:
            append_jsonl(
                memory_dir / "poses.jsonl",
                {
                    "id": "pose_0002",
                    "timestamp": "2026-01-01T00:00:02+00:00",
                    "pose": {
                        "x": 1.0,
                        "y": 0.0,
                        "z": 3.0,
                        "yaw_deg": 15.0,
                        "coordinate_frame": "unrealcv_world_m",
                    },
                    "coordinate_frame": "unrealcv_world_m",
                    "source": "action_result",
                    "action_id": "act_0002",
                },
            )
            append_jsonl(
                memory_dir / "poses.jsonl",
                {
                    "id": "pose_0003",
                    "timestamp": "2026-01-01T00:00:03+00:00",
                    "pose": {
                        "x": 1.0,
                        "y": 0.0,
                        "z": 3.0,
                        "yaw_deg": 15.25,
                        "coordinate_frame": "unrealcv_world_m",
                    },
                    "coordinate_frame": "unrealcv_world_m",
                    "source": "observation",
                    "observation_id": "obs_0002",
                },
            )
            append_jsonl(
                memory_dir / "observations.jsonl",
                {
                    "id": "obs_0002",
                    "timestamp": "2026-01-01T00:00:03+00:00",
                    "source": "phase_window",
                    "pose_ref": {"ref": "memory/poses.jsonl#3"},
                },
            )
            pose_line = 3
        else:
            append_jsonl(
                memory_dir / "poses.jsonl",
                {
                    "id": "pose_0002",
                    "timestamp": "2026-01-01T00:00:03+00:00",
                    "pose": {
                        "x": 1.0,
                        "y": 0.0,
                        "z": 3.0,
                        "yaw_deg": 15.0,
                        "coordinate_frame": "unrealcv_world_m",
                    },
                    "coordinate_frame": "unrealcv_world_m",
                    "source": "observation",
                    "observation_id": "obs_0002",
                },
            )
            append_jsonl(
                memory_dir / "observations.jsonl",
                {
                    "id": "obs_0002",
                    "timestamp": "2026-01-01T00:00:03+00:00",
                    "source": "phase_window",
                    "pose_ref": {"ref": "memory/poses.jsonl#2"},
                },
            )
            pose_line = 2
    else:
        pose_line = 1
    if include_action:
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "timestamp": "2026-01-01T00:00:03+00:00",
                "event": window_event,
                "payload": {
                    "schema_version": "physical_window_v2",
                    "decision_id": "decision_0001",
                    "phase": "turn",
                    "phase_key": "1:turn" if window_event == "phase_window_recorded" else None,
                    "window_index": 1,
                    "observation_before_id": "obs_0001",
                    "observation_after_id": "obs_0002",
                    "status": "ok",
                    "ordered_steps": [
                        {
                            "step_index": 0,
                            "tool_call_id": "call_turn",
                            "tool": "uav.turn_relative",
                            "effect": "physical",
                            "status": "ok",
                            "action_ref": (
                                "memory/actions.jsonl#3"
                                if include_read_only
                                else "memory/actions.jsonl#2"
                            ),
                            "result_pose_ref": (
                                "memory/poses.jsonl#2"
                                if action_pose
                                else None
                            ),
                            "requested_args": {"yaw_deg": 15.0},
                            "executed_args": {"yaw_deg": 15.0},
                            "safety_interventions": [],
                        }
                    ],
                },
            },
        )
    (state_dir / "flight_memory.json").write_text(
        json.dumps(
            {
                "observation_anchors": {
                    "task_initial": {"ref": "memory/observations.jsonl#1"}
                },
                "latest_pose_line": pose_line,
            }
        ),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_audit_fixture(run_dir: Path) -> None:
    """Write one initial observation plus one formal action audit window."""

    memory = run_dir / "memory"
    state = run_dir / "state"
    memory.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        memory / "frames.jsonl",
        {
            "id": "frame_0001",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "observation_id": "obs_0001",
            "modality": "rgb",
            "path": None,
            "metadata": {},
        },
    )
    append_jsonl(
        memory / "poses.jsonl",
        {
            "id": "pose_0001",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "observation_id": "obs_0001",
            "source": "observation",
            "pose": {"x": 0.0, "y": 0.0, "z": 2.0},
        },
    )
    append_jsonl(
        memory / "observations.jsonl",
        {
            "id": "obs_0001",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "pose_ref": {"ref": "memory/poses.jsonl#1"},
            "frame_refs": [{"ref": "memory/frames.jsonl#1"}],
        },
    )
    append_jsonl(
        memory / "actions.jsonl",
        {
            "id": "act_0001",
            "timestamp": "2026-01-01T00:00:02+00:00",
            "tool": "uav.turn_relative",
            "args": {"yaw_deg": 15.0},
            "result": {"ok": True},
        },
    )
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": "2026-01-01T00:00:02.100000+00:00",
            "event": "tool_step_completed",
            "payload": {
                "phase": "align",
                "step_index": 0,
                "tool": "uav.turn_relative",
                "ok": True,
            },
        },
    )
    append_jsonl(
        memory / "actions.jsonl",
        {
            "id": "act_safety_hover",
            "timestamp": "2026-01-01T00:00:02.500000+00:00",
            "tool": "uav.hover",
            "args": {},
            "result": {"ok": True},
        },
    )
    append_jsonl(
        memory / "agent_transcript.jsonl",
        {
            "id": "turn_0001",
            "schema_version": "agent_transcript_v3",
            "timestamp": "2026-01-01T00:00:00.500000+00:00",
            "instruction_id": "inst_1",
            "kind": "user_instruction",
            "payload": {"instruction": "Turn toward the target."},
        },
    )
    append_jsonl(
        memory / "agent_transcript.jsonl",
        {
            "id": "turn_0002",
            "schema_version": "agent_transcript_v3",
            "timestamp": "2026-01-01T00:00:01.100000+00:00",
            "instruction_id": "inst_1",
            "kind": "state_checkpoint",
            "payload": {
                "schema_version": "state_checkpoint_v3",
                "observation_id": "obs_0001",
                "context": {"current_phase": {"name": "align"}},
            },
        },
    )
    (state / "flight_memory.json").write_text(
        json.dumps(
            {
                "observation_anchors": {
                    "task_initial": {
                        "ref": "memory/observations.jsonl#1"
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _png_bytes() -> bytes:
    """Return a valid one-pixel PNG without requiring image libraries."""

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4n"
        "GNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )


def _transcript_event(kind: str, payload: dict, index: int) -> dict:
    return {
        "id": f"turn_{index:04d}",
        "schema_version": "agent_transcript_v3",
        "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
        "instruction_id": "inst_transcript",
        "kind": kind,
        "payload": payload,
    }


def _tool_result_event(
    call_id: str,
    tool_name: str,
    index: int,
) -> dict:
    return _transcript_event(
        "tool_result",
        {
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "status": "ok",
            "message": "completed",
            "requested_args": {},
            "executed_args": {},
            "safety_interventions": [],
            "artifact_refs": [],
            "data": {},
        },
        index,
    )


def _write_case_review(
    eval_dir: Path,
    audit: BenchmarkAuditLog,
    assets: tuple[ResolvedFrameAsset, ...],
) -> None:
    BenchmarkReviewRenderer(detailed_audit=True).write_case(
        eval_dir,
        BenchmarkCaseReview(
            case_id="transcript",
            benchmark_name="Test",
            protocol_variant="test.v1",
            instruction="Inspect and move",
            outcome={"status": "ok"},
            metrics={},
            audit=audit,
            frame_assets=assets,
        ),
    )
