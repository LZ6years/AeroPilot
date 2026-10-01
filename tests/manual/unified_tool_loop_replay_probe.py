"""Offline scripted replay through the public Runtime, never a model response.

An optional existing RGB file is replayed as recorded imagery. It does not
simulate visual consequences of motion and cannot establish navigation success.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.core import (
    BenchmarkAuditReader, BenchmarkCaseReview, BenchmarkReviewRenderer, BenchmarkSummaryReview,
    EpisodeAuditCursor, EpisodeTraceReader, FrameAssetResolver, TraceCursor,
)
from src.env_client import Observation
from src.plugins import PluginRegistry
from src.plugins.contracts.skill import Skill
from src.plugins.skills.catalog import SkillCatalog
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.uav.config import UavActionConfig
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.factory import RuntimeFactory
from tests.test_unified_tool_loop import ManagedFakeBackend, ScriptedPlanner
from tests.support.runtime import call, done, move
from tests.manual.native_tool_loop_real_audit import _audit_transcript, _audit_physical_trace
from evaluation.benchmarks.uav_flow.metric_suite import _backend_action_tracking


def run_replay(output: Path, rgb: Path | None = None) -> dict:
    if output.exists():
        raise ValueError("Use a new output directory; historical artifacts are never overwritten")
    if rgb is not None and not rgb.is_file():
        raise ValueError("RGB source does not exist")
    backend = ManagedFakeBackend()
    if rgb is not None:
        original = backend.get_observation
        def observe(scene, agent, modalities):
            observation = original(scene, agent, modalities)
            return Observation(scene, agent, modalities, {
                **observation.data, "frames": {"rgb": {"path": str(rgb.resolve()), "mock": False}},
            })
        backend.get_observation = observe
    recorder = AsyncRunLogger(output / "run")
    planner = ScriptedPlanner([
        [call("knowledge", "knowledge.load", name="offline_guide")],
        [call("plan", "runtime.update_plan", explanation="Scripted replay only", steps=[
            {"id": "move", "description": "Exercise one ordered motion batch", "status": "in_progress",
             "evidence_refs": []}])],
        [move("move1"), move("move2")], [done()],
    ])
    plugins = PluginRegistry(skill_catalog=SkillCatalog([
            Skill(plugin_id="knowledge.offline_fixture", name="offline_guide",
                  description="Scripted offline demonstration", summary="In-memory fixture",
                  skill_dir=output, skill_path=output / "IN_MEMORY_FIXTURE",
                  body="This fixture exercises explicit loading; it does not establish navigation success.")
        ]))
    runtime = RuntimeFactory(
        output / "plugins.json", plugin_registry=plugins
    ).create_task_runtime(
        backend,
        recorder,
        planner=planner,
        stabilization_config=UavStabilizationConfig(enabled=False),
    )
    try:
        state = runtime.run_task("map", "Offline scripted two-move replay",
                                 protocol_context="Offline fixture; not a navigation experiment.")
    finally:
        recorder.close()
    audit = BenchmarkAuditReader().read(EpisodeAuditCursor.freeze(recorder.run_dir))
    trace = EpisodeTraceReader().read(TraceCursor.freeze(recorder.run_dir))
    tracking = _backend_action_tracking(trace, action_config=UavActionConfig(), completed=state.completed)
    case_data = {"trace": trace.to_dict(), "backend_action_tracking": tracking}
    case_dir = output / "cases" / "replay"
    assets = FrameAssetResolver().materialize(audit, case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = {**audit.to_dict(), "frame_assets": [asset.to_dict() for asset in assets]}
    (case_dir / "audit.json").write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "trace.json").write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "case.json").write_text(json.dumps(case_data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Audit the serialized boundary, exactly as the real offline auditor does.
    # Dataclass asdict retains tuples; durable JSON correctly contains arrays.
    case_data = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    paths = BenchmarkReviewRenderer().write_case(output, BenchmarkCaseReview(
        case_id="replay", benchmark_name="Offline scripted replay", protocol_variant="unified_tool_loop_v2",
        instruction=state.user_instruction, outcome={"status": "ok", **state.to_dict()},
        metrics={"backend_action_tracking": tracking},
        audit=audit, frame_assets=assets,
        metadata={"scripted": True, "real_provider_called": False, "navigation_assessed": False},
    ))
    violations = []
    BenchmarkReviewRenderer().write_summary(output, BenchmarkSummaryReview(
        benchmark_name="Scripted offline replay — no Provider or navigation assessment",
        protocol_variant="unified_tool_loop_v2", metrics={},
        cases=({"case_id": "replay", "status": "scripted"},),
    ))
    coverage = _audit_transcript(list(audit.agent_transcript), audit_payload, case_data, case_dir,
                                 runtime.tools.registry, violations, "replay")
    _audit_physical_trace(case_data, list(audit.agent_transcript), audit_payload,
                          runtime.tools.registry, violations, "replay")
    report = {"mode": "scripted_offline_replay", "real_provider_called": False,
              "trace_integrity": trace.trace_integrity, "tracking": tracking, "coverage": coverage,
              "violations": [item.to_dict() for item in violations], **paths}
    (output / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb", type=Path)
    args = parser.parse_args()
    report = run_replay(args.output_dir, args.rgb)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["violations"] or report["trace_integrity"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
