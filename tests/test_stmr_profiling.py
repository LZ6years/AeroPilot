"""Tests for STMR CPU/CUDA stage profiling and report aggregation."""

from __future__ import annotations

import json

from plugins.research.stmr_map.profiling import StageProfiler, artifact_size_bytes
from plugins.research.stmr_map.profiling_report import build_profile_summary, write_profile_report


class _MemoryInfo:
    rss = 128 * 1024 * 1024


class _Process:
    def memory_info(self):
        return _MemoryInfo()


class _FakeCuda:
    def __init__(self):
        self.sync_count = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.sync_count += 1

    def reset_peak_memory_stats(self):
        return None

    def memory_allocated(self):
        return 64 * 1024 * 1024

    def memory_reserved(self):
        return 96 * 1024 * 1024

    def max_memory_allocated(self):
        return 80 * 1024 * 1024


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


def test_stage_profiler_records_cpu_cuda_counts_and_cold_start():
    torch = _FakeTorch()
    profiler = StageProfiler(
        device="cuda",
        context={"case_id": "case_a"},
        torch_module=torch,
        process=_Process(),
    )

    with profiler.stage("model_load", cold_start=True, metadata={"input_count": 1}) as stage:
        stage.update(output_count=2)
    with profiler.stage("inference"):
        pass

    assert profiler.records[0]["run_phase"] == "cold_start"
    assert profiler.records[0]["output_count"] == 2
    assert profiler.records[0]["cuda_peak_allocated_mb"] == 80.0
    assert profiler.records[0]["cpu_rss_after_mb"] == 128.0
    assert torch.cuda.sync_count == 4
    summary = profiler.summary()
    assert summary["stages"]["model_load"]["cold_start_count"] == 1
    assert summary["stages"]["inference"]["sample_count"] == 1


def test_disabled_profiler_does_not_record_or_require_cuda():
    profiler = StageProfiler(enabled=False)

    with profiler.stage("ignored") as stage:
        stage.update(value=1)

    assert profiler.records == []
    assert profiler.summary()["sample_count"] == 0


def test_profile_summary_aggregates_cases_and_variants():
    runs = [
        {
            "case_id": "turn",
            "variant": "default_uav_prior",
            "frame_count": 5,
            "queries": ["person", "road"],
            "point_count": 100,
            "voxel_count": 20,
            "detection_count": 2,
            "mask_count": 2,
            "timing_records": [{"stage": "grounding_inference", "elapsed_ms": 10.0, "status": "ok", "run_phase": "warm"}],
        },
        {
            "case_id": "shift",
            "variant": "default_uav_prior",
            "frame_count": 5,
            "queries": ["road", "building"],
            "point_count": 150,
            "voxel_count": 30,
            "detection_count": 3,
            "mask_count": 3,
            "timing_records": [{"stage": "grounding_inference", "elapsed_ms": 20.0, "status": "ok", "run_phase": "warm"}],
        },
    ]

    summary = build_profile_summary(runs)
    variant = summary["variants"]["default_uav_prior"]

    assert variant["case_count"] == 2
    assert variant["frame_count"] == 10
    assert variant["projected_point_count"] == 250
    assert variant["timing"]["stages"]["grounding_inference"]["p50_ms"] == 15.0
    assert variant["queries"] == ["person", "road", "building"]
    assert "timing_records" not in summary["runs"][0]


def test_profile_summary_derives_incremental_online_latency_and_speedup():
    records = [
        {
            "stage": "grounding_inference",
            "observation_id": "obs_1",
            "elapsed_ms": 200.0,
            "status": "ok",
            "run_phase": "warm",
        },
        {
            "stage": "sam_inference",
            "observation_id": "obs_1",
            "elapsed_ms": 300.0,
            "status": "ok",
            "run_phase": "warm",
        },
        {
            "stage": "incremental_frame_total",
            "observation_id": "obs_1",
            "elapsed_ms": 400.0,
            "status": "ok",
            "run_phase": "warm",
        },
        {
            "stage": "batch_rebuild_reference",
            "elapsed_ms": 2000.0,
            "status": "ok",
            "run_phase": "warm",
        },
    ]
    runs = [
        {
            "case_id": "turn",
            "variant": "default_uav_prior",
            "frame_count": 1,
            "timing_records": records,
            "incremental_update": {
                "batch_reference_parity": {"equivalent": True},
            },
        }
    ]

    variant = build_profile_summary(runs)["variants"]["default_uav_prior"]

    assert variant["timing"]["stages"]["online_frame_component_total"]["p50_ms"] == 900.0
    assert variant["incremental_performance"]["map_frame_p50_ms"] == 400.0
    assert variant["incremental_performance"]["batch_to_incremental_map_speedup"] == 5.0
    assert variant["parity_pass_count"] == 1


def test_online_latency_keeps_same_observation_ids_separate_across_cases():
    runs = []
    for case_id, elapsed_ms in (("turn", 100.0), ("shift", 300.0)):
        runs.append(
            {
                "case_id": case_id,
                "variant": "default_uav_prior",
                "frame_count": 1,
                "timing_records": [
                    {
                        "case_id": case_id,
                        "stage": "incremental_frame_total",
                        "observation_id": "obs_1",
                        "elapsed_ms": elapsed_ms,
                        "status": "ok",
                        "run_phase": "warm",
                    }
                ],
            }
        )

    stage = build_profile_summary(runs)["variants"]["default_uav_prior"]["timing"]["stages"]

    assert stage["online_frame_component_total"]["sample_count"] == 2
    assert stage["online_frame_component_total"]["p50_ms"] == 200.0


def test_profile_report_writes_lightweight_json_and_html(tmp_path):
    png = tmp_path / "map.png"
    png.write_bytes(b"png")
    runs = [
        {
            "case_id": "turn",
            "class_name": "Turn",
            "variant": "urban_uav_prior_v1",
            "queries": ["person", "building"],
            "frame_count": 5,
            "point_count": 10,
            "voxel_count": 4,
            "detection_count": 1,
            "mask_count": 1,
            "stmr_png": str(png),
            "timing_records": [],
        }
    ]

    json_path, html_path, _ = write_profile_report(runs, tmp_path / "report")

    assert json_path.exists()
    assert html_path.exists()
    assert "STMR Incremental Urban Prior Profile" in html_path.read_text(encoding="utf-8")
    assert "urban_uav_prior_v1" in html_path.read_text(encoding="utf-8")
    assert "timing_records" not in json_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_count"] == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "stmr_incremental_profile_v2"
    assert artifact_size_bytes(png) == 3
