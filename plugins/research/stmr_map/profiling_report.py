"""Aggregate STMR profiling runs into compact JSON and static HTML."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from typing import Any

from .profiling import StageProfiler


_PER_FRAME_PERCEPTION_STAGES = {
    "grounding_preprocess",
    "grounding_inference",
    "grounding_postprocess",
    "tfidf_and_semantic_filter",
    "sam_preprocess",
    "sam_inference",
    "sam_mask_postprocess",
    "depth_and_dynamic_candidate_stats",
    "mask_overlay_and_memory_write",
}


def build_profile_summary(
    runs: list[dict[str, Any]],
    *,
    cold_start_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate per-case records without discarding the case-level evidence."""

    variants: dict[str, dict[str, Any]] = {}
    for run in runs:
        variant = str(run.get("variant") or "unknown")
        item = variants.setdefault(
            variant,
            {
                "case_count": 0,
                "frame_count": 0,
                "queries": [],
                "records": [],
                "projected_point_count": 0,
                "voxel_count": 0,
                "detection_count": 0,
                "mask_count": 0,
                "parity_pass_count": 0,
            },
        )
        item["case_count"] += 1
        item["frame_count"] += int(run.get("frame_count", 0))
        item["projected_point_count"] += int(run.get("point_count", 0))
        item["voxel_count"] += int(run.get("voxel_count", 0))
        item["detection_count"] += int(run.get("detection_count", 0))
        item["mask_count"] += int(run.get("mask_count", 0))
        parity = ((run.get("incremental_update") or {}).get("batch_reference_parity") or {})
        item["parity_pass_count"] += int(bool(parity.get("equivalent")))
        item["records"].extend(run.get("timing_records", []))
        for query in run.get("queries", []):
            if query not in item["queries"]:
                item["queries"].append(query)

    for item in variants.values():
        profiler = StageProfiler(enabled=False)
        records = list(item.pop("records"))
        profiler.records = [*records, *_online_frame_component_records(records)]
        item["timing"] = profiler.summary()
        stages = item["timing"].get("stages", {})
        incremental = stages.get("incremental_frame_total", {})
        batch = stages.get("batch_rebuild_reference", {})
        incremental_p50 = float(incremental.get("p50_ms", 0.0) or 0.0)
        batch_p50 = float(batch.get("p50_ms", 0.0) or 0.0)
        item["incremental_performance"] = {
            "map_frame_p50_ms": incremental_p50,
            "map_frame_p95_ms": float(incremental.get("p95_ms", 0.0) or 0.0),
            "online_component_p50_ms": float(
                stages.get("online_frame_component_total", {}).get("p50_ms", 0.0) or 0.0
            ),
            "online_component_p95_ms": float(
                stages.get("online_frame_component_total", {}).get("p95_ms", 0.0) or 0.0
            ),
            "batch_rebuild_p50_ms": batch_p50,
            "batch_to_incremental_map_speedup": round(batch_p50 / incremental_p50, 3)
            if incremental_p50 > 0.0
            else None,
        }

    cold_profiler = StageProfiler(enabled=False)
    cold_profiler.records = list(cold_start_records or [])
    return {
        "schema_version": "stmr_incremental_profile_v2",
        "run_count": len(runs),
        "case_count": len({str(item.get("case_id")) for item in runs}),
        "variant_count": len(variants),
        "cold_start": cold_profiler.summary(),
        "variants": variants,
        "runs": [_without_timing_records(item) for item in runs],
    }


def write_profile_report(
    runs: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    cold_start_records: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write ``profile_summary.json`` and a self-contained comparison page."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_profile_summary(runs, cold_start_records=cold_start_records)
    json_path = output_dir / "profile_summary.json"
    html_path = output_dir / "profile_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(summary, output_dir), encoding="utf-8")
    return json_path, html_path, summary


def _without_timing_records(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "timing_records"}


def _online_frame_component_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build measured component sums for perception plus incremental map work."""

    per_observation: dict[tuple[str, str], float] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        stage = str(record.get("stage") or "")
        if stage not in _PER_FRAME_PERCEPTION_STAGES and stage != "incremental_frame_total":
            continue
        observation_id = record.get("observation_id")
        if not observation_id:
            continue
        key = (str(record.get("case_id") or "unknown_case"), str(observation_id))
        per_observation[key] = per_observation.get(key, 0.0) + float(record.get("elapsed_ms", 0.0) or 0.0)

    dynamic_policy_by_case: dict[str, float] = {}
    for record in records:
        if record.get("status") != "ok" or record.get("stage") != "dynamic_track_policy":
            continue
        case_id = str(record.get("case_id") or "unknown_case")
        dynamic_policy_by_case[case_id] = dynamic_policy_by_case.get(case_id, 0.0) + float(
            record.get("elapsed_ms", 0.0) or 0.0
        )
    frame_counts: dict[str, int] = {}
    for case_id, _observation_id in per_observation:
        frame_counts[case_id] = frame_counts.get(case_id, 0) + 1
    return [
        {
            "stage": "online_frame_component_total",
            "case_id": case_id,
            "observation_id": observation_id,
            "elapsed_ms": round(
                elapsed_ms + dynamic_policy_by_case.get(case_id, 0.0) / max(1, frame_counts.get(case_id, 1)),
                4,
            ),
            "status": "ok",
            "run_phase": "warm",
        }
        for (case_id, observation_id), elapsed_ms in per_observation.items()
    ]


def _render_html(summary: dict[str, Any], output_dir: Path) -> str:
    runs = summary.get("runs", [])
    variants = summary.get("variants", {})
    variant_cards = "".join(_variant_card(name, item) for name, item in variants.items())
    stage_comparison = _stage_comparison(variants)
    run_rows = "".join(_run_row(item, output_dir) for item in runs)
    cold = summary.get("cold_start", {}).get("stages", {})
    cold_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{data.get('p50_ms', 0):.2f}</td>"
        f"<td>{data.get('p95_ms', 0):.2f}</td><td>{_fmt(data.get('cuda_peak_allocated_mb'))}</td>"
        f"<td>{_fmt(data.get('cpu_rss_peak_mb'))}</td></tr>"
        for name, data in cold.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STMR Incremental Urban Prior Profile</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#64748b; --line:#d8dee9; --soft:#f4f7fa; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:14px/1.45 Arial,sans-serif; color:var(--ink); background:#fff; }}
    header {{ padding:24px 30px; border-bottom:1px solid var(--line); }} main {{ padding:24px 30px 40px; }}
    h1,h2,h3 {{ margin:0 0 12px; letter-spacing:0; }} h2 {{ margin-top:28px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:14px; }}
    .card {{ border:1px solid var(--line); border-radius:6px; padding:14px; background:#fff; }}
    .queries {{ color:var(--muted); word-break:break-word; }} table {{ width:100%; border-collapse:collapse; }}
    th,td {{ padding:8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ background:var(--soft); }}
    img {{ display:block; width:180px; max-height:120px; object-fit:contain; background:#111827; }}
    .media {{ display:flex; gap:8px; flex-wrap:wrap; }} a {{ color:#075985; }} code {{ background:var(--soft); padding:2px 4px; }}
    .muted {{ color:var(--muted); }}
  </style>
</head>
<body>
  <header><h1>STMR Incremental Urban Prior Profile</h1><div class="muted">Same 15 source frames, detector-query prior A/B with all-history incremental world voxels.</div></header>
  <main>
    <div class="grid">{variant_cards}</div>
    <h2>Cold Start</h2>
    <table><thead><tr><th>Stage</th><th>P50 ms</th><th>P95 ms</th><th>CUDA peak MB</th><th>RSS peak MB</th></tr></thead><tbody>{cold_rows}</tbody></table>
    <h2>Warm Stage Comparison</h2>
    {stage_comparison}
    <h2>Case / Variant Evidence</h2>
    <table><thead><tr><th>Case</th><th>Variant and queries</th><th>Perception</th><th>Map</th><th>Artifacts</th></tr></thead><tbody>{run_rows}</tbody></table>
  </main>
</body>
</html>"""


def _variant_card(name: str, item: dict[str, Any]) -> str:
    stages = item.get("timing", {}).get("stages", {})
    total = stages.get("complete_pipeline_total", {})
    cuda_peak = max(
        (float(stage["cuda_peak_allocated_mb"]) for stage in stages.values() if stage.get("cuda_peak_allocated_mb") is not None),
        default=0.0,
    )
    rss_peak = max(
        (float(stage["cpu_rss_peak_mb"]) for stage in stages.values() if stage.get("cpu_rss_peak_mb") is not None),
        default=0.0,
    )
    incremental = item.get("incremental_performance", {})
    return (
        '<article class="card">'
        f"<h3>{escape(name)}</h3>"
        f"<div>{item.get('case_count', 0)} cases / {item.get('frame_count', 0)} frames</div>"
        f"<div>{item.get('detection_count', 0)} detections / {item.get('mask_count', 0)} masks</div>"
        f"<div>{item.get('projected_point_count', 0)} points / {item.get('voxel_count', 0)} voxels</div>"
        f"<div>pipeline P50/P95: {_fmt(total.get('p50_ms'))} / {_fmt(total.get('p95_ms'))} ms</div>"
        f"<div>incremental map P50/P95: {_fmt(incremental.get('map_frame_p50_ms'))} / {_fmt(incremental.get('map_frame_p95_ms'))} ms</div>"
        f"<div>online component P50/P95: {_fmt(incremental.get('online_component_p50_ms'))} / {_fmt(incremental.get('online_component_p95_ms'))} ms</div>"
        f"<div>batch rebuild P50 / speedup: {_fmt(incremental.get('batch_rebuild_p50_ms'))} ms / {_fmt(incremental.get('batch_to_incremental_map_speedup'))}x</div>"
        f"<div>batch parity: {item.get('parity_pass_count', 0)}/{item.get('case_count', 0)}</div>"
        f"<div>peak CUDA/RSS: {_fmt(cuda_peak)} / {_fmt(rss_peak)} MB</div>"
        f"<div class=\"queries\">{escape(', '.join(item.get('queries', [])))}</div>"
        "</article>"
    )


def _stage_comparison(variants: dict[str, dict[str, Any]]) -> str:
    names = list(variants)
    stage_names = sorted(
        {
            stage
            for variant in variants.values()
            for stage in variant.get("timing", {}).get("stages", {})
        }
    )
    header = "".join(f"<th>{escape(name)}</th>" for name in names)
    rows = []
    for stage_name in stage_names:
        cells = []
        for name in names:
            stage = variants[name].get("timing", {}).get("stages", {}).get(stage_name, {})
            cells.append(
                "<td>"
                f"P50 {_fmt(stage.get('p50_ms'))} ms<br>"
                f"P95 {_fmt(stage.get('p95_ms'))} ms<br>"
                f"CUDA {_fmt(stage.get('cuda_peak_allocated_mb'))} MB<br>"
                f"RSS {_fmt(stage.get('cpu_rss_peak_mb'))} MB"
                "</td>"
            )
        rows.append(f"<tr><td><code>{escape(stage_name)}</code></td>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>Stage</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _run_row(item: dict[str, Any], output_dir: Path) -> str:
    boxes = _media(item.get("boxes_path"), output_dir, "boxes")
    overlay = _media(item.get("overlay_path"), output_dir, "SAM overlay")
    png = _media(item.get("stmr_png"), output_dir, "STMR map")
    html_link = _link(item.get("stmr_html"), output_dir, "open 2D/3D viewer")
    incremental = item.get("incremental_update") or {}
    parity = incremental.get("batch_reference_parity") or {}
    return (
        "<tr>"
        f"<td><strong>{escape(str(item.get('class_name', '')))}</strong><br><code>{escape(str(item.get('case_id', '')))}</code></td>"
        f"<td><strong>{escape(str(item.get('variant', '')))}</strong><div class=\"queries\">{escape(', '.join(item.get('queries', [])))}</div></td>"
        f"<td>{item.get('detection_count', 0)} detections<br>{item.get('mask_count', 0)} masks<div class=\"media\">{boxes}{overlay}</div></td>"
        f"<td>{item.get('point_count', 0)} points<br>{item.get('voxel_count', 0)} voxels<br>"
        f"{incremental.get('processed_observation_count', 0)} observations / parity {escape(str(parity.get('equivalent', 'n/a')))}"
        f"<div class=\"media\">{png}</div></td>"
        f"<td>{html_link}<br>{_link(item.get('timing_json'), output_dir, 'timing JSON')}</td>"
        "</tr>"
    )


def _media(path: Any, output_dir: Path, label: str) -> str:
    if not path:
        return ""
    src = _relative(path, output_dir)
    return f'<a href="{escape(src)}"><img src="{escape(src)}" alt="{escape(label)}"></a>'


def _link(path: Any, output_dir: Path, label: str) -> str:
    if not path:
        return ""
    return f'<a href="{escape(_relative(path, output_dir))}">{escape(label)}</a>'


def _relative(path: Any, output_dir: Path) -> str:
    try:
        return os.path.relpath(Path(str(path)), output_dir).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return escape(str(value))


__all__ = ["build_profile_summary", "write_profile_report"]
