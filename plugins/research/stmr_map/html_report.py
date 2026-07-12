"""Static HTML report for STMR open-vocabulary perception and matrix output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .core import StmrMap
from .debug_payload import (
    projection_debug_summary,
    slim_custom_map_for_html,
)


def render_stmr_html(
    stmr_map: StmrMap,
    image_path: Path,
    output_dir: Path,
    *,
    landmarks: list[dict[str, Any]] | None = None,
    landmark_debug: dict[str, Any] | None = None,
    perception_results: list[dict[str, Any]] | None = None,
    custom_map_json: Path | None = None,
    points_preview: dict[str, Any] | None = None,
    pipeline_timing: dict[str, Any] | None = None,
) -> Path:
    """Render ``latest_stmr_map.html``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "latest_stmr_map.html"
    custom_map = _load_custom_map(custom_map_json)
    payload = {
        "image": image_path.name,
        "semantic_matrix": stmr_map.semantic_matrix,
        "semantic_legend": {str(key): dict(value) for key, value in sorted(stmr_map.semantic_legend.items())},
        "path_cells": stmr_map.path_cells,
        "current_cell": stmr_map.current_cell,
        "center_pose": stmr_map.center_pose,
        "map_config": stmr_map.map_config,
        "coordinate_frame": stmr_map.coordinate_frame,
        "projection_debug": projection_debug_summary(stmr_map.projection_debug),
        "source_frames": _frames_for_html(stmr_map.source_frames, output_dir),
        "landmarks": landmarks or [],
        "landmark_debug": landmark_debug or {},
        "perception_results": _perception_for_html(perception_results or [], output_dir),
        "custom_map": slim_custom_map_for_html(custom_map),
        "alignment_checks": _alignment_checks(stmr_map, custom_map),
        "points_preview": points_preview or {},
        "pipeline_timing": pipeline_timing or {},
    }
    html_path.write_text(_html(json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    return html_path


def _frames_for_html(frames: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(json.dumps(frames, ensure_ascii=False))
    for bundle in data.values():
        for frame in bundle.values():
            if isinstance(frame, dict) and frame.get("artifact_path"):
                frame["html_src"] = _rel(frame["artifact_path"], output_dir)
    return data


def _perception_for_html(items: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    data = json.loads(json.dumps(items, ensure_ascii=False))
    for item in data:
        for key in ("rgb_path", "mask_path", "overlay_path", "boxes_path"):
            if item.get(key):
                item[f"{key}_src"] = _rel(item[key], output_dir)
    return data


def _load_custom_map(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw.get("content", raw) if isinstance(raw, dict) else None


def _alignment_checks(stmr_map: StmrMap, custom_map: dict[str, Any] | None) -> dict[str, Any]:
    warnings = list(stmr_map.projection_debug.get("alignment_warnings", []))
    if custom_map is not None:
        custom_center = custom_map.get("center_pose") or {}
        if not _poses_close(stmr_map.center_pose, custom_center):
            warnings.append("custom_map_center_pose_differs_from_stmr_center_pose")
    semantic_counts = stmr_map.projection_debug.get("voxel_stats", {}).get("semantic_voxel_counts", {})
    positive_semantics = [key for key in semantic_counts if str(key) not in {"-2", "-1", "0"}]
    if len(positive_semantics) <= 1:
        warnings.append("stmr_sparse_open_vocabulary_map_single_positive_semantic")
    return {
        "status": "ok" if not warnings else "check_warnings",
        "warnings": sorted(set(warnings)),
        "custom_map_available": custom_map is not None,
    }


def _poses_close(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1e-3) -> bool:
    for key in ("x", "y", "z", "yaw_deg"):
        try:
            if abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) > tolerance:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _rel(path: str, output_dir: Path) -> str:
    try:
        return os.path.relpath(Path(path), output_dir).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _html(payload: str) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STMR Debug Viewer</title>
  <style>
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
      --soft: #eef2ff;
      --blue: #2563eb;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, sans-serif; }
    header { padding: 18px 24px; background: white; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 4px; font-size: 22px; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    h3 { margin: 0 0 8px; font-size: 13px; }
    main { padding: 18px 24px 28px; display: grid; grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr); gap: 16px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-width: 0; }
    .full { grid-column: 1 / -1; }
    img { width: 100%; max-height: 560px; object-fit: contain; border: 1px solid #e2e8f0; background: white; }
    .matrix { display: grid; gap: 1px; background: var(--line); aspect-ratio: 1 / 1; width: min(520px, 100%); border: 1px solid var(--line); }
    .cell { min-width: 0; min-height: 0; }
    canvas { width: 100%; height: 460px; display: block; background: #ffffff; border: 1px solid var(--line); cursor: grab; }
    canvas:active { cursor: grabbing; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: white; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
    .class-card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: white; cursor: pointer; }
    .class-card canvas { height: 170px; margin-top: 8px; cursor: default; }
    .mini-matrix { display: grid; gap: 1px; width: 120px; aspect-ratio: 1 / 1; margin-top: 8px; background: var(--line); border: 1px solid var(--line); }
    .filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .filter-btn { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); background: white; border-radius: 6px; padding: 6px 9px; cursor: pointer; }
    .filter-btn.active { border-color: var(--blue); background: var(--soft); }
    .legend-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; padding: 4px 6px; border-radius: 6px; cursor: pointer; }
    .legend-row:hover, .legend-row.active { background: var(--soft); }
    .swatch { width: 16px; height: 16px; border: 1px solid var(--line); flex: 0 0 auto; }
    .kv { display: grid; grid-template-columns: 180px 1fr; gap: 6px 12px; margin: 0; }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; word-break: break-word; }
    .muted { color: var(--muted); }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e2e8f0; color: #334155; }
    .warn { color: #991b1b; background: #fee2e2; padding: 8px; border-radius: 6px; }
    pre { overflow: auto; max-height: 280px; background: #f1f5f9; padding: 10px; border-radius: 6px; }
    code { background: var(--soft); padding: 1px 4px; border-radius: 4px; }
    .thumbs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
    .thumb img { height: 150px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } .kv { grid-template-columns: 130px 1fr; } canvas { height: 360px; } }
  </style>
</head>
<body>
  <header>
    <h1>STMR Debug Viewer</h1>
    <div class="muted">2D top-down STMR inspection: GroundingDINO/SAM masks -> voxel top-height projection -> semantic matrix.</div>
  </header>
  <main>
    <section class="full">
      <h2>Semantic Filter</h2>
      <div id="semanticSelector" class="filters"></div>
      <dl id="selectionDetails" class="kv"></dl>
    </section>
    <section>
      <h2>STMR Top-down PNG</h2>
      <img id="stmrImage" alt="STMR map">
    </section>
    <section>
      <h2>STMR Matrix</h2>
      <div id="matrix" class="matrix"></div>
    </section>
    <section class="full">
      <h2>STMR Lightweight 3D Preview</h2>
      <canvas id="pointCloud" width="1100" height="460"></canvas>
      <div id="pointStats" class="muted" style="margin-top:8px"></div>
    </section>
    <section>
      <h2>Legend</h2>
      <div id="legend"></div>
    </section>
    <section>
      <h2>Map Summary</h2>
      <dl id="mapSummary" class="kv"></dl>
    </section>
    <section class="full">
      <h2>Pipeline Timing</h2>
      <dl id="pipelineTiming" class="kv"></dl>
    </section>
    <section class="full">
      <h2>Semantic Class Gallery</h2>
      <div id="classGallery" class="gallery"></div>
    </section>
    <section>
      <h2>Landmarks</h2>
      <pre id="landmarks"></pre>
    </section>
    <section>
      <h2>Landmark Extractor Debug</h2>
      <pre id="landmarkDebug"></pre>
    </section>
    <section class="full">
      <h2>Coordinate Frame</h2>
      <div id="alignmentWarnings"></div>
      <dl id="coordinateFrame" class="kv"></dl>
    </section>
    <section class="full">
      <h2>GroundingDINO + SAM Results</h2>
      <div id="perception" class="cards"></div>
    </section>
    <section class="full">
      <h2>Dynamic Target Candidates</h2>
      <div id="dynamicCandidates" class="cards"></div>
    </section>
    <section class="full">
      <h2>Source Frames</h2>
      <div id="sources" class="cards"></div>
    </section>
    <section class="full">
      <h2>Projection Debug</h2>
      <dl id="debug" class="kv"></dl>
    </section>
  </main>
  <script id="STMR_DATA" type="application/json">__STMR_PAYLOAD__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('STMR_DATA').textContent);
    let selectedSemanticId = null;
    document.getElementById('stmrImage').src = DATA.image;
    function color(id) {
      const item = DATA.semantic_legend[String(id)] || {};
      if (Array.isArray(item.color)) return `rgb(${item.color[0]}, ${item.color[1]}, ${item.color[2]})`;
      if (id === -1) return '#2563eb';
      if (id === -2) return '#94a3b8';
      if (id === 0) return '#ffffff';
      return '#dbeafe';
    }
    function label(id) {
      return (DATA.semantic_legend[String(id)] || {}).label || `semantic_${id}`;
    }
    function esc(value) {
      const span = document.createElement('span');
      span.textContent = value == null ? '' : String(value);
      return span.innerHTML;
    }
    function matrixCounts() {
      const counts = {};
      (DATA.semantic_matrix || []).forEach(row => row.forEach(value => counts[value] = (counts[value] || 0) + 1));
      return counts;
    }
    const matrix = DATA.semantic_matrix || [];
    const counts = matrixCounts();
    const debug = DATA.projection_debug || {};
    const preview = DATA.points_preview || {};
    const previewCounts = preview.counts || {};
    const previewBudgets = preview.budgets || {};
    const stmrPoints = (preview.stmr_points || []).map(toLocalPoint);
    const stmrVoxels = (preview.stmr_voxels || []).map(toLocalPoint);
    let rotX = -0.82;
    let rotZ = 0.72;
    let zoom = 14;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    function rgbArray(id) {
      const item = DATA.semantic_legend[String(id)] || {};
      if (Array.isArray(item.color)) return item.color;
      if (id === -1) return [37, 99, 235];
      if (id === -2) return [148, 163, 184];
      if (id === 0) return [255, 255, 255];
      return [219, 234, 254];
    }
    function toLocalPoint(point) {
      return {
        x: Number(point.local_right_m || 0),
        y: Number(point.local_up_m || 0),
        z: Number(point.local_forward_m || 0),
        semantic_id: Number(point.semantic_id || 0),
        distance_confidence: Number(point.distance_confidence == null ? 1 : point.distance_confidence),
        point_count: Number(point.point_count || 1),
      };
    }
    function semanticPointCount(id) {
      return stmrPoints.filter(point => point.semantic_id === id).length + stmrVoxels.filter(point => point.semantic_id === id).length;
    }
    function semanticClasses() {
      return Object.entries(DATA.semantic_legend || {})
        .map(([id, item]) => ({
          semantic_id: Number(id),
          label: item.label || label(Number(id)),
          point_count: semanticPointCount(Number(id)),
        }))
        .filter(item => item.semantic_id !== 0)
        .sort((a, b) => Number(b.point_count || 0) - Number(a.point_count || 0));
    }
    function selectedPoints() {
      const points = selectedSemanticId == null ? stmrPoints : stmrPoints.filter(point => point.semantic_id === selectedSemanticId);
      const voxels = selectedSemanticId == null ? stmrVoxels : stmrVoxels.filter(point => point.semantic_id === selectedSemanticId);
      return {points, voxels};
    }
    function project(canvas, x, y, z, localRotX, localRotZ, localZoom) {
      const cosZ = Math.cos(localRotZ);
      const sinZ = Math.sin(localRotZ);
      const cosX = Math.cos(localRotX);
      const sinX = Math.sin(localRotX);
      const x1 = x * cosZ - z * sinZ;
      const z1 = x * sinZ + z * cosZ;
      const y1 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const scale = localZoom * (900 / (900 + z2));
      return {
        x: canvas.width / 2 + x1 * scale,
        y: canvas.height / 2 - y1 * scale,
        z: z2,
      };
    }
    function drawAxes(canvas, ctx, localRotX, localRotZ, localZoom) {
      const origin = project(canvas, 0, 0, 0, localRotX, localRotZ, localZoom);
      const right = project(canvas, 5, 0, 0, localRotX, localRotZ, localZoom);
      const up = project(canvas, 0, 5, 0, localRotX, localRotZ, localZoom);
      const forward = project(canvas, 0, 0, 5, localRotX, localRotZ, localZoom);
      [[right, '#dc2626', 'right'], [up, '#16a34a', 'up'], [forward, '#2563eb', 'forward']].forEach(([p, axisColor, axisLabel]) => {
        ctx.strokeStyle = axisColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.fillStyle = axisColor;
        ctx.fillText(axisLabel, p.x + 4, p.y + 4);
      });
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    function drawCloudToCanvas(canvas, cloudPoints, cloudVoxels, options = {}) {
      const ctx = canvas.getContext('2d');
      const localRotX = options.rotX ?? rotX;
      const localRotZ = options.rotZ ?? rotZ;
      const localZoom = options.zoom ?? zoom;
      const pointRadius = options.pointRadius ?? 2.4;
      const voxelRadius = options.voxelRadius ?? 4.2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = '12px system-ui';
      const projectedPoints = cloudPoints.map(point => ({...project(canvas, point.x, point.y, point.z, localRotX, localRotZ, localZoom), point, layer: 'point'}));
      const projectedVoxels = cloudVoxels.map(point => ({...project(canvas, point.x, point.y, point.z, localRotX, localRotZ, localZoom), point, layer: 'voxel'}));
      const projected = [...projectedPoints, ...projectedVoxels].sort((a, b) => a.z - b.z);
      if (!projected.length) {
        ctx.fillStyle = '#64748b';
        ctx.fillText('No STMR preview points available for this semantic filter.', 18, 28);
        drawAxes(canvas, ctx, localRotX, localRotZ, localZoom);
        return;
      }
      projected.forEach(item => {
        const raw = rgbArray(item.point.semantic_id);
        const alpha = Math.max(0.2, Math.min(1, item.point.distance_confidence || 1));
        ctx.fillStyle = `rgba(${raw[0]}, ${raw[1]}, ${raw[2]}, ${alpha})`;
        ctx.strokeStyle = `rgba(${raw[0]}, ${raw[1]}, ${raw[2]}, 0.95)`;
        ctx.beginPath();
        ctx.arc(item.x, item.y, item.layer === 'voxel' ? voxelRadius : pointRadius, 0, Math.PI * 2);
        if (item.layer === 'voxel') {
          ctx.stroke();
        } else {
          ctx.fill();
        }
      });
      drawAxes(canvas, ctx, localRotX, localRotZ, localZoom);
    }
    function drawPointCloud() {
      const selected = selectedPoints();
      drawCloudToCanvas(document.getElementById('pointCloud'), selected.points, selected.voxels);
      const totalPoints = Number(previewCounts.stmr_points_total || 0);
      const totalVoxels = Number(previewCounts.stmr_voxels_total || 0);
      document.getElementById('pointStats').textContent =
        `${selected.points.length}/${totalPoints} STMR points, ${selected.voxels.length}/${totalVoxels} voxels displayed` +
        ` (budgets p=${previewBudgets.stmr_point_budget ?? 0}, v=${previewBudgets.stmr_voxel_budget ?? 0}; custom preview disabled by default)`;
    }
    function renderMiniMatrix(root, semanticId) {
      root.innerHTML = '';
      root.style.gridTemplateColumns = `repeat(${matrix.length || 1}, 1fr)`;
      matrix.forEach(row => row.forEach(value => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.style.background = value === semanticId ? color(value) : '#e2e8f0';
        cell.style.opacity = value === semanticId ? '1' : '0.28';
        root.appendChild(cell);
      }));
    }
    function renderClassGallery() {
      const root = document.getElementById('classGallery');
      root.innerHTML = '';
      const visibleClasses = semanticClasses().filter(item => item.point_count > 0).slice(0, 24);
      if (!visibleClasses.length) {
        root.innerHTML = '<div class="muted">No STMR preview points are available.</div>';
        return;
      }
      visibleClasses.forEach(item => {
        const semanticId = Number(item.semantic_id);
        const card = document.createElement('article');
        card.className = 'class-card';
        card.innerHTML = `<h3><span class="swatch" style="background:${color(semanticId)}"></span> ${esc(item.label)}</h3>
          <div class="muted">${item.point_count} preview points/voxels · id ${semanticId}</div>
          <canvas width="300" height="170"></canvas>
          <div class="mini-matrix"></div>`;
        card.addEventListener('click', () => selectSemantic(semanticId));
        root.appendChild(card);
        const miniPoints = stmrPoints.filter(point => point.semantic_id === semanticId);
        const miniVoxels = stmrVoxels.filter(point => point.semantic_id === semanticId);
        drawCloudToCanvas(card.querySelector('canvas'), miniPoints, miniVoxels, {rotX: -0.82, rotZ: 0.72, zoom: 16, pointRadius: 2.1, voxelRadius: 3.4});
        renderMiniMatrix(card.querySelector('.mini-matrix'), semanticId);
      });
    }
    function renderMatrix() {
      const root = document.getElementById('matrix');
      root.innerHTML = '';
      root.style.gridTemplateColumns = `repeat(${matrix.length || 1}, 1fr)`;
      matrix.forEach(row => row.forEach(value => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.style.background = color(value);
        cell.title = `${value}: ${label(value)}`;
        if (selectedSemanticId != null && value !== selectedSemanticId) {
          cell.style.opacity = value === 0 ? '0.25' : '0.18';
          cell.style.filter = 'grayscale(1)';
        }
        if (selectedSemanticId != null && value === selectedSemanticId) {
          cell.style.outline = '1px solid #0f172a';
          cell.style.zIndex = 1;
        }
        root.appendChild(cell);
      }));
    }
    function renderSelector() {
      const root = document.getElementById('semanticSelector');
      root.innerHTML = '';
      const all = document.createElement('button');
      all.className = `filter-btn ${selectedSemanticId == null ? 'active' : ''}`;
      all.innerHTML = `<span class="swatch" style="background:#0f172a"></span><span>All</span><span class="pill">${matrix.length * (matrix[0] || []).length} cells</span>`;
      all.addEventListener('click', () => selectSemantic(null));
      root.appendChild(all);
      Object.entries(DATA.semantic_legend || {}).forEach(([id, item]) => {
        const semanticId = Number(id);
        const button = document.createElement('button');
        button.className = `filter-btn ${selectedSemanticId === semanticId ? 'active' : ''}`;
        button.innerHTML = `<span class="swatch" style="background:${color(semanticId)}"></span><span>${esc(item.label || label(semanticId))}</span><span class="pill">${counts[semanticId] || 0} cells</span>`;
        button.addEventListener('click', () => selectSemantic(semanticId));
        root.appendChild(button);
      });
    }
    function renderLegend() {
      const root = document.getElementById('legend');
      root.innerHTML = '';
      Object.entries(DATA.semantic_legend || {}).forEach(([id, item]) => {
        const semanticId = Number(id);
        const row = document.createElement('div');
        row.className = `legend-row ${selectedSemanticId === semanticId ? 'active' : ''}`;
        row.innerHTML = `<span class="swatch" style="background:${color(semanticId)}"></span><code>${id}</code><span>${esc(item.label || 'unknown')}</span><span class="pill">${counts[semanticId] || 0} cells</span>`;
        row.addEventListener('click', () => selectSemantic(semanticId));
        root.appendChild(row);
      });
    }
    function renderKv(id, entries) {
      const root = document.getElementById(id);
      root.innerHTML = '';
      entries.forEach(([key, value]) => {
        const dt = document.createElement('dt');
        const dd = document.createElement('dd');
        dt.textContent = key;
        dd.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value);
        root.appendChild(dt);
        root.appendChild(dd);
      });
    }
    function selectSemantic(id) {
      selectedSemanticId = id;
      renderSelector();
      renderMatrix();
      renderLegend();
      renderSelectionDetails();
      drawPointCloud();
    }
    function renderSelectionDetails() {
      if (selectedSemanticId == null) {
        renderKv('selectionDetails', [
          ['selected', 'All semantic classes'],
          ['positive_semantics', Object.keys(DATA.semantic_legend || {}).filter(id => Number(id) > 0).length],
          ['source_observations', debug.source_observation_count || 0],
        ]);
        return;
      }
      const item = DATA.semantic_legend[String(selectedSemanticId)] || {};
      renderKv('selectionDetails', [
        ['semantic_id', selectedSemanticId],
        ['label', item.label || 'unknown'],
        ['cell_count', counts[selectedSemanticId] || 0],
        ['source', item.source || 'unknown'],
        ['object_name', item.object_name || 'unknown'],
        ['confidence', item.confidence || 'unknown'],
      ]);
    }
    document.getElementById('landmarks').textContent = JSON.stringify(DATA.landmarks, null, 2);
    document.getElementById('landmarkDebug').textContent = JSON.stringify(DATA.landmark_debug || {}, null, 2);
    renderKv('mapSummary', [
      ['projection_status', debug.projection_status || DATA.map_config.semantic_projection],
      ['point_count', debug.point_count || 0],
      ['voxel_count', debug.voxel_count || 0],
      ['source_observation_count', debug.source_observation_count || 0],
      ['subgoal_priority_hits', (debug.subgoal_priority_hits || []).length],
      ['current_cell', DATA.current_cell || {}],
      ['path_cells', (DATA.path_cells || []).length],
    ]);
    const timing = DATA.pipeline_timing || {};
    const timingStages = timing.stages || {};
    renderKv('pipelineTiming', [
      ['sample_count', timing.sample_count || 0],
      ['stage_count', timing.stage_count || 0],
      ...Object.entries(timingStages).map(([name, item]) => [name, {
        samples: item.sample_count,
        p50_ms: item.p50_ms,
        p95_ms: item.p95_ms,
        cuda_peak_mb: item.cuda_peak_allocated_mb,
        cpu_rss_peak_mb: item.cpu_rss_peak_mb,
      }]),
    ]);
    renderKv('coordinateFrame', [
      ['center_pose', DATA.center_pose || {}],
      ['world_frame', DATA.coordinate_frame.world_frame || 'unknown'],
      ['local_frame', DATA.coordinate_frame.local_frame || 'unknown'],
      ['grid_size', DATA.map_config.grid_size],
      ['meters_per_cell', DATA.map_config.meters_per_cell],
      ['alignment_checks', DATA.alignment_checks || {}],
    ]);
    renderKv('debug', [
      ['voxel_stats', debug.voxel_stats || {}],
      ['semantic_cell_counts', debug.semantic_cell_counts || {}],
      ['depth_interpretation_used', debug.depth_interpretation_used || 'unknown'],
      ['depth_interpretation_ranges', (debug.depth_interpretation_comparison || {}).ranges || {}],
      ['multi_frame_projection_drift', debug.multi_frame_projection_drift || {}],
      ['frame_point_ranges', debug.frame_point_ranges || []],
      ['stride', debug.stride || 'unknown'],
      ['max_depth_m', debug.max_depth_m || 'unknown'],
    ]);
    const warningRoot = document.getElementById('alignmentWarnings');
    const warnings = (DATA.alignment_checks && DATA.alignment_checks.warnings) || [];
    if (warnings.length) {
      warningRoot.innerHTML = `<div class="warn">${warnings.join('<br>')}</div>`;
    } else {
      warningRoot.innerHTML = '<span class="pill">coordinate checks passed</span>';
    }
    const perception = document.getElementById('perception');
    (DATA.perception_results || []).forEach(item => {
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `<h3>${item.observation_id}</h3>
        <div class="muted">${(item.kept_masks || []).length} kept masks / ${(item.rejected_masks || []).length} rejected / ${(item.kept_non_primary_masks || []).length} non-primary</div>
        ${item.primary_dynamic_track ? `<div class="pill">primary: ${item.primary_dynamic_track.label} (${item.primary_dynamic_track.candidate_count} masks)</div>` : ''}
        ${item.boxes_path_src ? `<h4>Boxes</h4><img src="${item.boxes_path_src}">` : ''}
        ${item.overlay_path_src ? `<h4>Overlay</h4><img src="${item.overlay_path_src}">` : ''}
        <h4>TF-IDF Matches</h4><pre>${JSON.stringify(item.tfidf_matches || [], null, 2)}</pre>
        <h4>Rejected Dynamic Masks</h4><pre>${JSON.stringify(item.rejected_masks || [], null, 2)}</pre>`;
      perception.appendChild(card);
    });
    const dynamicRoot = document.getElementById('dynamicCandidates');
    const dynamicItems = (DATA.perception_results || []).filter(item => (item.dynamic_target_candidates || []).length);
    if (!dynamicItems.length) {
      dynamicRoot.innerHTML = '<div class="muted">No dynamic target candidates were recorded.</div>';
    }
    dynamicItems.forEach(item => {
      const card = document.createElement('article');
      card.className = 'card';
      const statusCounts = {};
      (item.dynamic_target_candidates || []).forEach(candidate => {
        const status = candidate.status || 'unknown';
        statusCounts[status] = (statusCounts[status] || 0) + 1;
      });
      card.innerHTML = `<h3>${item.observation_id}</h3>
        <div class="muted">${JSON.stringify(statusCounts)}</div>
        <h4>Primary Track</h4><pre>${JSON.stringify(item.primary_dynamic_track || null, null, 2)}</pre>
        <h4>Candidates</h4><pre>${JSON.stringify(item.dynamic_target_candidates || [], null, 2)}</pre>`;
      dynamicRoot.appendChild(card);
    });
    const sources = document.getElementById('sources');
    Object.entries(DATA.source_frames || {}).forEach(([obs, bundle]) => {
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `<h3>${obs}</h3>` + Object.entries(bundle).map(([name, frame]) => {
        const img = frame.html_src && (name === 'rgb' || name === 'stmr_open_vocab_mask') ? `<img src="${frame.html_src}">` : '';
        return `<div><code>${name}</code><div class="muted">${frame.artifact_path || ''}</div>${img}</div>`;
      }).join('');
      sources.appendChild(card);
    });
    renderClassGallery();
    selectSemantic(null);
    const pointCloudCanvas = document.getElementById('pointCloud');
    pointCloudCanvas.addEventListener('mousedown', event => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
    });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', event => {
      if (!dragging) return;
      rotZ += (event.clientX - lastX) * 0.01;
      rotX += (event.clientY - lastY) * 0.01;
      lastX = event.clientX;
      lastY = event.clientY;
      drawPointCloud();
    });
    pointCloudCanvas.addEventListener('wheel', event => {
      event.preventDefault();
      zoom = Math.max(4, Math.min(80, zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
      drawPointCloud();
    }, {passive: false});
  </script>
</body>
</html>
"""
    return template.replace("__STMR_PAYLOAD__", payload)
