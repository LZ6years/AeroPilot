"""Static HTML debug report for custom_map artifacts.

The report is a human-facing artifact only. It keeps all data local and uses
native canvas rendering so a run can be inspected without a dev server or CDN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schemas import GridMap


def render_custom_map_html(grid_map: GridMap, image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "latest_custom_map.html"
    data = {
        "image": image_path.name,
        "semantic_matrix": grid_map.semantic_matrix,
        "semantic_legend": _string_keyed_legend(grid_map.semantic_legend),
        "path_summary": grid_map.path_summary,
        "center_pose": grid_map.center_pose,
        "path_cells": grid_map.path_cells,
        "current_cell": grid_map.current_cell,
        "map_config": grid_map.map_config,
        "projection_debug": _projection_debug_for_html(grid_map.projection_debug, output_dir),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html_path.write_text(_html(payload), encoding="utf-8")
    return html_path


def _string_keyed_legend(legend: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in sorted(legend.items())}


def _projection_debug_for_html(projection_debug: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    debug = json.loads(json.dumps(projection_debug, ensure_ascii=False))
    frames_by_observation = debug.get("source_frames_by_observation_id")
    if not isinstance(frames_by_observation, dict):
        return debug
    for frames in frames_by_observation.values():
        if not isinstance(frames, dict):
            continue
        for frame in frames.values():
            if not isinstance(frame, dict):
                continue
            artifact_path = frame.get("artifact_path")
            if artifact_path:
                frame["html_src"] = _relative_html_src(artifact_path, output_dir)
            if frame.get("modality", "").lower() in {"depth", "depth_perspective"} and artifact_path:
                heatmap_path = _write_depth_heatmap(artifact_path, output_dir)
                if heatmap_path is not None:
                    frame["depth_heatmap_src"] = _relative_html_src(str(heatmap_path), output_dir)
    return debug


def _write_depth_heatmap(artifact_path: str, output_dir: Path) -> Path | None:
    path = Path(artifact_path)
    candidate = path if path.is_absolute() else Path.cwd() / path
    if not candidate.exists() or candidate.suffix.lower() != ".npy":
        return None
    try:
        depth = np.asarray(np.load(candidate), dtype=np.float32)
    except (OSError, ValueError):
        return None
    if depth.ndim != 2:
        return None
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return None
    near = float(np.percentile(valid, 2))
    far = float(np.percentile(valid, 95))
    if far <= near:
        far = near + 1.0
    normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
    invalid = ~np.isfinite(depth) | (depth <= 0)
    red = (255 * normalized).astype(np.uint8)
    green = (255 * (1.0 - np.abs(normalized - 0.5) * 2.0)).astype(np.uint8)
    blue = (255 * (1.0 - normalized)).astype(np.uint8)
    image = np.stack([red, green, blue], axis=-1)
    image[invalid] = np.array([15, 23, 42], dtype=np.uint8)
    heatmap_dir = output_dir / "depth_heatmap"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = heatmap_dir / f"{candidate.stem}_heatmap.png"
    Image.fromarray(image, mode="RGB").save(heatmap_path)
    return heatmap_path


def _relative_html_src(artifact_path: str, output_dir: Path) -> str:
    raw = Path(artifact_path)
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    try:
        relative = os.path.relpath(candidate, output_dir)
    except ValueError:
        return str(raw).replace("\\", "/")
    return relative.replace("\\", "/")


def _html(payload: str) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Custom Map Debug Viewer</title>
  <style>
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --line: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #2563eb;
      --green: #16a34a;
      --soft: #eef2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0 0 4px; font-size: 22px; }
    h2 { margin: 0 0 12px; font-size: 15px; }
    h3 { margin: 0; font-size: 13px; }
    main { padding: 18px 24px 28px; }
    .grid {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .map-img {
      width: 100%;
      max-height: 540px;
      object-fit: contain;
      border: 1px solid var(--line);
      background: #fff;
    }
    canvas {
      width: 100%;
      height: 460px;
      display: block;
      background: #ffffff;
      border: 1px solid var(--line);
      cursor: grab;
    }
    canvas:active { cursor: grabbing; }
    .matrix {
      display: grid;
      gap: 1px;
      width: min(460px, 100%);
      aspect-ratio: 1 / 1;
      background: var(--line);
      border: 1px solid var(--line);
    }
    .cell { min-width: 0; min-height: 0; }
    .kv {
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 6px 12px;
      margin: 0;
    }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; word-break: break-word; }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      padding: 4px 6px;
      border-radius: 6px;
      cursor: pointer;
    }
    .legend-row:hover, .legend-row.active { background: var(--soft); }
    .swatch {
      width: 16px;
      height: 16px;
      border: 1px solid var(--line);
      flex: 0 0 auto;
    }
    .muted { color: var(--muted); }
    .full { grid-column: 1 / -1; }
    code {
      background: var(--soft);
      padding: 1px 4px;
      border-radius: 4px;
    }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .filter-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 9px;
      cursor: pointer;
      max-width: 280px;
    }
    .filter-btn.active {
      border-color: var(--blue);
      background: var(--soft);
    }
    .badge {
      color: var(--muted);
      font-size: 12px;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 12px;
    }
    .class-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .class-card canvas {
      height: 170px;
      margin-top: 8px;
      cursor: default;
    }
    .mini-matrix {
      display: grid;
      gap: 1px;
      width: 120px;
      aspect-ratio: 1 / 1;
      margin-top: 8px;
      background: var(--line);
      border: 1px solid var(--line);
    }
    .source-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }
    .source-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .thumbs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .thumb img {
      width: 100%;
      height: 130px;
      object-fit: contain;
      border: 1px solid var(--line);
      background: #fff;
    }
    .thumb-title {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      canvas { height: 360px; }
      .kv { grid-template-columns: 130px 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Custom Map Debug Viewer</h1>
    <div class="muted">Standalone visual check for top-down map, semantic matrix, and depth-mask 3D projection.</div>
  </header>
  <main class="grid">
    <section class="panel full">
      <h2>Semantic Filter</h2>
      <div id="semanticClassSelector" class="filters"></div>
      <dl id="classDetails" class="kv"></dl>
    </section>
    <section class="panel">
      <h2>2D Top-down Map</h2>
      <img id="mapImage" class="map-img" alt="Custom map">
    </section>
    <section class="panel">
      <h2>3D Semantic Point Cloud</h2>
      <canvas id="pointCloud" width="760" height="460"></canvas>
      <div class="muted">Drag to rotate, wheel to zoom. Coordinates are UAV-local: right / up / forward.</div>
    </section>
    <section class="panel full">
      <h2>Depth Projection Diagnostics</h2>
      <canvas id="depthModeCompare" width="1100" height="420"></canvas>
      <dl id="depthModeDebug" class="kv" style="margin-top:12px"></dl>
    </section>
    <section class="panel">
      <h2>20x20 Semantic Matrix</h2>
      <div id="matrix" class="matrix"></div>
    </section>
    <section class="panel">
      <h2>Legend + Path Summary</h2>
      <div id="legend"></div>
      <hr>
      <dl id="summary" class="kv"></dl>
    </section>
    <section class="panel full">
      <h2>Semantic Class Gallery</h2>
      <div id="classGallery" class="gallery"></div>
    </section>
    <section class="panel full">
      <h2>Source Frames</h2>
      <div id="sourceFrames" class="source-grid"></div>
    </section>
    <section class="panel full">
      <h2>Projection Debug</h2>
      <dl id="debug" class="kv"></dl>
    </section>
  </main>
  <script id="CUSTOM_MAP_DATA" type="application/json">__CUSTOM_MAP_PAYLOAD__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('CUSTOM_MAP_DATA').textContent);
    const debug = DATA.projection_debug || {};
    const COLORS = {
      0: [255, 255, 255],
      path: [37, 99, 235],
      uav: [22, 163, 74]
    };
    let selectedSemanticId = null;
    let rotX = -0.75;
    let rotZ = 0.65;
    let zoom = 28;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function legendColor(id) {
      const item = DATA.semantic_legend[String(id)] || {};
      if (Array.isArray(item.color) && item.color.length >= 3) return item.color.slice(0, 3);
      const palette = [[254,226,226],[219,234,254],[220,252,231],[254,249,195],[237,233,254],[255,237,213]];
      return id === 0 ? COLORS[0] : palette[Math.abs(Number(id)) % palette.length];
    }
    function rgb(arr) { return `rgb(${arr[0]}, ${arr[1]}, ${arr[2]})`; }
    function escapeText(value) {
      const span = document.createElement('span');
      span.textContent = value == null ? '' : String(value);
      return span.innerHTML;
    }
    function classLabel(id) {
      const item = DATA.semantic_legend[String(id)] || {};
      return item.label || `semantic_${id}`;
    }
    function pointCountFor(id) {
      const counts = debug.semantic_point_counts || {};
      return Number(counts[String(id)] || 0);
    }
    function sourceObservationIdsFor(id) {
      const sourceMap = debug.semantic_source_observations || {};
      return sourceMap[String(id)] || [];
    }
    function rawExamplesFor(id, item = {}) {
      const fromItem = Array.isArray(item.raw_label_top_examples) ? item.raw_label_top_examples : [];
      if (fromItem.length) return fromItem;
      const examples = debug.semantic_raw_label_examples || {};
      return examples[String(id)] || [];
    }
    function rawExampleText(id, item = {}) {
      const examples = rawExamplesFor(id, item).slice(0, 5);
      if (!examples.length) return '';
      return examples.map(entry => `${entry.label}:${entry.pixel_count}`).join(', ');
    }
    function pointRatioFor(id) {
      const ratios = debug.semantic_point_ratios || {};
      return Number(ratios[String(id)] || 0);
    }
    function semanticClasses() {
      const fromDebug = Array.isArray(debug.semantic_classes) ? debug.semantic_classes : [];
      if (fromDebug.length) return fromDebug;
      return Object.entries(DATA.semantic_legend || {})
        .filter(([id]) => Number(id) !== 0)
        .map(([id, item]) => ({
          semantic_id: Number(id),
          label: item.label || `semantic_${id}`,
          color: item.color,
          source: item.source,
          object_name: item.object_name,
          confidence: item.confidence,
          point_count: pointCountFor(Number(id)),
          source_observation_ids: sourceObservationIdsFor(Number(id))
        }));
    }
    const classes = semanticClasses();
    const center = DATA.center_pose || {};
    const centerX = Number(center.x || 0);
    const centerY = Number(center.y || 0);
    const centerZ = Number(center.z || 0);
    function localFromWorld(point) {
      const yaw = ((center.yaw_deg || 0) * Math.PI) / 180;
      const dx = Number(point.world_x || 0) - centerX;
      const dy = Number(point.world_y || 0) - centerY;
      return {
        local_forward_m: Math.cos(yaw) * dx + Math.sin(yaw) * dy,
        local_right_m: -Math.sin(yaw) * dx + Math.cos(yaw) * dy,
        local_up_m: Number(point.world_z || 0) - centerZ
      };
    }
    const points = (debug.sampled_points || []).map(p => {
      const fallback = localFromWorld(p);
      return {
      x: Number(p.local_right_m ?? fallback.local_right_m),
      y: Number(p.local_up_m ?? fallback.local_up_m),
      z: Number(p.local_forward_m ?? fallback.local_forward_m),
      world_x: Number(p.world_x || 0),
      world_y: Number(p.world_y || 0),
      world_z: Number(p.world_z || 0),
      pixel_u: Number(p.pixel_u || 0),
      pixel_v: Number(p.pixel_v || 0),
      semantic_id: Number(p.semantic_id || 0),
      source_observation_id: p.source_observation_id || '',
      distance_m: Number(p.distance_m || 0),
      distance_confidence: Number(p.distance_confidence == null ? 1 : p.distance_confidence)
      };
    });

    document.getElementById('mapImage').src = DATA.image;

    function selectedPoints() {
      if (selectedSemanticId == null) return points;
      return points.filter(point => point.semantic_id === selectedSemanticId);
    }
    function renderMatrix() {
      const matrix = DATA.semantic_matrix || [];
      const container = document.getElementById('matrix');
      container.style.gridTemplateColumns = `repeat(${matrix.length || 1}, 1fr)`;
      container.innerHTML = '';
      matrix.forEach(row => row.forEach(value => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.title = `${value}: ${classLabel(value)}`;
        cell.style.background = rgb(legendColor(value));
        if (selectedSemanticId != null && value !== selectedSemanticId) {
          cell.style.opacity = value === 0 ? '0.24' : '0.18';
          cell.style.filter = 'grayscale(1)';
        }
        if (selectedSemanticId != null && value === selectedSemanticId) {
          cell.style.outline = '1px solid #0f172a';
          cell.style.zIndex = 1;
        }
        container.appendChild(cell);
      }));
    }
    function renderLegend() {
      const root = document.getElementById('legend');
      root.innerHTML = '';
      Object.entries(DATA.semantic_legend || {}).forEach(([id, item]) => {
        const semanticId = Number(id);
        const row = document.createElement('div');
        row.className = `legend-row ${selectedSemanticId === semanticId ? 'active' : ''}`;
        const examples = rawExampleText(semanticId, item);
        row.title = examples ? `raw labels: ${examples}` : '';
        row.innerHTML = `<span class="swatch" style="background:${rgb(legendColor(semanticId))}"></span><code>${id}</code><span>${escapeText(item.label || 'unknown')}</span><span class="badge">${pointCountFor(semanticId)} pts · ${(pointRatioFor(semanticId) * 100).toFixed(2)}%</span>${examples ? `<span class="badge">${escapeText(examples)}</span>` : ''}`;
        row.addEventListener('click', () => selectSemanticClass(semanticId === 0 ? null : semanticId));
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
    function renderClassSelector() {
      const root = document.getElementById('semanticClassSelector');
      root.innerHTML = '';
      const all = document.createElement('button');
      all.className = `filter-btn ${selectedSemanticId == null ? 'active' : ''}`;
      all.innerHTML = `<span class="swatch" style="background:${rgb([15,23,42])}"></span><span>All</span><span class="badge">${points.length} pts</span>`;
      all.addEventListener('click', () => selectSemanticClass(null));
      root.appendChild(all);
      classes.forEach(item => {
        const semanticId = Number(item.semantic_id);
        const button = document.createElement('button');
        button.className = `filter-btn ${selectedSemanticId === semanticId ? 'active' : ''}`;
        button.innerHTML = `<span class="swatch" style="background:${rgb(legendColor(semanticId))}"></span><span>${escapeText(item.label || classLabel(semanticId))}</span><span class="badge">${item.point_count || pointCountFor(semanticId)} pts</span>`;
        button.addEventListener('click', () => selectSemanticClass(semanticId));
        root.appendChild(button);
      });
    }
    function renderClassDetails() {
      if (selectedSemanticId == null) {
        renderKv('classDetails', [
          ['selected', 'All semantic classes'],
          ['point_count', points.length],
          ['semantic_class_count', classes.length],
          ['source_observations', debug.source_observation_count || 0]
        ]);
        return;
      }
      const item = classes.find(entry => Number(entry.semantic_id) === selectedSemanticId) || {};
      const legend = DATA.semantic_legend[String(selectedSemanticId)] || {};
      renderKv('classDetails', [
        ['semantic_id', selectedSemanticId],
        ['label', item.label || legend.label || 'unknown'],
        ['object_name', item.object_name || legend.object_name || 'unknown'],
        ['source', item.source || legend.source || 'unknown'],
        ['confidence', item.confidence || legend.confidence || 'unknown'],
        ['point_count', item.point_count || pointCountFor(selectedSemanticId)],
        ['point_ratio', pointRatioFor(selectedSemanticId)],
        ['avg_distance_m', item.avg_distance_m ?? 'unknown'],
        ['avg_distance_confidence', item.avg_distance_confidence ?? 'unknown'],
        ['max_distance_confidence', item.max_distance_confidence ?? 'unknown'],
        ['far_point_ratio', item.far_point_ratio ?? 'unknown'],
        ['raw_label_top_examples', rawExamplesFor(selectedSemanticId, item)],
        ['source_observation_ids', item.source_observation_ids || sourceObservationIdsFor(selectedSemanticId)]
      ]);
    }

    function project(canvas, x, y, z, localRotX = rotX, localRotZ = rotZ, localZoom = zoom) {
      const cz = Math.cos(localRotZ), sz = Math.sin(localRotZ);
      const cx = Math.cos(localRotX), sx = Math.sin(localRotX);
      const x1 = x * cz - y * sz;
      const y1 = x * sz + y * cz;
      const z1 = z;
      const y2 = y1 * cx - z1 * sx;
      const z2 = y1 * sx + z1 * cx;
      return {
        x: canvas.width / 2 + x1 * localZoom,
        y: canvas.height / 2 - y2 * localZoom,
        z: z2
      };
    }
    function drawAxes(canvas, ctx, localRotX = rotX, localRotZ = rotZ, localZoom = zoom) {
      const origin = project(canvas, 0, 0, 0, localRotX, localRotZ, localZoom);
      const right = project(canvas, 5, 0, 0, localRotX, localRotZ, localZoom);
      const up = project(canvas, 0, 5, 0, localRotX, localRotZ, localZoom);
      const forward = project(canvas, 0, 0, 5, localRotX, localRotZ, localZoom);
      [[right, '#dc2626', 'right'], [up, '#16a34a', 'up'], [forward, '#2563eb', 'forward']].forEach(([p, color, label]) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(p.x, p.y); ctx.stroke();
        ctx.fillStyle = color; ctx.fillText(label, p.x + 4, p.y + 4);
      });
      ctx.fillStyle = '#16a34a';
      ctx.beginPath(); ctx.arc(origin.x, origin.y, 6, 0, Math.PI * 2); ctx.fill();
    }
    function drawCloudToCanvas(canvas, cloudPoints, options = {}) {
      const ctx = canvas.getContext('2d');
      const localRotX = options.rotX ?? rotX;
      const localRotZ = options.rotZ ?? rotZ;
      const localZoom = options.zoom ?? zoom;
      const radius = options.radius ?? 3.2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#64748b';
      if (!cloudPoints.length) {
        ctx.fillText('No projected semantic points available.', 18, 28);
        drawAxes(canvas, ctx, localRotX, localRotZ, localZoom);
        return;
      }
      const projected = cloudPoints
        .map(p => ({
          ...project(canvas, p.x, p.y, p.z, localRotX, localRotZ, localZoom),
          semantic_id: p.semantic_id,
          distance_confidence: p.distance_confidence
        }))
        .sort((a, b) => a.z - b.z);
      projected.forEach(p => {
        const color = legendColor(p.semantic_id);
        const alpha = Math.max(0.18, Math.min(1, p.distance_confidence || 1));
        const pointRadius = radius * (0.65 + 0.35 * alpha);
        ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, pointRadius, 0, Math.PI * 2); ctx.fill();
      });
      drawAxes(canvas, ctx, localRotX, localRotZ, localZoom);
    }
    function drawPointCloud() {
      drawCloudToCanvas(document.getElementById('pointCloud'), selectedPoints());
    }
    function renderMiniMatrix(root, semanticId) {
      const matrix = DATA.semantic_matrix || [];
      root.style.gridTemplateColumns = `repeat(${matrix.length || 1}, 1fr)`;
      root.innerHTML = '';
      matrix.forEach(row => row.forEach(value => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.style.background = value === semanticId ? rgb(legendColor(value)) : '#e2e8f0';
        root.appendChild(cell);
      }));
    }
    function renderClassGallery() {
      const root = document.getElementById('classGallery');
      root.innerHTML = '';
      const visibleClasses = classes
        .filter(item => Number(item.semantic_id) !== 0)
        .sort((a, b) => Number(b.point_count || 0) - Number(a.point_count || 0))
        .slice(0, 24);
      if (!visibleClasses.length) {
        root.innerHTML = '<div class="muted">No semantic classes with projected points.</div>';
        return;
      }
      visibleClasses.forEach(item => {
        const semanticId = Number(item.semantic_id);
        const card = document.createElement('article');
        card.className = 'class-card';
        const examples = rawExampleText(semanticId, item);
        card.innerHTML = `
          <h3><span class="swatch" style="background:${rgb(legendColor(semanticId))}"></span> ${escapeText(item.label || classLabel(semanticId))}</h3>
          <div class="muted">${Number(item.point_count || pointCountFor(semanticId))} points · ${(pointRatioFor(semanticId) * 100).toFixed(2)}% · id ${semanticId}</div>
          ${examples ? `<div class="muted">raw: ${escapeText(examples)}</div>` : ''}
          <canvas width="300" height="170"></canvas>
          <div class="mini-matrix"></div>
        `;
        card.addEventListener('click', () => selectSemanticClass(semanticId));
        root.appendChild(card);
        const miniCanvas = card.querySelector('canvas');
        const miniPoints = points.filter(point => point.semantic_id === semanticId);
        drawCloudToCanvas(miniCanvas, miniPoints, {rotX: -0.82, rotZ: 0.72, zoom: 16, radius: 2.5});
        renderMiniMatrix(card.querySelector('.mini-matrix'), semanticId);
      });
    }
    function renderSourceFrames() {
      const root = document.getElementById('sourceFrames');
      const framesByObservation = debug.source_frames_by_observation_id || {};
      let observationIds;
      if (selectedSemanticId == null) {
        observationIds = Object.keys(framesByObservation);
      } else {
        observationIds = sourceObservationIdsFor(selectedSemanticId);
      }
      root.innerHTML = '';
      if (!observationIds.length) {
        root.innerHTML = '<div class="muted">No source frame references for this selection.</div>';
        return;
      }
      observationIds.slice(0, 24).forEach(observationId => {
        const frames = framesByObservation[observationId] || {};
        const card = document.createElement('article');
        card.className = 'source-card';
        const rgbFrame = frames.rgb || {};
        const maskFrame = frames.mask || {};
        const depthFrame = frames.depth || {};
        card.innerHTML = `
          <h3>${escapeText(observationId)}</h3>
          <div class="thumbs">
            <div class="thumb"><div class="thumb-title">RGB</div>${rgbFrame.html_src ? `<img alt="rgb ${escapeText(observationId)}" src="${escapeText(rgbFrame.html_src)}">` : '<div class="muted">missing</div>'}</div>
            <div class="thumb"><div class="thumb-title">Mask</div>${maskFrame.html_src ? `<img alt="mask ${escapeText(observationId)}" src="${escapeText(maskFrame.html_src)}">` : '<div class="muted">missing</div>'}</div>
            <div class="thumb"><div class="thumb-title">Depth heatmap</div>${depthFrame.depth_heatmap_src ? `<img alt="depth ${escapeText(observationId)}" src="${escapeText(depthFrame.depth_heatmap_src)}">` : '<div class="muted">missing</div>'}</div>
          </div>
          <dl class="kv" style="margin-top:8px">
            <dt>depth</dt><dd>${escapeText(depthFrame.artifact_path || 'missing')}</dd>
            <dt>rgb</dt><dd>${escapeText(rgbFrame.artifact_path || 'missing')}</dd>
            <dt>mask</dt><dd>${escapeText(maskFrame.artifact_path || 'missing')}</dd>
          </dl>
        `;
        root.appendChild(card);
      });
    }
    function renderDebug() {
      renderKv('debug', [
        ['projection_status', debug.projection_status || DATA.map_config.semantic_projection],
        ['point_count', debug.point_count || 0],
        ['sampled_points', (debug.sampled_points || []).length],
        ['debug_point_limit', debug.debug_point_limit == null ? 'none' : debug.debug_point_limit],
        ['cell_pooling_strategy', debug.cell_pooling_strategy || 'none'],
        ['priority_override_count', debug.priority_override_count || 0],
        ['low_confidence_suppressed_counts', debug.low_confidence_suppressed_counts || {}],
        ['semantic_point_counts', debug.semantic_point_counts || {}],
        ['semantic_confidence_summary', debug.semantic_confidence_summary || {}],
        ['semantic_point_ratios', debug.semantic_point_ratios || {}],
        ['semantic_raw_label_examples', debug.semantic_raw_label_examples || {}],
        ['distance_confidence_config', debug.distance_confidence_config || {}],
        ['depth_interpretation_used', debug.depth_interpretation_used || 'unknown'],
        ['depth_interpretation_comparison', debug.depth_interpretation_comparison || {}],
        ['frame_point_ranges', debug.frame_point_ranges || []],
        ['source_observation_count', debug.source_observation_count || 0],
        ['camera_model', debug.camera_model || {}],
        ['map_config', DATA.map_config || {}],
        ['center_pose', DATA.center_pose || {}]
      ]);
    }
    function depthPoint(modePoint) {
      return {
        x: Number(modePoint.local_right_m || 0),
        y: Number(modePoint.local_up_m || 0),
        z: Number(modePoint.local_forward_m || 0)
      };
    }
    function drawDepthModeCompare() {
      const canvas = document.getElementById('depthModeCompare');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const comparison = debug.depth_interpretation_comparison || {};
      const samples = comparison.samples || [];
      drawAxes(canvas, ctx, -0.72, 0.55, 6);
      samples.forEach(sample => {
        const optical = sample.depth_as_optical_z;
        const ray = sample.depth_as_ray_distance;
        if (!optical || !ray) return;
        const po = project(canvas, depthPoint(optical).x, depthPoint(optical).y, depthPoint(optical).z, -0.72, 0.55, 6);
        const pr = project(canvas, depthPoint(ray).x, depthPoint(ray).y, depthPoint(ray).z, -0.72, 0.55, 6);
        ctx.strokeStyle = 'rgba(100,116,139,0.22)';
        ctx.beginPath(); ctx.moveTo(po.x, po.y); ctx.lineTo(pr.x, pr.y); ctx.stroke();
        ctx.fillStyle = 'rgba(37,99,235,0.7)';
        ctx.fillRect(po.x, po.y, 2, 2);
        ctx.fillStyle = 'rgba(220,38,38,0.7)';
        ctx.fillRect(pr.x, pr.y, 2, 2);
      });
      ctx.fillStyle = '#2563eb'; ctx.fillText('blue = optical-z (current map)', 18, 24);
      ctx.fillStyle = '#dc2626'; ctx.fillText('red = ray-distance debug interpretation', 18, 44);
      renderKv('depthModeDebug', [
        ['primary_mode', comparison.primary_mode || 'unknown'],
        ['comparison_mode', comparison.comparison_mode || 'unknown'],
        ['sample_count', samples.length],
        ['ranges', comparison.ranges || {}]
      ]);
    }
    function selectSemanticClass(semanticId) {
      selectedSemanticId = semanticId;
      renderClassSelector();
      renderClassDetails();
      renderMatrix();
      renderLegend();
      drawPointCloud();
      renderSourceFrames();
    }

    renderKv('summary', Object.entries(DATA.path_summary || {}));
    renderDebug();
    renderClassGallery();
    selectSemanticClass(null);
    drawDepthModeCompare();

    const canvas = document.getElementById('pointCloud');
    canvas.addEventListener('mousedown', event => { dragging = true; lastX = event.clientX; lastY = event.clientY; });
    window.addEventListener('mouseup', () => dragging = false);
    window.addEventListener('mousemove', event => {
      if (!dragging) return;
      rotZ += (event.clientX - lastX) * 0.01;
      rotX += (event.clientY - lastY) * 0.01;
      lastX = event.clientX; lastY = event.clientY;
      drawPointCloud();
    });
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      zoom = Math.max(4, Math.min(160, zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
      drawPointCloud();
    }, {passive: false});
  </script>
</body>
</html>
"""
    return template.replace("__CUSTOM_MAP_PAYLOAD__", payload)
