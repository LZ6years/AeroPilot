"""Static HTML report for STMR open-vocabulary perception and matrix output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .core import StmrMap


def render_stmr_html(
    stmr_map: StmrMap,
    image_path: Path,
    output_dir: Path,
    *,
    landmarks: list[dict[str, Any]] | None = None,
    landmark_debug: dict[str, Any] | None = None,
    perception_results: list[dict[str, Any]] | None = None,
    custom_map_json: Path | None = None,
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
        "projection_debug": stmr_map.projection_debug,
        "source_frames": _frames_for_html(stmr_map.source_frames, output_dir),
        "landmarks": landmarks or [],
        "landmark_debug": landmark_debug or {},
        "perception_results": _perception_for_html(perception_results or [], output_dir),
        "custom_map": custom_map,
        "alignment_checks": _alignment_checks(stmr_map, custom_map),
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
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STMR Debug Viewer</title>
  <style>
    body {{ margin: 0; background: #f8fafc; color: #0f172a; font: 14px/1.45 system-ui, sans-serif; }}
    header {{ padding: 18px 24px; background: white; border-bottom: 1px solid #cbd5e1; }}
    main {{ padding: 18px 24px; display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 16px; }}
    section {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; min-width: 0; }}
    .full {{ grid-column: 1 / -1; }}
    img {{ width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #e2e8f0; background: white; }}
    .matrix {{ display: grid; gap: 1px; background: #cbd5e1; aspect-ratio: 1 / 1; max-width: 520px; }}
    .cell {{ min-width: 0; min-height: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; }}
    canvas {{ width: 100%; height: 520px; border: 1px solid #cbd5e1; background: #020617; border-radius: 8px; }}
    .toolbar {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin: 8px 0 12px; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e2e8f0; color: #334155; }}
    .warn {{ color: #991b1b; background: #fee2e2; padding: 8px; border-radius: 6px; }}
    pre {{ overflow: auto; max-height: 380px; background: #f1f5f9; padding: 10px; border-radius: 6px; }}
    code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
    .muted {{ color: #64748b; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>STMR Debug Viewer</h1>
    <div class="muted">GroundingDINO/SAM masks -> voxel top-height projection -> 20x20 STMR matrix.</div>
  </header>
  <main>
    <section>
      <h2>STMR Top-down PNG</h2>
      <img id="stmrImage" alt="STMR map">
    </section>
    <section>
      <h2>STMR Matrix</h2>
      <div id="matrix" class="matrix"></div>
    </section>
    <section class="full">
      <h2>Coordinate Frame</h2>
      <div id="alignmentWarnings"></div>
      <pre id="coordinateFrame"></pre>
    </section>
    <section class="full">
      <h2>3D Alignment Compare</h2>
      <div class="toolbar">
        <label><input id="showCustom" type="checkbox" checked> custom_map projected points</label>
        <label><input id="showStmr" type="checkbox" checked> STMR voxel centers</label>
        <span id="pointStats" class="pill"></span>
      </div>
      <canvas id="compare3d" width="1200" height="620"></canvas>
      <div class="muted">Drag to rotate, wheel to zoom. Axes use the current UAV-centered frame: forward/right/up.</div>
      <pre id="rangeDebug"></pre>
    </section>
    <section class="full">
      <h2>Landmarks</h2>
      <pre id="landmarks"></pre>
    </section>
    <section class="full">
      <h2>Landmark Extractor Debug</h2>
      <pre id="landmarkDebug"></pre>
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
      <h2>Depth Projection Diagnostics</h2>
      <canvas id="depthModeCompare" width="1200" height="420"></canvas>
      <pre id="depthModeDebug"></pre>
    </section>
    <section class="full">
      <h2>Projection Debug</h2>
      <pre id="debug"></pre>
    </section>
  </main>
  <script id="STMR_DATA" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('STMR_DATA').textContent);
    document.getElementById('stmrImage').src = DATA.image;
    function color(id) {{
      const item = DATA.semantic_legend[String(id)] || {{}};
      if (Array.isArray(item.color)) return `rgb(${{item.color[0]}}, ${{item.color[1]}}, ${{item.color[2]}})`;
      if (id === -1) return '#2563eb';
      if (id === -2) return '#94a3b8';
      if (id === 0) return '#ffffff';
      return '#dbeafe';
    }}
    const matrix = DATA.semantic_matrix || [];
    const root = document.getElementById('matrix');
    root.style.gridTemplateColumns = `repeat(${{matrix.length || 1}}, 1fr)`;
    matrix.forEach(row => row.forEach(value => {{
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.style.background = color(value);
      cell.title = `${{value}}: ${{(DATA.semantic_legend[String(value)] || {{}}).label || 'unknown'}}`;
      root.appendChild(cell);
    }}));
    document.getElementById('landmarks').textContent = JSON.stringify(DATA.landmarks, null, 2);
    document.getElementById('landmarkDebug').textContent = JSON.stringify(DATA.landmark_debug || {{}}, null, 2);
    document.getElementById('debug').textContent = JSON.stringify(DATA.projection_debug, null, 2);
    document.getElementById('coordinateFrame').textContent = JSON.stringify({{
      coordinate_frame: DATA.coordinate_frame,
      alignment_checks: DATA.alignment_checks,
      stmr_center_pose: DATA.center_pose,
      custom_center_pose: DATA.custom_map ? DATA.custom_map.center_pose : null
    }}, null, 2);
    const warningRoot = document.getElementById('alignmentWarnings');
    const warnings = (DATA.alignment_checks && DATA.alignment_checks.warnings) || [];
    if (warnings.length) {{
      warningRoot.innerHTML = `<div class="warn">${{warnings.join('<br>')}}</div>`;
    }} else {{
      warningRoot.innerHTML = '<span class="pill">coordinate checks passed</span>';
    }}
    const perception = document.getElementById('perception');
    (DATA.perception_results || []).forEach(item => {{
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `<h3>${{item.observation_id}}</h3>
        <div class="muted">${{(item.kept_masks || []).length}} kept masks / ${{(item.rejected_masks || []).length}} rejected / ${{(item.kept_non_primary_masks || []).length}} non-primary</div>
        ${{item.primary_dynamic_track ? `<div class="pill">primary: ${{item.primary_dynamic_track.label}} (${{item.primary_dynamic_track.candidate_count}} masks)</div>` : ''}}
        ${{item.boxes_path_src ? `<h4>Boxes</h4><img src="${{item.boxes_path_src}}">` : ''}}
        ${{item.overlay_path_src ? `<h4>Overlay</h4><img src="${{item.overlay_path_src}}">` : ''}}
        <h4>TF-IDF Matches</h4><pre>${{JSON.stringify(item.tfidf_matches || [], null, 2)}}</pre>
        <h4>Rejected Dynamic Masks</h4><pre>${{JSON.stringify(item.rejected_masks || [], null, 2)}}</pre>`;
      perception.appendChild(card);
    }});
    const dynamicRoot = document.getElementById('dynamicCandidates');
    const dynamicItems = (DATA.perception_results || []).filter(item => (item.dynamic_target_candidates || []).length);
    if (!dynamicItems.length) {{
      dynamicRoot.innerHTML = '<div class="muted">No dynamic target candidates were recorded.</div>';
    }}
    dynamicItems.forEach(item => {{
      const card = document.createElement('article');
      card.className = 'card';
      const counts = {{}};
      (item.dynamic_target_candidates || []).forEach(candidate => {{
        const status = candidate.status || 'unknown';
        counts[status] = (counts[status] || 0) + 1;
      }});
      card.innerHTML = `<h3>${{item.observation_id}}</h3>
        <div class="muted">${{JSON.stringify(counts)}}</div>
        <h4>Primary Track</h4><pre>${{JSON.stringify(item.primary_dynamic_track || null, null, 2)}}</pre>
        <h4>Candidates</h4><pre>${{JSON.stringify(item.dynamic_target_candidates || [], null, 2)}}</pre>`;
      dynamicRoot.appendChild(card);
    }});
    const sources = document.getElementById('sources');
    Object.entries(DATA.source_frames || {{}}).forEach(([obs, bundle]) => {{
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `<h3>${{obs}}</h3>` + Object.entries(bundle).map(([name, frame]) => {{
        const img = frame.html_src && (name === 'rgb' || name === 'stmr_open_vocab_mask') ? `<img src="${{frame.html_src}}">` : '';
        return `<div><code>${{name}}</code><div class="muted">${{frame.artifact_path || ''}}</div>${{img}}</div>`;
      }}).join('');
      sources.appendChild(card);
    }});

    function worldToLocal(point, center) {{
      if (point.local_right_m != null && point.local_up_m != null && point.local_forward_m != null) {{
        return {{
          x: point.local_right_m || 0,
          y: point.local_up_m || 0,
          z: point.local_forward_m || 0,
          semantic_id: point.semantic_id,
          layer: 'custom'
        }};
      }}
      const yaw = ((center.yaw_deg || 0) * Math.PI) / 180;
      const dx = (point.world_x || 0) - (center.x || 0);
      const dy = (point.world_y || 0) - (center.y || 0);
      return {{
        x: -Math.sin(yaw) * dx + Math.cos(yaw) * dy,
        y: (point.world_z || 0) - (center.z || 0),
        z: Math.cos(yaw) * dx + Math.sin(yaw) * dy,
        semantic_id: point.semantic_id,
        layer: 'custom'
      }};
    }}
    function stmrVoxelToLocal(point) {{
      return {{
        x: point.local_right_m || 0,
        y: point.local_up_m || 0,
        z: point.local_forward_m || 0,
        semantic_id: point.semantic_id,
        point_count: point.point_count || 1,
        layer: 'stmr'
      }};
    }}
    const customPoints = ((((DATA.custom_map || {{}}).projection_debug || {{}}).sampled_points || [])
      .map(point => worldToLocal(point, DATA.center_pose || {{}})));
    const stmrPoints = (((DATA.projection_debug || {{}}).voxel_world_centers || []).map(stmrVoxelToLocal));
    const ranges = computeRanges(customPoints, stmrPoints);
    document.getElementById('rangeDebug').textContent = JSON.stringify(ranges, null, 2);
    document.getElementById('pointStats').textContent = `${{customPoints.length}} custom points / ${{stmrPoints.length}} STMR voxels`;

    const canvas = document.getElementById('compare3d');
    const ctx = canvas.getContext('2d');
    let rotX = -0.8;
    let rotY = 0.65;
    let zoom = 7.5;
    let dragging = false;
    let last = null;
    canvas.addEventListener('pointerdown', event => {{
      dragging = true;
      last = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    }});
    canvas.addEventListener('pointerup', () => {{ dragging = false; last = null; }});
    canvas.addEventListener('pointermove', event => {{
      if (!dragging || !last) return;
      rotY += (event.clientX - last[0]) * 0.008;
      rotX += (event.clientY - last[1]) * 0.008;
      last = [event.clientX, event.clientY];
      drawCompare();
    }});
    canvas.addEventListener('wheel', event => {{
      event.preventDefault();
      zoom *= event.deltaY > 0 ? 0.9 : 1.1;
      zoom = Math.max(1.5, Math.min(35, zoom));
      drawCompare();
    }}, {{ passive: false }});
    document.getElementById('showCustom').addEventListener('change', drawCompare);
    document.getElementById('showStmr').addEventListener('change', drawCompare);

    function semanticColor(id, alpha) {{
      const item = DATA.semantic_legend[String(id)] || (DATA.custom_map && DATA.custom_map.semantic_legend ? DATA.custom_map.semantic_legend[String(id)] : null) || {{}};
      const raw = Array.isArray(item.color) ? item.color : [219, 234, 254];
      return `rgba(${{raw[0]}}, ${{raw[1]}}, ${{raw[2]}}, ${{alpha}})`;
    }}
    function project(point) {{
      const cy = Math.cos(rotY), sy = Math.sin(rotY);
      const cx = Math.cos(rotX), sx = Math.sin(rotX);
      const x1 = point.x * cy - point.z * sy;
      const z1 = point.x * sy + point.z * cy;
      const y1 = point.y * cx - z1 * sx;
      const z2 = point.y * sx + z1 * cx;
      const perspective = 900 / (900 + z2);
      return {{
        x: canvas.width / 2 + x1 * zoom * perspective,
        y: canvas.height / 2 - y1 * zoom * perspective,
        depth: z2
      }};
    }}
    function drawAxis() {{
      const axes = [
        [{{x:0,y:0,z:0}}, {{x:0,y:0,z:20}}, '#22c55e', 'forward'],
        [{{x:0,y:0,z:0}}, {{x:20,y:0,z:0}}, '#38bdf8', 'right'],
        [{{x:0,y:0,z:0}}, {{x:0,y:12,z:0}}, '#f97316', 'up']
      ];
      ctx.lineWidth = 2;
      axes.forEach(([a, b, colorValue, label]) => {{
        const pa = project(a), pb = project(b);
        ctx.strokeStyle = colorValue;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
        ctx.fillStyle = colorValue;
        ctx.fillText(label, pb.x + 4, pb.y + 4);
      }});
    }}
    function drawCompare() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.font = '13px system-ui';
      drawAxis();
      const items = [];
      if (document.getElementById('showCustom').checked) {{
        customPoints.forEach(point => items.push(point));
      }}
      if (document.getElementById('showStmr').checked) {{
        stmrPoints.forEach(point => items.push(point));
      }}
      const projected = items.map(point => [point, project(point)]).sort((a, b) => a[1].depth - b[1].depth);
      projected.forEach(([point, p]) => {{
        if (point.layer === 'stmr') {{
          const radius = Math.max(3, Math.min(9, Math.sqrt(point.point_count || 1)));
          ctx.fillStyle = semanticColor(point.semantic_id, 0.95);
          ctx.strokeStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }} else {{
          ctx.fillStyle = semanticColor(point.semantic_id, 0.28);
          ctx.fillRect(p.x, p.y, 1.4, 1.4);
        }}
      }});
      ctx.fillStyle = '#f8fafc';
      ctx.fillText('origin = current UAV pose', 16, 24);
    }}
    function computeRanges(custom, stmr) {{
      function one(points) {{
        if (!points.length) return null;
        const acc = {{count: points.length, right: [Infinity, -Infinity], up: [Infinity, -Infinity], forward: [Infinity, -Infinity]}};
        points.forEach(p => {{
          acc.right[0] = Math.min(acc.right[0], p.x); acc.right[1] = Math.max(acc.right[1], p.x);
          acc.up[0] = Math.min(acc.up[0], p.y); acc.up[1] = Math.max(acc.up[1], p.y);
          acc.forward[0] = Math.min(acc.forward[0], p.z); acc.forward[1] = Math.max(acc.forward[1], p.z);
        }});
        return acc;
      }}
      return {{ custom_map_local_ranges: one(custom), stmr_local_ranges: one(stmr) }};
    }}
    function drawDepthModeCompare() {{
      const depthCanvas = document.getElementById('depthModeCompare');
      const depthCtx = depthCanvas.getContext('2d');
      depthCtx.clearRect(0, 0, depthCanvas.width, depthCanvas.height);
      depthCtx.fillStyle = '#020617';
      depthCtx.fillRect(0, 0, depthCanvas.width, depthCanvas.height);
      const comparison = (DATA.projection_debug || {{}}).depth_interpretation_comparison || {{}};
      const samples = comparison.samples || [];
      const localZoom = 6;
      function projectDepth(point) {{
        const cy = Math.cos(0.6), sy = Math.sin(0.6);
        const cx = Math.cos(-0.75), sx = Math.sin(-0.75);
        const x = Number(point.local_right_m || 0);
        const y = Number(point.local_up_m || 0);
        const z = Number(point.local_forward_m || 0);
        const x1 = x * cy - z * sy;
        const z1 = x * sy + z * cy;
        const y1 = y * cx - z1 * sx;
        const z2 = y * sx + z1 * cx;
        const perspective = 900 / (900 + z2);
        return {{
          x: depthCanvas.width / 2 + x1 * localZoom * perspective,
          y: depthCanvas.height / 2 - y1 * localZoom * perspective
        }};
      }}
      samples.forEach(sample => {{
        const optical = sample.depth_as_optical_z;
        const ray = sample.depth_as_ray_distance;
        if (!optical || !ray) return;
        const po = projectDepth(optical);
        const pr = projectDepth(ray);
        depthCtx.strokeStyle = 'rgba(148,163,184,0.26)';
        depthCtx.beginPath(); depthCtx.moveTo(po.x, po.y); depthCtx.lineTo(pr.x, pr.y); depthCtx.stroke();
        depthCtx.fillStyle = 'rgba(37,99,235,0.85)';
        depthCtx.fillRect(po.x, po.y, 2, 2);
        depthCtx.fillStyle = 'rgba(220,38,38,0.85)';
        depthCtx.fillRect(pr.x, pr.y, 2, 2);
      }});
      depthCtx.fillStyle = '#bfdbfe';
      depthCtx.fillText('blue = optical-z current STMR projection', 16, 24);
      depthCtx.fillStyle = '#fecaca';
      depthCtx.fillText('red = ray-distance debug projection', 16, 44);
      document.getElementById('depthModeDebug').textContent = JSON.stringify({{
        primary_mode: comparison.primary_mode,
        comparison_mode: comparison.comparison_mode,
        sample_count: samples.length,
        ranges: comparison.ranges,
        frame_point_ranges: (DATA.projection_debug || {{}}).frame_point_ranges || []
      }}, null, 2);
    }}
    drawCompare();
    drawDepthModeCompare();
  </script>
</body>
</html>
"""
