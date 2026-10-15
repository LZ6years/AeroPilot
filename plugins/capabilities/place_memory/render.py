"""Render immutable model maps and a clickable, self-contained HTML archive."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

from src.memory.context.media_refs import image_media_ref
from .store import atomic_json, annotation_status
from .layout import draw_map, RENDER_VERSION, IMAGE_SIZE


def render_map(store, graph):
    """Publish a new immutable version; old evidence hashes remain valid."""
    payload = json.dumps({"renderer": RENDER_VERSION,
                          "graph": {k: v for k, v in graph.items() if k != "committed_calls"}},
                         sort_keys=True, ensure_ascii=False).encode("utf-8")
    version = hashlib.sha256(payload).hexdigest()[:20]
    path = store.directory / f"map_{version}.png"
    layout = draw_map(graph)
    if not path.exists():
        store.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        layout.image.save(temporary, format="PNG")
        temporary.replace(path)
        atomic_json(store.directory / f"map_{version}.json", graph)
    _write_html(store, graph, path, layout)
    current_observation = next(t["observation_id"] for t in reversed(graph["trajectory"]) if t["observation_id"])
    media = image_media_ref(path, run_dir=store.run_dir, observation_id=current_observation,
        frame_id=f"place_map_{version}", role="visited_place_map", latest_group="task_place_map",
        usage="Task-local historical map of measured observation sites, visited path and UAV heading; place labels are model annotations, not current visibility or verified free space.")
    return media.to_model_input(), store.directory / "index.html"


def _write_html(store, graph, path, layout, *, photo_run_dir=None, offline=False):
    """Keep original image references relative so moving the whole run works."""
    parts = ['<!doctype html><meta charset="utf-8"><title>Visited places</title>',
             '<style>body{font:16px/1.6 system-ui;margin:24px;color:#172b4d}img{max-width:100%}.map{position:relative;width:1600px;max-width:100%}.map>img{display:block;width:100%}.point{position:absolute;background:transparent;border:0;cursor:pointer;text-decoration:none;border-radius:6px}.point:hover{background:#1878f522;outline:2px solid #1878f5}article{padding:16px;border-top:1px solid #ccd}article img{width:256px;margin:8px}pre{white-space:pre-wrap}.group{padding:12px;background:#eff5fb;border-radius:8px}a{color:#1878f5}</style>',
             '<h1>地点拓扑与访问记录</h1><p>地点为无人机观察位置；标注属于模型记忆，不代表目标真值或已确认的通行空间。</p>',
             '<div class="map"><img src="' + path.name + '">']
    width, height = IMAGE_SIZE
    for index, hit in enumerate(layout.hit_regions):
        ids, (x1, y1, x2, y2) = hit["ids"], hit["box"]
        target = ids[0] if len(ids) == 1 else f"group-{index}"
        title = html.escape(" / ".join(ids), quote=True)
        parts.append(f'<a class="point" aria-label="{title}" title="{title}" href="#{target}" style="left:{x1 / width * 100:.6f}%;top:{y1 / height * 100:.6f}%;width:{(x2-x1) / width * 100:.6f}%;height:{(y2-y1) / height * 100:.6f}%"></a>')
    parts.append('</div>')
    if offline:
        parts.append('<p>离线重绘：只读取原实验的实测记录；这些新版地图未用于旧模型决策。</p>')
    for index, hit in enumerate(layout.hit_regions):
        if len(hit["ids"]) > 1:
            links = ' · '.join(f'<a href="#{pid}">{pid}</a>' for pid in hit["ids"])
            parts.append(f'<p class="group" id="group-{index}">重叠地点：{links}</p>')
    for node in graph["nodes"]:
        status = annotation_status(node)
        parts.append(f'<article id="{node["id"]}"><h2>{node["id"]}</h2><p>' + html.escape(node["summary"]) + '</p><p>地标：' + html.escape(', '.join(node["landmarks"])) + '</p><p>观察：' + html.escape(', '.join(node["visits"])) + '</p><p>最近存档观察：' + html.escape(status["last_saved_observation_id"] or '尚无存档记录') + '；存档后新增观察：' + str(status["new_observations_since_save"]) + '</p>')
        for photo in node["photos"]:
            target = store.photo_path(photo["ref"]) if photo_run_dir is None else (Path(photo_run_dir) / photo["ref"]).resolve()
            if photo_run_dir is not None and (Path(photo_run_dir).resolve() not in target.parents or not target.is_file()):
                raise ValueError("offline photo is missing or outside its source run")
            ref = Path(os.path.relpath(target, store.directory)).as_posix()
            parts.append('<a href="' + html.escape(ref, quote=True) + '"><img src="' + html.escape(ref, quote=True) + '"></a><span>' + html.escape(photo["observation_id"] + ' ' + photo["timestamp"]) + '</span>')
        parts.append('<details><summary>实际执行动作</summary><pre>' + html.escape(json.dumps(node["actions"], ensure_ascii=False, indent=2)) + '</pre></details></article>')
    temporary = store.directory / "index.tmp"
    temporary.write_text(''.join(parts), encoding="utf-8")
    temporary.replace(store.directory / "index.html")
