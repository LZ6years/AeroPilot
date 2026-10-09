"""Render public execution history and original images without provider debug.

Reports reference one retained media tree, so regenerating HTML never flies or
copies images. Evaluation truth and trajectories remain human-only evidence.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from .report_io import atomic_write_text
from .trajectory_plots import write_ned_comparison


def _records(path: Path, limit: int | None = None):
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines[:limit] if line.strip()]


def _text(value):
    return html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def _link(path: Path, directory: Path):
    return quote(Path(os.path.relpath(path, directory)).as_posix(), safe="/.")


def write_process_review(case_dir, result, instruction=None, *, profile="paper_process"):
    """Regenerate solely from durable transcript, frames and scoring records."""
    directory = Path(case_dir)
    run = Path(result["run_dir"])
    cursor_path = directory / "audit_cursor.json"
    bounds = json.loads(cursor_path.read_text(encoding="utf-8"))["end_lines"] if cursor_path.is_file() else {}
    transcript = _records(run / "memory/agent_transcript.jsonl", bounds.get("memory/agent_transcript.jsonl"))
    frames = _records(run / "memory/frames.jsonl", bounds.get("memory/frames.jsonl"))
    if instruction is None:
        instructions = _records(run / "memory/instructions.jsonl")
        instruction = instructions[0].get("instruction", "") if instructions else ""
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<style>body{font:16px system-ui;margin:2rem auto;max-width:1080px;padding:0 1rem;color:#17212c;background:#fafafa}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:1rem}figure{margin:1.5rem 0}article{border-top:1px solid #ccd4df;padding:1rem 0}</style>',
             f'<title>{_text(result["episode_id"])}</title><h1>{_text(result["dataset_split"])} · {_text(result["episode_id"])}</h1>',
             f'<p>{_text(instruction or "")}</p>',
             '<h2>Outcome</h2><pre>' + _text({key: result.get(key) for key in (
                 "status", "completed", "stopped_reason", "final_score_available", "error", "difficulty")}) + '</pre>',
             '<pre>' + _text({key: value for key, value in result.get("metrics", {}).items()
                             if key not in {"trajectory_ned", "reference_path_ned"}}) + '</pre>']
    # Frames are linked by their actual chronological observation IDs.
    images = {}
    rendered = set()
    for frame in frames:
        if frame.get("modality") != "rgb":
            continue
        metadata = frame.get("metadata") or {}
        raw_path = metadata.get("artifact_path") or frame.get("path") or metadata.get("path")
        if raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = run / path
            if path.is_file():
                images.setdefault(frame.get("observation_id"), []).append((frame, path))

    def image_part(path, caption):
        rendered.add(path.resolve())
        return f'<figure><figcaption>{_text(caption)}</figcaption><a href="{_link(path, directory)}"><img loading="lazy" src="{_link(path, directory)}"></a></figure>'

    parts.append('<h2>Execution timeline</h2>')
    for event in transcript:
        payload = event.get("payload") or {}
        kind = event.get("kind")
        if kind == "state_checkpoint":
            observation_id = payload.get("observation_id")
            for frame, path in images.pop(observation_id, []):
                parts.append(image_part(path, f'{observation_id} · {frame.get("timestamp", "")}'))
        elif kind == "assistant_turn":
            parts.append('<article><p>' + _text(payload.get("public_intent") or "") + '</p><pre>' +
                         _text(payload.get("tool_calls", [])) + '</pre></article>')
        elif kind == "tool_result" and profile == "paper_process":
            parts.append('<pre>' + _text(payload) + '</pre>')
    # Retain initial, terminal and interrupted observations even without a checkpoint.
    for group in images.values():
        for frame, path in group:
            parts.append(image_part(path, f'{frame.get("observation_id")} · {frame.get("timestamp", "")}'))
    evidence = [path for path in sorted((run / "artifacts").rglob("*.png"))
                if path.resolve() not in rendered and not any("depth" in part.lower() for part in path.parts)]
    if evidence:
        parts.append('<h2>Tool evidence</h2>')
        for path in evidence:
            parts.append(image_part(path, path.name))
    if profile == "paper_process":
        metrics = result.get("metrics", {})
        plots = write_ned_comparison(directory, metrics.get("trajectory_ned", []), metrics.get("reference_path_ned", []))
        for key in ("path_2d", "path_3d"):
            if plots.get(key):
                parts.append(image_part(Path(plots[key]), "Trajectory comparison (NED metres; display Z up)"))
    parts.append('</html>')
    output = directory / "review.html"
    atomic_write_text(output, "\n".join(parts))
    return {"path": str(output), "image_count": len(rendered), "profile": profile}
