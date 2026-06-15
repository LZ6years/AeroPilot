"""Static HTML review pages for UAV-Flow benchmark outputs.

The HTML layer is intentionally a post-processing view over existing
``summary.json`` and run artifacts. It does not copy RGB frames or change the
benchmark result schema; links point back to the original files so manual
inspection stays close to the raw evidence.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.agent.planning.model_inputs import build_model_user_content


def write_html_review(eval_dir: str | Path, summary: dict[str, Any]) -> dict[str, str]:
    """Write a batch overview and per-case HTML review pages."""

    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)
    case_paths: dict[str, str] = {}
    for result in summary.get("cases", []):
        if not isinstance(result, dict):
            continue
        case_id = str(result.get("case_id") or "unknown")
        case_dir = _path_from_result(result.get("case_dir"), eval_path / "cases" / case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_html = case_dir / "review.html"
        case_html.write_text(_case_page(eval_path, case_dir, result), encoding="utf-8")
        case_paths[case_id] = str(case_html)

    index_path = eval_path / "review.html"
    index_path.write_text(_index_page(eval_path, summary), encoding="utf-8")
    paths: dict[str, str] = {"html_review": str(index_path)}
    paths.update({f"case_html:{case_id}": path for case_id, path in case_paths.items()})
    return paths


def _index_page(eval_path: Path, summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    failures = summary.get("failure_counts", {}) if isinstance(summary.get("failure_counts"), dict) else {}
    cases = [item for item in summary.get("cases", []) if isinstance(item, dict)]
    rows = "\n".join(_case_row(eval_path, result) for result in cases)
    evidence_cards = "\n".join(_case_evidence_card(eval_path, result) for result in cases)
    class_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _e(class_name),
            _e(item.get("evaluated_count")),
            _e(item.get("mean_ndtw")),
        )
        for class_name, item in sorted((metrics.get("by_class") or {}).items())
        if isinstance(item, dict)
    )
    failure_items = "\n".join(
        f"<li><span>{_e(name)}</span><strong>{_e(count)}</strong></li>"
        for name, count in sorted(failures.items())
    )
    return _page(
        title="UAV-Flow Review",
        body=f"""
<header class="hero">
  <div>
    <p class="eyebrow">UAV-Flow Benchmark</p>
    <h1>Review Dashboard</h1>
  </div>
  <div class="metrics">
    <div><span>Cases</span><strong>{_e(metrics.get("case_count", len(cases)))}</strong></div>
    <div><span>Evaluated</span><strong>{_e(metrics.get("evaluated_count"))}</strong></div>
    <div><span>Mean nDTW</span><strong>{_e(metrics.get("overall_mean_ndtw"))}</strong></div>
  </div>
</header>
<section class="grid two">
  <article>
    <h2>By Class</h2>
    <table>
      <thead><tr><th>Class</th><th>Evaluated</th><th>Mean nDTW</th></tr></thead>
      <tbody>{class_rows or '<tr><td colspan="3">No class metrics.</td></tr>'}</tbody>
    </table>
  </article>
  <article>
    <h2>Failure Counts</h2>
    <ul class="stat-list">{failure_items or '<li>No failure counts.</li>'}</ul>
  </article>
</section>
<section>
  <h2>Cases</h2>
  <table class="case-table">
    <thead>
      <tr>
        <th>Case</th><th>Class</th><th>Status</th><th>Completed</th><th>nDTW</th>
        <th>SR</th><th>RGB</th><th>Turn</th><th>Move</th><th>Hover</th><th>Links</th>
      </tr>
    </thead>
    <tbody>{rows or '<tr><td colspan="11">No cases.</td></tr>'}</tbody>
  </table>
</section>
<section>
  <h2>Case Evidence</h2>
  <div class="case-cards">{evidence_cards or '<p class="empty">No case evidence.</p>'}</div>
</section>
""",
    )


def _case_row(eval_path: Path, result: dict[str, Any]) -> str:
    case_id = str(result.get("case_id") or "unknown")
    artifacts = result.get("run_artifacts", {}) if isinstance(result.get("run_artifacts"), dict) else {}
    ndtw = result.get("ndtw", {}) if isinstance(result.get("ndtw"), dict) else {}
    sr = result.get("sr_judge", {}) if isinstance(result.get("sr_judge"), dict) else {}
    counts = _action_counts(artifacts.get("actions", []))
    case_dir = _path_from_result(result.get("case_dir"), eval_path / "cases" / case_id)
    run_dir = _path_from_result(result.get("run_dir"), None)
    case_html = case_dir / "review.html"
    links = [
        _link(eval_path, case_html, "case"),
        _link(eval_path, case_dir, "case dir"),
    ]
    if run_dir is not None:
        links.append(_link(eval_path, run_dir, "run dir"))
    return """
<tr>
  <td><code>{case_id}</code></td>
  <td>{class_name}</td>
  <td>{status}</td>
  <td>{completed}</td>
  <td>{ndtw}</td>
  <td>{sr}</td>
  <td>{rgb}</td>
  <td>{turn}</td>
  <td>{move}</td>
  <td>{hover}</td>
  <td>{links}</td>
</tr>
""".format(
        case_id=_e(case_id),
        class_name=_e(result.get("class_name")),
        status=_badge(result.get("status")),
        completed=_e(result.get("completed")),
        ndtw=_e(ndtw.get("ndtw")),
        sr=_sr_label(sr),
        rgb=_e(artifacts.get("rgb_count", 0)),
        turn=_e(counts.get("uav.turn_relative", 0)),
        move=_e(counts.get("uav.move_body", 0)),
        hover=_e(counts.get("uav.hover", 0)),
        links=" ".join(links),
    )


def _case_evidence_card(eval_path: Path, result: dict[str, Any]) -> str:
    case_id = str(result.get("case_id") or "unknown")
    artifacts = result.get("run_artifacts", {}) if isinstance(result.get("run_artifacts"), dict) else {}
    ndtw = result.get("ndtw", {}) if isinstance(result.get("ndtw"), dict) else {}
    sr = result.get("sr_judge", {}) if isinstance(result.get("sr_judge"), dict) else {}
    case_dir = _path_from_result(result.get("case_dir"), eval_path / "cases" / case_id)
    case_payload = _read_json(case_dir / "case.json")
    case = case_payload.get("case", {}) if isinstance(case_payload.get("case"), dict) else {}
    instruction = (
        case.get("selected_instruction")
        or case.get("instruction_unified")
        or case.get("instruction")
        or result.get("instruction")
        or ""
    )
    run_dir = _path_from_result(result.get("run_dir"), None)
    frames = _rgb_frames(result, artifacts)
    actions = artifacts.get("actions", []) if isinstance(artifacts.get("actions"), list) else []
    messages = _agent_messages(run_dir, artifacts)
    counts = _action_counts(actions)
    display_actions = _display_actions(actions)
    action_preview = _action_preview(eval_path, display_actions, frames)
    message_preview = _message_preview(messages)
    return f"""
<article class="case-card" id="case-{_e(case_id)}">
  <div class="case-card-head">
    <div>
      <p class="eyebrow">{_e(result.get("class_name"))}</p>
      <h3>{_e(case_id)}</h3>
      <p class="case-instruction">{_e(instruction)}</p>
    </div>
    <div class="case-actions">
      {_link(eval_path, case_dir / "review.html", "open case review")}
      {_link(eval_path, case_dir, "case output")}
      {_link(eval_path, run_dir, "run dir") if run_dir is not None else ''}
    </div>
  </div>
  <dl class="mini-kv">
    {_kv("status", result.get("status"))}
    {_kv("completed", result.get("completed"))}
    {_kv("nDTW", ndtw.get("ndtw"))}
    {_kv("SR", sr.get("success"))}
    {_kv("turn/move/hover", f"{counts.get('uav.turn_relative', 0)} / {counts.get('uav.move_body', 0)} / {counts.get('uav.hover', 0)}")}
  </dl>
  <div class="evidence-grid">
    <div class="rgb-panel">
      <h4>RGB</h4>
      {_key_frame_gallery(eval_path, frames, include_descriptions=True, run_dir=run_dir)}
      <div class="links compact">{_link(eval_path, run_dir / 'artifacts' / 'rgb', 'rgb folder') if run_dir is not None else ''}</div>
    </div>
    <div class="actions-panel">
      <h4>Actions</h4>
      {action_preview}
      <div class="links compact">{_link(eval_path, run_dir / 'memory' / 'actions.jsonl', 'actions.jsonl') if run_dir is not None else ''}</div>
    </div>
    <div class="messages-panel">
      <h4>Agent Messages</h4>
      {message_preview}
      <div class="links compact">{_link(eval_path, run_dir / 'memory' / 'agent_messages.jsonl', 'agent_messages.jsonl') if run_dir is not None else ''}</div>
    </div>
  </div>
</article>
"""


def _case_page(eval_path: Path, case_dir: Path, result: dict[str, Any]) -> str:
    case_id = str(result.get("case_id") or "unknown")
    case_payload = _read_json(case_dir / "case.json")
    case = case_payload.get("case", {}) if isinstance(case_payload.get("case"), dict) else {}
    artifacts = result.get("run_artifacts", {}) if isinstance(result.get("run_artifacts"), dict) else {}
    ndtw = result.get("ndtw", {}) if isinstance(result.get("ndtw"), dict) else {}
    sr = result.get("sr_judge", {}) if isinstance(result.get("sr_judge"), dict) else {}
    visual = result.get("visual_diagnostics", {}) if isinstance(result.get("visual_diagnostics"), dict) else {}
    plot = result.get("plot_result", {}) if isinstance(result.get("plot_result"), dict) else {}
    leak = result.get("evaluation_truth_leak_check", {}) if isinstance(result.get("evaluation_truth_leak_check"), dict) else {}
    rgb_frames = _rgb_frames(result, artifacts)
    actions = artifacts.get("actions", []) if isinstance(artifacts.get("actions"), list) else []
    run_dir = _path_from_result(result.get("run_dir"), None)
    messages = _agent_messages(run_dir, artifacts)
    planner_debug = _action_planner_debug_records(run_dir)
    phase_debug = _phase_transition_debug_records(run_dir)
    phase_events = _phase_transition_events(run_dir)
    return _page(
        title=f"UAV-Flow Case {case_id}",
        body=f"""
<nav class="top-nav">{_link(case_dir, eval_path / 'review.html', 'Back to batch review')}</nav>
<header class="hero">
  <div>
    <p class="eyebrow">{_e(result.get("class_name"))}</p>
    <h1>{_e(case_id)}</h1>
    <p class="instruction">{_e(case.get("selected_instruction") or case.get("instruction") or '')}</p>
  </div>
  <div class="metrics">
    <div><span>nDTW</span><strong>{_e(ndtw.get("ndtw"))}</strong></div>
    <div><span>SR</span><strong>{_e(sr.get("success"))}</strong></div>
    <div><span>RGB</span><strong>{_e(len(rgb_frames) or artifacts.get("rgb_count", 0))}</strong></div>
  </div>
</header>
<section class="grid two">
  <article>
    <h2>Quick Review</h2>
    <dl class="kv">
      {_kv("Status", result.get("status"))}
      {_kv("Completed", result.get("completed"))}
      {_kv("Stopped reason", result.get("stopped_reason"))}
      {_kv("Metric mode", (ndtw.get("metric_interpretation") or {}).get("mode"))}
      {_kv("SR status", sr.get("status"))}
      {_kv("SR failure", sr.get("failure_type"))}
      {_kv("Truth leak", leak.get("status"))}
      {_kv("Visual diagnostics", _visual_summary(visual))}
    </dl>
  </article>
  <article>
    <h2>Artifacts</h2>
    <div class="links">
      {_artifact_links(case_dir, result, run_dir, plot)}
    </div>
  </article>
</section>
<section>
  <h2>Visual</h2>
  {_key_frame_gallery(case_dir, rgb_frames, include_descriptions=True, run_dir=run_dir)}
  <details open>
    <summary>All RGB Frames</summary>
    {_rgb_gallery(case_dir, rgb_frames)}
  </details>
</section>
<section>
  <h2>Timeline</h2>
  {_timeline(case_dir, actions, rgb_frames)}
</section>
<section>
  <h2>Action Planner Debug</h2>
  {_planner_debug(case_dir, planner_debug)}
</section>
<section>
  <h2>Phase Judge Debug</h2>
  {_phase_transition_events_panel(phase_events)}
  {_phase_judge_debug(case_dir, phase_debug)}
</section>
<section>
  <h2>Messages</h2>
  {_messages(messages)}
</section>
""",
    )


def _key_frame_gallery(
    base: Path,
    frames: list[Path],
    *,
    include_descriptions: bool = False,
    run_dir: Path | None = None,
) -> str:
    if not frames:
        return '<p class="empty">No RGB frames found.</p>'
    initial_frame = _anchored_rgb_frame(run_dir, "phase_start") if run_dir is not None else None
    if initial_frame is None:
        initial_frame = _anchored_rgb_frame(run_dir, "task_initial") if run_dir is not None else None
    if initial_frame is None:
        initial_frame = frames[0]
    picks: list[tuple[str, str, Path]] = [
        ("Initial", "First RGB observation for the current semantic phase", initial_frame)
    ]
    if len(frames) > 1 and frames[1] not in {item[2] for item in picks}:
        picks.append(("After first action", "Observation after action #0", frames[1]))
    if len(frames) > 2 and frames[-1] not in {item[2] for item in picks}:
        picks.append(("Final", "Final observation", frames[-1]))
    return (
        '<div class="frames key">'
        + "\n".join(
            _frame_card(base, frame, label, description if include_descriptions else None)
            for label, description, frame in picks
        )
        + "</div>"
    )


def _rgb_gallery(base: Path, frames: list[Path]) -> str:
    if not frames:
        return '<p class="empty">No RGB frames found.</p>'
    return '<div class="frames all">' + "\n".join(_frame_card(base, frame, frame.name) for frame in frames) + "</div>"


def _frame_card(base: Path, frame: Path, label: str, description: str | None = None) -> str:
    href = _href(base, frame)
    caption = f"{_e(label)} - <code>{_e(frame.name)}</code>"
    if description:
        caption += f"<br><span>{_e(description)}</span>"
    return f"""
<figure>
  <a href="{href}"><img src="{href}" alt="{_e(label)}"></a>
  <figcaption>{caption}</figcaption>
</figure>
"""


def _timeline(base: Path, actions: list[Any], frames: list[Path]) -> str:
    if not actions:
        return '<p class="empty">No tool actions recorded.</p>'
    rows = []
    display_actions = _display_actions(actions)
    for index, action in enumerate(display_actions):
        if not isinstance(action, dict):
            continue
        before, after = _action_frame_pair(frames, index)
        rows.append(
            """
<tr>
  <td>{idx}</td><td><code>{tool}</code></td><td>{phase}</td><td><code>{args}</code></td>
  <td><code>{actual}</code></td><td>{ok}</td><td>{before}</td><td>{after}</td>
</tr>
""".format(
                idx=_e(action.get("step_index", index)),
                tool=_e(action.get("tool")),
                phase=_e(action.get("phase")),
                args=_e(json.dumps(action.get("args", {}), ensure_ascii=False)),
                actual=_e(json.dumps(action.get("actual_delta_m"), ensure_ascii=False)),
                ok=_e(action.get("ok")),
                before=_thumb_link(base, before),
                after=_thumb_link(base, after),
            )
        )
    return """
<table>
  <thead><tr><th>#</th><th>Tool</th><th>Phase</th><th>Args</th><th>Actual delta</th><th>OK</th><th>Before RGB</th><th>After RGB</th></tr></thead>
  <tbody>{}</tbody>
</table>
""".format("\n".join(rows))


def _messages(messages: list[Any]) -> str:
    if not messages:
        return '<p class="empty">No agent messages recorded.</p>'
    items = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        body = item.get("message", {}) if isinstance(item.get("message"), dict) else {}
        ref = item.get("message_ref", {}) if isinstance(item.get("message_ref"), dict) else {}
        items.append(
            """
<article class="message">
  <div class="message-meta">
    <span>{source}</span><span>{phase}</span><span>{confidence}</span><span>{timestamp}</span>
  </div>
  <p>{content}</p>
</article>
""".format(
                source=_e(item.get("source")),
                phase=_e(item.get("phase")),
                confidence=_e(body.get("confidence")),
                timestamp=_e(ref.get("timestamp") or item.get("timestamp")),
                content=_e(body.get("content")),
            )
        )
    return "\n".join(items) or '<p class="empty">No agent messages recorded.</p>'


def _action_preview(base: Path, actions: list[Any], frames: list[Path]) -> str:
    if not actions:
        return '<p class="empty">No tool actions recorded.</p>'
    cards = []
    for index, action in enumerate(actions[:8]):
        if not isinstance(action, dict):
            continue
        before, after = _action_frame_pair(frames, index)
        cards.append(
            """
<article class="action-card">
  <div class="action-card-head">
    <span>#{idx}</span>
    <code>{tool}</code>
    <span>{ok}</span>
  </div>
  <div class="action-meta">
    <span>{phase}</span>
    <code>{args}</code>
  </div>
  <div class="action-frames">
    <div><p>Before</p>{before}</div>
    <div><p>After</p>{after}</div>
  </div>
</article>
""".format(
                idx=_e(action.get("step_index")),
                tool=_e(action.get("tool")),
                phase=_e(action.get("phase")),
                args=_e(json.dumps(action.get("args", {}), ensure_ascii=False)),
                before=_thumb_link(base, before),
                after=_thumb_link(base, after),
                ok=_e(action.get("ok")),
            )
        )
    more = "" if len(actions) <= 8 else f'<p class="empty">Showing first 8 of {len(actions)} actions. Open case review for all.</p>'
    return '<div class="action-cards">' + "\n".join(cards) + "</div>" + more


def _message_preview(messages: list[Any]) -> str:
    if not messages:
        return '<p class="empty">No agent messages recorded.</p>'
    items = []
    for item in messages[:4]:
        if not isinstance(item, dict):
            continue
        body = item.get("message", {}) if isinstance(item.get("message"), dict) else {}
        items.append(
            """
<article class="message compact-message">
  <div class="message-meta"><span>{source}</span><span>{phase}</span><span>{confidence}</span></div>
  <p>{content}</p>
</article>
""".format(
                source=_e(item.get("source")),
                phase=_e(item.get("phase")),
                confidence=_e(body.get("confidence")),
                content=_e(body.get("content")),
            )
        )
    if len(messages) > 4:
        items.append(f'<p class="empty">Showing first 4 of {len(messages)} messages. Open case review for all.</p>')
    return "\n".join(items)


def _artifact_links(case_dir: Path, result: dict[str, Any], run_dir: Path | None, plot: dict[str, Any]) -> str:
    links = [
        _link(case_dir, case_dir / "review.md", "review.md"),
        _link(case_dir, case_dir / "case.json", "case.json"),
        _link(case_dir, case_dir / "ndtw.json", "ndtw.json"),
        _link(case_dir, case_dir / "sr_judge.json", "sr_judge.json"),
    ]
    for key, label in [("trajectory_2d", "trajectory_2d"), ("trajectory_3d", "trajectory_3d")]:
        if plot.get(key):
            links.append(_link(case_dir, _path_from_result(plot.get(key), None), label))
    if run_dir is not None:
        links.extend(
            [
                _link(case_dir, run_dir, "run dir"),
                _link(case_dir, run_dir / "memory" / "actions.jsonl", "actions.jsonl"),
                _link(case_dir, run_dir / "memory" / "agent_messages.jsonl", "agent_messages.jsonl"),
                _link(case_dir, run_dir / "artifacts" / "rgb", "rgb folder"),
                _link(case_dir, run_dir / "debug" / "action_planner", "action planner debug"),
            ]
        )
    review_path = result.get("review_path")
    if review_path:
        links.append(_link(case_dir, _path_from_result(review_path, None), "summary review path"))
    return "\n".join(links)


def _rgb_frames(result: dict[str, Any], artifacts: dict[str, Any]) -> list[Path]:
    run_dir = _path_from_result(result.get("run_dir"), None)
    frames: list[Path] = []
    if run_dir is not None:
        frames = sorted((run_dir / "artifacts" / "rgb").glob("*.png"))
    if frames:
        return frames
    for value in [artifacts.get("first_rgb"), artifacts.get("last_rgb")]:
        path = _path_from_result(value, None)
        if path is not None and path not in frames:
            frames.append(path)
    return frames


def _anchored_rgb_frame(run_dir: Path | None, anchor_name: str) -> Path | None:
    if run_dir is None:
        return None
    memory = _read_json(run_dir / "state" / "flight_memory.json")
    anchors = memory.get("observation_anchors", {}) if isinstance(memory.get("observation_anchors"), dict) else {}
    observation = _read_memory_ref(run_dir, anchors.get(anchor_name))
    if not observation:
        return None
    frame_refs = observation.get("frame_refs", []) if isinstance(observation.get("frame_refs"), list) else []
    for frame_ref in frame_refs:
        frame = _read_memory_ref(run_dir, frame_ref)
        if frame.get("modality") != "rgb":
            continue
        raw_path = frame.get("path")
        path = Path(str(raw_path)) if raw_path not in (None, "") else None
        if path is not None and not path.is_absolute():
            path = run_dir / path
        if path is None or not path.exists():
            path = _path_from_result(raw_path, None)
        if path is not None and path.exists():
            return path
    return None


def _read_memory_ref(run_dir: Path, ref: Any) -> dict[str, Any]:
    ref_value = ref.get("ref") if isinstance(ref, dict) else None
    if not isinstance(ref_value, str) or "#" not in ref_value:
        return {}
    relative_path, line_text = ref_value.split("#", 1)
    try:
        line_number = int(line_text)
    except ValueError:
        return {}
    path = run_dir / relative_path
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if line_number < 1 or line_number > len(lines):
        return {}
    try:
        value = json.loads(lines[line_number - 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _display_actions(actions: Any) -> list[dict[str, Any]]:
    """Return benchmark motion actions that should be paired with RGB frames."""

    if not isinstance(actions, list):
        return []
    return [
        dict(action)
        for action in actions
        if isinstance(action, dict) and action.get("tool") in {"uav.turn_relative", "uav.move_body"}
    ]


def _action_frame_pair(frames: list[Path], action_index: int) -> tuple[Path | None, Path | None]:
    """Map action N to the observation before and after that action."""

    before = frames[action_index] if action_index < len(frames) else None
    after_index = action_index + 1
    after = frames[after_index] if after_index < len(frames) else None
    return before, after


def _agent_messages(run_dir: Path | None, artifacts: dict[str, Any]) -> list[Any]:
    """Prefer raw UTF-8 memory records over summary-cached message text."""

    if run_dir is not None:
        records = _read_jsonl(run_dir / "memory" / "agent_messages.jsonl")
        if records:
            return records
    messages = artifacts.get("agent_messages", [])
    return messages if isinstance(messages, list) else []


def _action_planner_debug_records(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    debug_dir = run_dir / "debug" / "action_planner"
    if not debug_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(debug_dir.glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _phase_transition_debug_records(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    debug_dir = run_dir / "debug" / "phase_transition"
    if not debug_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(debug_dir.glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _phase_transition_events(run_dir: Path | None) -> list[dict[str, Any]]:
    """Return runtime transition events that represent the applied phase state."""

    if run_dir is None:
        return []
    names = {
        "model_phase_transition_judged",
        "phase_transition_applied",
        "phase_transition_continued",
    }
    records = []
    for item in _read_jsonl(run_dir / "events.jsonl"):
        event = item.get("event") or item.get("kind")
        if event in names:
            records.append({"event": event, "payload": item.get("payload", {})})
    return records


def _planner_debug(base: Path, records: list[dict[str, Any]]) -> str:
    if not records:
        return '<p class="empty">Action planner debug was not enabled for this run.</p>'
    items = []
    for record in records:
        context = record.get("planner_context", {}) if isinstance(record.get("planner_context"), dict) else {}
        model_context = record.get("model_context") if isinstance(record.get("model_context"), dict) else context
        model_context_text = _model_facing_context_text(model_context)
        fragments = {item.get("name"): item for item in model_context.get("fragments", []) if isinstance(item, dict)}
        image = fragments.get("image", {})
        action = fragments.get("action", {})
        raw_plan = record.get("raw_plan", {}) if isinstance(record.get("raw_plan"), dict) else {}
        safe_plan = record.get("safe_plan", {}) if isinstance(record.get("safe_plan"), dict) else {}
        safety = record.get("safety_decision", {}) if isinstance(record.get("safety_decision"), dict) else {}
        items.append(
            """
<details class="planner-debug">
  <summary>Window {window} · {phase} · raw tools {raw_tools} · safe tools {safe_tools}</summary>
  <div class="debug-grid">
    <article>
      <h3>Model-Facing Context</h3>
      <pre>{model_context}</pre>
      <h3>Context Summary</h3>
      <dl class="kv">
        {_active}
        {_phase}
        {_metrics}
        {_exec}
        {_image_roles}
        {_action_trend}
      </dl>
      <div class="links compact">{artifact}</div>
    </article>
    <article>
      <h3>Planner Output</h3>
      <p class="empty">Planner output is only the current action window; phase switching is decided by Phase Judge events.</p>
      <h4>Raw plan</h4>
      {_plan_raw}
      <h4>Safety-reviewed plan</h4>
      {_plan_safe}
      <h4>Safety</h4>
      <pre>{safety}</pre>
    </article>
  </div>
</details>
""".format(
                window=_e(record.get("window_index")),
                phase=_e(record.get("phase")),
                raw_tools=_e(_plan_tools(raw_plan)),
                safe_tools=_e(_plan_tools(safe_plan)),
                _active=_kv("Instruction", context.get("active_instruction")),
                _phase=_kv("Current phase", (context.get("current_phase") or {}).get("name") if isinstance(context.get("current_phase"), dict) else context.get("current_phase")),
                _metrics=_kv("Phase metrics", json.dumps(context.get("phase_metrics", []), ensure_ascii=False)),
                _exec=_kv("Phase execution summary", json.dumps(_compact_phase_execution(context.get("phase_execution_summary", {})), ensure_ascii=False)),
                _image_roles=_kv("Image model inputs", json.dumps(_fragment_model_input_roles(model_context), ensure_ascii=False)),
                _action_trend=_kv("Action trend", json.dumps((action.get("content", {}) if isinstance(action.get("content"), dict) else {}).get("action_trend", {}), ensure_ascii=False)),
                artifact=_link(base, _path_from_result(record.get("_path"), None), "open full debug json"),
                model_context=_e(model_context_text),
                _plan_raw=_plan_block(raw_plan),
                _plan_safe=_plan_block(safe_plan),
                safety=_e(json.dumps(_compact_safety(safety), ensure_ascii=False, indent=2)),
            )
        )
    return "\n".join(items)


def _phase_transition_events_panel(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="empty">No runtime phase transition events recorded.</p>'
    rows = []
    for index, item in enumerate(events):
        payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
        rows.append(
            """
<tr>
  <td>{idx}</td><td><code>{event}</code></td><td>{phase}</td><td>{decision}</td><td>{reason}</td>
</tr>
""".format(
                idx=_e(index),
                event=_e(item.get("event")),
                phase=_e(payload.get("phase")),
                decision=_e(payload.get("applied_decision") or payload.get("decision")),
                reason=_e(payload.get("guard_reason") or payload.get("reason")),
            )
        )
    return """
<article>
  <h3>Runtime Phase Transition Events</h3>
  <p class="empty">These events, not planner action_window fields, represent phase continuation or switching.</p>
  <table class="compact-table">
    <thead><tr><th>#</th><th>Event</th><th>Phase</th><th>Decision</th><th>Reason</th></tr></thead>
    <tbody>{}</tbody>
  </table>
</article>
""".format("\n".join(rows))


def _phase_judge_debug(base: Path, records: list[dict[str, Any]]) -> str:
    if not records:
        return '<p class="empty">Phase transition judge debug was not enabled for this run.</p>'
    items = []
    for record in records:
        model_context = record.get("model_context") if isinstance(record.get("model_context"), dict) else record
        model_context_text = _model_facing_context_text(model_context)
        action_plan = record.get("action_plan", {}) if isinstance(record.get("action_plan"), dict) else {}
        executed_steps = record.get("executed_steps", []) if isinstance(record.get("executed_steps"), list) else []
        summary = record.get("phase_execution_summary", {}) if isinstance(record.get("phase_execution_summary"), dict) else {}
        items.append(
            """
<details class="planner-debug">
  <summary>Window {window} · {phase} · executed tools {tools}</summary>
  <div class="debug-grid">
    <article>
      <h3>Model-Facing Context</h3>
      <pre>{model_context}</pre>
      <h3>Judge Context Summary</h3>
      <dl class="kv">
        {_active}
        {_phase}
        {_metrics}
        {_action_window}
        {_exec}
        {_measurements}
        {_safety}
      </dl>
      <div class="links compact">{artifact}</div>
    </article>
    <article>
      <h3>Executed Window</h3>
      <h4>Action plan</h4>
      <pre>{action_plan}</pre>
      <h4>Executed steps</h4>
      <pre>{executed_steps}</pre>
      <h4>Observation roles</h4>
      <pre>{image_roles}</pre>
    </article>
  </div>
</details>
""".format(
                window=_e(record.get("phase_window_count")),
                phase=_e((record.get("current_phase") or {}).get("name") if isinstance(record.get("current_phase"), dict) else record.get("current_phase")),
                tools=_e(_executed_tools(executed_steps)),
                model_context=_e(model_context_text),
                _active=_kv("Instruction", record.get("active_instruction")),
                _phase=_kv("Current phase", (record.get("current_phase") or {}).get("name") if isinstance(record.get("current_phase"), dict) else record.get("current_phase")),
                _metrics=_kv("Phase metrics", json.dumps(record.get("phase_metrics", []), ensure_ascii=False)),
                _action_window=_kv("Planner action_window (debug-only)", json.dumps(record.get("action_window", {}), ensure_ascii=False)),
                _exec=_kv("Phase execution summary", json.dumps(_compact_phase_execution(summary), ensure_ascii=False)),
                _measurements=_kv("Measurements", json.dumps(record.get("measurements", {}), ensure_ascii=False)),
                _safety=_kv("Safety summary", json.dumps(_compact_phase_safety(record), ensure_ascii=False)),
                artifact=_link(base, _path_from_result(record.get("_path"), None), "open full phase judge json"),
                action_plan=_e(json.dumps(action_plan, ensure_ascii=False, indent=2)),
                executed_steps=_e(json.dumps(_compact_executed_steps(executed_steps), ensure_ascii=False, indent=2)),
                image_roles=_e(json.dumps(_fragment_model_input_roles(record), ensure_ascii=False, indent=2)),
            )
        )
    return "\n".join(items)


def _model_facing_context_text(context: dict[str, Any]) -> str:
    """Return the exact text JSON portion that the action planner model sees."""

    if not context:
        return "missing planner_context"
    user_content = build_model_user_content(context)
    if isinstance(user_content, str):
        return _pretty_json_text(user_content)
    text_parts = [
        str(part.get("text", ""))
        for part in user_content
        if isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip()
    ]
    return "\n\n--- text model input ---\n\n".join(_pretty_json_text(text) for text in text_parts) or "missing text payload"


def _pretty_json_text(text: str) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(value, ensure_ascii=False, indent=2)


def _plan_tools(plan: dict[str, Any]) -> list[str]:
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    return [str(item.get("tool")) for item in steps if isinstance(item, dict) and item.get("tool")]


def _executed_tools(executed_steps: list[Any]) -> list[str]:
    tools = []
    for item in executed_steps:
        step = item.get("step", {}) if isinstance(item, dict) else {}
        if isinstance(step, dict) and step.get("tool"):
            tools.append(str(step.get("tool")))
    return tools


def _compact_phase_safety(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "safety_intervention_summary": record.get("safety_intervention_summary"),
        "visual_phase_budget_exhausted": record.get("visual_phase_budget_exhausted"),
        "recent_executed_tools": record.get("recent_executed_tools"),
        "recent_safety_reasons": record.get("recent_safety_reasons"),
    }


def _compact_executed_steps(executed_steps: list[Any]) -> list[dict[str, Any]]:
    compact = []
    for item in executed_steps:
        if not isinstance(item, dict):
            continue
        step = item.get("step", {}) if isinstance(item.get("step"), dict) else {}
        result = item.get("result", {}) if isinstance(item.get("result"), dict) else {}
        compact.append(
            {
                "phase": item.get("phase"),
                "step_index": item.get("step_index"),
                "tool": step.get("tool"),
                "args": step.get("args"),
                "ok": result.get("ok"),
                "message": result.get("message"),
            }
        )
    return compact


def _fragment_model_input_roles(record: dict[str, Any]) -> list[dict[str, Any]]:
    roles = []
    for fragment in record.get("fragments", []):
        if not isinstance(fragment, dict):
            continue
        model_inputs = fragment.get("model_inputs", [])
        if not isinstance(model_inputs, list):
            continue
        for item in model_inputs:
            if isinstance(item, dict):
                roles.append(
                    {
                        "fragment": fragment.get("name"),
                        "type": item.get("type"),
                        "input_id": item.get("input_id"),
                        "role": item.get("role"),
                        "usage": item.get("usage"),
                    }
                )
    return roles


def _plan_block(plan: dict[str, Any]) -> str:
    if not plan:
        return '<p class="empty">missing</p>'
    payload = {
        "action_window": plan.get("action_window"),
        "agent_messages": plan.get("agent_messages"),
        "steps": plan.get("steps"),
    }
    return f"<pre>{_e(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>"


def _compact_phase_execution(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "phase": summary.get("phase"),
        "phase_key": summary.get("phase_key"),
        "window_count": summary.get("window_count"),
        "max_window_count": summary.get("max_window_count"),
        "measurements": summary.get("measurements"),
        "last_window": summary.get("last_window"),
        "tool_facts": summary.get("tool_facts"),
    }


def _compact_safety(safety: dict[str, Any]) -> dict[str, Any]:
    return {
        "interventions": safety.get("interventions"),
        "plan_changed": safety.get("plan_changed"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _action_counts(actions: Any) -> dict[str, int]:
    counts = {"uav.turn_relative": 0, "uav.move_body": 0, "uav.hover": 0}
    if not isinstance(actions, list):
        return counts
    for action in actions:
        if isinstance(action, dict) and action.get("tool") in counts:
            counts[str(action.get("tool"))] += 1
    return counts


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{ color-scheme: light; --ink: #1d2430; --muted: #637083; --line: #d9e0ea; --bg: #f7f9fc; --panel: #ffffff; --accent: #0b6f85; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; color: var(--ink); background: var(--bg); }}
    header.hero, section, nav.top-nav {{ max-width: 1280px; margin: 0 auto; }}
    header.hero {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; padding: 32px 24px 18px; }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; letter-spacing: 0; }}
    p {{ line-height: 1.55; }}
    .eyebrow {{ margin: 0 0 6px; color: var(--accent); font-weight: 700; text-transform: uppercase; font-size: 12px; }}
    .instruction {{ margin: 8px 0 0; color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(110px, 1fr)); gap: 10px; min-width: 360px; }}
    .metrics div, section article, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .metrics div {{ padding: 14px; }}
    .metrics span, .message-meta, figcaption, .empty {{ color: var(--muted); font-size: 12px; }}
    .metrics strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    section {{ margin-top: 18px; padding: 18px; overflow-x: auto; }}
    .grid {{ display: grid; gap: 18px; background: transparent; border: 0; padding: 0 24px; }}
    .grid.two {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    .grid article {{ padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #344054; background: #eef3f8; position: sticky; top: 0; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
    a {{ color: #075f73; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #e8f4f7; color: #075f73; font-weight: 700; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .links a, td a {{ display: inline-block; margin: 2px 4px 2px 0; }}
    .stat-list {{ list-style: none; padding: 0; margin: 0; }}
    .stat-list li {{ display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--line); }}
    .kv {{ display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 14px; margin: 0; }}
    .kv dt {{ color: var(--muted); }}
    .kv dd {{ margin: 0; overflow-wrap: anywhere; }}
    .frames {{ display: grid; gap: 14px; }}
    .frames.key {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 18px; }}
    .frames.all {{ grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); margin-top: 14px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; height: auto; image-rendering: auto; }}
    figcaption {{ padding: 8px; }}
    .thumb {{ width: 88px; border: 1px solid var(--line); border-radius: 6px; }}
    .message {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 10px 0; background: #fff; }}
    .message p {{ margin: 8px 0 0; }}
    .message-meta {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8fb; border: 1px solid var(--line); border-radius: 8px; padding: 10px; max-height: 360px; overflow: auto; font-size: 12px; }}
    details.planner-debug {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; margin: 10px 0; padding: 12px; }}
    details.planner-debug summary {{ cursor: pointer; font-weight: 700; color: #344054; }}
    .debug-grid {{ display: grid; grid-template-columns: minmax(0, 0.46fr) minmax(0, 0.54fr); gap: 14px; margin-top: 12px; }}
    .debug-grid article {{ padding: 12px; }}
    .debug-grid h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .debug-grid h4 {{ margin: 12px 0 6px; font-size: 13px; color: #344054; }}
    .case-cards {{ display: grid; gap: 18px; }}
    .case-card {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 16px; }}
    .case-card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 12px; }}
    .case-card h3 {{ margin: 0; font-size: 20px; letter-spacing: 0; }}
    .case-instruction {{ margin: 8px 0 0; color: #344054; font-size: 14px; line-height: 1.45; max-width: 760px; }}
    .case-card h4 {{ margin: 0 0 10px; font-size: 14px; color: #344054; }}
    .case-actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .mini-kv {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin: 0 0 14px; }}
    .mini-kv dt {{ color: var(--muted); font-size: 12px; }}
    .mini-kv dd {{ margin: 2px 0 0; font-weight: 700; overflow-wrap: anywhere; }}
    .evidence-grid {{ display: grid; grid-template-columns: minmax(320px, 0.42fr) minmax(0, 0.58fr); grid-template-areas: "rgb actions" "rgb messages"; gap: 18px; align-items: start; }}
    .rgb-panel {{ grid-area: rgb; min-width: 0; }}
    .actions-panel {{ grid-area: actions; min-width: 0; }}
    .messages-panel {{ grid-area: messages; min-width: 0; }}
    .case-card .frames.key {{ grid-template-columns: 1fr; }}
    .case-card .frames.key figure {{ max-width: 520px; }}
    .compact-table th, .compact-table td {{ padding: 6px 7px; font-size: 12px; }}
    .action-cards {{ display: grid; gap: 10px; }}
    .action-card {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; padding: 10px; }}
    .action-card-head {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; font-size: 12px; color: var(--muted); }}
    .action-meta {{ display: grid; grid-template-columns: minmax(120px, max-content) minmax(0, 1fr); gap: 10px; margin-top: 8px; font-size: 12px; }}
    .action-meta code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .action-frames {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }}
    .action-frames p {{ margin: 0 0 5px; color: var(--muted); font-size: 12px; line-height: 1.2; }}
    .action-frames .thumb {{ width: 100%; max-width: 150px; }}
    .compact-message {{ padding: 9px; margin: 8px 0; }}
    .compact-message p {{ font-size: 13px; }}
    .links.compact {{ margin-top: 8px; }}
    .top-nav {{ padding: 18px 24px 0; }}
    @media (max-width: 1040px) {{ .evidence-grid {{ grid-template-columns: 1fr; grid-template-areas: "rgb" "actions" "messages"; }} .case-card .frames.key figure {{ max-width: none; }} .debug-grid {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 860px) {{ header.hero {{ display: block; }} .metrics {{ min-width: 0; margin-top: 16px; }} .grid.two {{ grid-template-columns: 1fr; }} .case-card-head {{ display: block; }} .case-actions {{ justify-content: flex-start; margin-top: 10px; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_from_result(value: Any, default: Path | None) -> Path | None:
    if value in (None, ""):
        return default
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _href(from_dir: Path, target: Path | None) -> str:
    if target is None:
        return "#"
    try:
        rel = os.path.relpath(str(target), str(from_dir))
    except ValueError:
        rel = str(target)
    return quote(rel.replace("\\", "/"), safe="/:#?&=%.")


def _link(from_dir: Path, target: Path | None, label: str) -> str:
    if target is None:
        return f'<span class="empty">{_e(label)} missing</span>'
    return f'<a href="{_href(from_dir, target)}">{_e(label)}</a>'


def _thumb_link(base: Path, frame: Path | None) -> str:
    if frame is None:
        return '<span class="empty">missing</span>'
    href = _href(base, frame)
    return f'<a href="{href}"><img class="thumb" src="{href}" alt="{_e(frame.name)}"></a>'


def _badge(value: Any) -> str:
    return f'<span class="badge">{_e(value)}</span>'


def _sr_label(sr: dict[str, Any]) -> str:
    if not sr:
        return ""
    success = sr.get("success")
    status = sr.get("status")
    confidence = sr.get("confidence")
    text = f"{status} / success={success}"
    if confidence is not None:
        text += f" / conf={confidence}"
    return _e(text)


def _visual_summary(visual: dict[str, Any]) -> str:
    if not visual:
        return ""
    return f"{visual.get('status')} enabled={visual.get('enabled')} judged={visual.get('judged_count')}/{visual.get('frame_count')}"


def _kv(label: str, value: Any) -> str:
    return f"<dt>{_e(label)}</dt><dd>{_e(value)}</dd>"


def _e(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)
