"""Offline image-history projection preview; never calls Provider or simulator.

Example (run from repository root):
  python tests/manual/history_image_sampling_probe.py --case-dir <case> --output-dir <new-dir>

Reads frozen audit.json plus its exact captured provider media. Each assistant
or rejected provider event delimits the preceding request. Missing captures,
unsafe paths or mismatched IDs are reported, never reconstructed from future
state. The output is a simulation, NOT a historical provider-request rewrite.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.benchmarks.core.report_io import atomic_write_text
from evaluation.benchmarks.core.review import history_projection_review
from src.agent.history_projection import HistoryProjector
from src.agent.planning.model_action_planner import _snapshot_request
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult, TranscriptEvent
from src.memory.context.registry import ContextModuleRegistry
from src.tools.registry import default_registry


def _event(record: dict) -> TranscriptEvent:
    """Decode existing durable payloads without a writer or Runtime instance."""
    kind, payload = record["kind"], copy.deepcopy(record["payload"])
    if kind == "assistant_turn":
        payload["tool_calls"] = [AgentToolCall(**call) for call in payload.get("tool_calls", [])]
        payload = AssistantTurn(**payload)
    elif kind == "tool_result":
        allowed = {field.name for field in fields(ToolResult)}
        payload = ToolResult(**{key: value for key, value in payload.items() if key in allowed})
    return TranscriptEvent(kind, payload)


def _captured_media(model_round: dict, run_dir: Path, output_dir: Path):
    """Resolve only this request's captured files, with containment and SHA checks."""
    observations, tools, errors = {}, {}, []
    for media in model_round.get("request_media", []):
        try:
            asset = (run_dir / media["asset_ref"]).resolve()
            if not asset.is_relative_to(run_dir) or not asset.is_file():
                raise ValueError("missing or unsafe captured asset")
            data = asset.read_bytes()
            if hashlib.sha256(data).hexdigest() != media["sha256"]:
                raise ValueError("captured asset SHA mismatch")
            # File type is from validated captured metadata, not an arbitrary suffix.
            suffix = {"image/png": ".png", "image/jpeg": ".jpg"}.get(media.get("mime_type"))
            if suffix is None:
                raise ValueError("unsupported captured MIME type")
            dest = output_dir / "review_assets" / (media["sha256"] + suffix)
            if not dest.exists():
                shutil.copyfile(asset, dest)
            value = {
                "data": f"data:{media['mime_type']};base64," + base64.b64encode(data).decode("ascii"),
                "path": str(dest), "relative": dest.relative_to(output_dir).as_posix(),
                "sha256": media["sha256"],
            }
            if media.get("timeline_role") == "TOOL_EVIDENCE":
                tools.setdefault(media.get("tool_call_id"), []).append(value)
            else:
                key = (media.get("observation_id"), media.get("frame_id"))
                if key in observations and observations[key]["sha256"] != value["sha256"]:
                    raise ValueError("ambiguous captured observation image")
                observations[key] = value
        except (KeyError, OSError, ValueError) as exc:
            errors.append(str(exc))
    return observations, tools, errors


def _rehydrate(events: list[TranscriptEvent], observations: dict, tools: dict):
    """Use exact request images, never re-render RGB/depth with today's code."""
    hydrated = copy.deepcopy(events)
    errors = []
    for event in hydrated:
        if event.kind == "state_checkpoint":
            for fragment in event.payload["context"].get("fragments", []):
                for item in fragment.get("model_inputs", []):
                    if item.get("type") != "image":
                        continue
                    key = (item.get("observation_id") or event.payload.get("observation_id"), item.get("frame_id"))
                    captured = observations.get(key)
                    if captured:
                        item["data"] = captured["data"]
                        item["encoding"] = "data_url"
                    else:
                        item.pop("data", None)
                        errors.append(f"No exact captured image for {key}")
        elif event.kind == "tool_result" and event.payload.model_inputs:
            captured = tools.get(event.payload.tool_call_id, [])
            inputs = event.payload.model_inputs
            expected = [item for item in inputs if item.get("type") == "image"]
            if len(captured) != len(expected):
                errors.append(f"No exact tool media mapping for {event.payload.tool_call_id}")
            for index, item in enumerate(expected):
                item["path"] = captured[index]["path"] if index < len(captured) else ""
                item.pop("artifact_ref", None)
    return tuple(hydrated), errors


def run_probe(case_dir: Path, output_dir: Path, config: Path) -> dict:
    source = case_dir / "audit.json"
    original = source.read_bytes()
    audit = json.loads(original)
    if output_dir.exists():
        raise ValueError("Output directory must be new; existing reviews are never overwritten")
    if output_dir.resolve().is_relative_to(case_dir.resolve()):
        raise ValueError("Preview must be outside the existing case directory")
    # Validate configuration before creating any preview artifacts.
    registry = ContextModuleRegistry.from_config(config)
    projector = HistoryProjector(registry.history_bindings())
    tool_registry = default_registry()
    run_dir = (REPO / audit["run_dir"]).resolve()
    output_dir.mkdir(parents=True)
    (output_dir / "review_assets").mkdir()
    rounds = [r for window in audit.get("planner_debug", []) for r in window.get("model_rounds", [])]
    pending, reports, sections = [], [], []
    request_index = 0
    for record in audit.get("agent_transcript", []):
        if record["kind"] in {"assistant_turn", "provider_turn_rejected"}:
            if request_index >= len(rounds):
                reports.append({"error": "Missing captured request for transcript turn", "event_id": record.get("id")})
                break
            model_round = rounds[request_index]
            request_index += 1
            observations, tools, errors = _captured_media(model_round, run_dir, output_dir)
            canonical = model_round.get("canonical_turn", {})
            if record["kind"] == "assistant_turn" and canonical.get("turn_id") != record["payload"].get("turn_id"):
                errors.append("Captured request/canonical turn alignment mismatch")
            if record["kind"] == "provider_turn_rejected" and canonical:
                errors.append("Rejected provider event aligned to a canonical assistant turn")
            events, missing = _rehydrate(pending, observations, tools)
            errors.extend(missing)
            full = HistoryProjector().project_snapshot(events, tool_registry)
            selected = projector.project_snapshot(events, tool_registry)
            if [m for m in full.messages if m["role"] in {"assistant", "tool"}] != [m for m in selected.messages if m["role"] in {"assistant", "tool"}]:
                errors.append("Assistant/tool history changed")
            captured_count = sum(m.get("timeline_role") != "TOOL_EVIDENCE" for m in model_round.get("request_media", []))
            if full.audit["observation_image_count"] != captured_count:
                errors.append("Full replay image count differs from captured request")
            summary = {
                "request_index": request_index, "before_event_id": record.get("id"),
                "preview_only": True, "captured_observation_images": captured_count,
                "all_history_observation_images": full.audit["observation_image_count"],
                "projection": selected.audit, "errors": sorted(set(errors)),
            }
            reports.append(summary)
            snapshot, _ = _snapshot_request({"messages": selected.messages})
            atomic_write_text(output_dir / f"request_{request_index:03d}.json", json.dumps({**summary, "simulated_request": snapshot}, ensure_ascii=False, indent=2))
            ids = list(dict.fromkeys(e.payload.get("observation_id") for e in events if e.kind == "state_checkpoint"))
            selected_ids = set(selected.audit["policies"].get("image", {}).get("selected_observation_ids", ids))
            cards = []
            for obs in ids:
                matches = [v for (oid, _), v in observations.items() if oid == obs]
                role = "CURRENT" if obs == ids[-1] else "HISTORY"
                label = f"{role} {obs} - {'selected' if obs in selected_ids else 'omitted'}"
                thumbnail = '<span>Exact image unavailable</span>'
                if matches:
                    thumbnail = f'<img loading="lazy" src="{html.escape(matches[0]["relative"], quote=True)}" alt="{html.escape(label, quote=True)}">'
                cards.append(f'<figure class="{"selected" if obs in selected_ids else "omitted"}">{thumbnail}<figcaption>{html.escape(label)}</figcaption></figure>')
            sections.append(
                f'<details id="request-{request_index}" {"open" if request_index == len(rounds) else ""}><summary>Request {request_index}: {captured_count} → {selected.audit["observation_image_count"]} observation images</summary>'
                + history_projection_review(selected.audit, preview=True)
                + '<p>Complete timeline below: omitted thumbnails remain visible for HUMAN audit only.</p>'
                + '<div class="timeline">' + ''.join(cards) + '</div>'
                + '<pre>' + html.escape(json.dumps(summary["errors"], ensure_ascii=False)) + '</pre></details>'
            )
        pending.append(_event(record))
    count_matches = request_index == len(rounds)
    source_unchanged = source.read_bytes() == original
    report = {
        "preview_only": True, "source_case": str(case_dir.resolve()),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "source_unchanged": source_unchanged,
        "passed": source_unchanged and count_matches and bool(reports) and all(not r.get("errors") and "error" not in r for r in reports),
        "captured_request_count": len(rounds), "preview_request_count": request_index,
        "requests": reports,
    }
    atomic_write_text(output_dir / "sampling_audit.json", json.dumps(report, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Offline history image sampling</title>
<style>body{font:15px system-ui;margin:24px;color:#172033}details{margin:16px 0;padding:12px;border:1px solid #bbb}summary{cursor:pointer}.timeline{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;width:210px;padding:8px;border:2px solid #198754}figure img{width:100%;height:auto}.omitted{border-color:#aaa;opacity:.55}pre{white-space:pre-wrap}td,th{padding:5px;text-align:left}</style></head><body>
<h1>Offline sampling preview - NOT sent to model</h1>
<p>Observation selection follows the configured module policy below; tool evidence is additional. No Provider or simulator was called.</p>'''
    page += '<pre>' + html.escape(json.dumps(registry.history_policies, ensure_ascii=False, indent=2)) + '</pre>'
    atomic_write_text(output_dir / "review.html", page + ''.join(sections) + '</body></html>')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/plugins.json")
    args = parser.parse_args()
    report = run_probe(args.case_dir, args.output_dir, args.config)
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}, ensure_ascii=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
