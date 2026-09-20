"""Offline compact-review preview of one frozen case; no Provider or simulator.

Usage: python tests/manual/rgb_phase_review_probe.py --case-dir CASE --output-dir NEW
The historical report keeps exact captured inputs. A separate raw-RGB page is
explicitly a new-mode preview, never a rewrite of what the model previously saw.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.benchmarks.core import BenchmarkAuditLog, BenchmarkCaseReview, BenchmarkReviewRenderer, FrameAssetResolver
from evaluation.benchmarks.core.report_io import atomic_write_text


def preview(case_dir: Path, output_dir: Path) -> dict:
    case_dir, output_dir = case_dir.resolve(), output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(case_dir):
        raise ValueError("Use a new independent output directory; never overwrite historical evidence")
    source = case_dir / "audit.json"
    original = source.read_bytes()
    data = json.loads(original)
    audit = BenchmarkAuditLog(**{f.name: data[f.name] for f in fields(BenchmarkAuditLog) if f.name in data})
    result = json.loads((case_dir / "case_result.json").read_text(encoding="utf-8"))
    destination = output_dir / "cases" / case_dir.name
    assets = FrameAssetResolver().materialize(audit, destination)
    instruction = next((r["payload"].get("instruction", "") for r in audit.agent_transcript
                        if r.get("kind") == "user_instruction"), "")
    if not instruction:
        instruction = next((r["model_context"].get("active_instruction", "") for r in audit.planner_debug
                            if isinstance(r.get("model_context"), dict)), "Instruction unavailable")
    review = BenchmarkCaseReview(
        case_id=case_dir.name, benchmark_name="Offline historical review preview",
        protocol_variant=result.get("protocol_variant", "unknown"), instruction=instruction,
        outcome={k: result.get(k) for k in ("status", "completed", "stopped", "stopped_reason", "error")},
        metrics={k: v for k, v in result.get("metrics", {}).items() if not isinstance(v, (list, dict))},
        audit=audit, frame_assets=assets,
    )
    paths = BenchmarkReviewRenderer().write_case(output_dir, review)
    cards = []
    for asset in assets:
        if asset.modality == "rgb" and asset.display_path:
            relative = Path(asset.display_path).relative_to(output_dir).as_posix()
            cards.append(f'<figure><img src="{html.escape(relative, quote=True)}" width="256">'
                         f'<figcaption>{html.escape(str(asset.observation_id))}</figcaption></figure>')
    atomic_write_text(output_dir / "rgb_mode_preview.html",
        '<!doctype html><html><meta charset="utf-8"><title>RGB mode preview</title>'
        '<h1>New RGB mode preview · NOT historical model input</h1>'
        '<p>Original recorded sensor RGB. No Provider request or simulator execution occurred.</p>'
        + ''.join(cards) + '</html>')
    report = {"preview_only": True, "source_case": str(case_dir),
              "source_unchanged": source.read_bytes() == original,
              "source_sha256": hashlib.sha256(original).hexdigest(), **paths}
    atomic_write_text(output_dir / "preview_audit.json", json.dumps(report, ensure_ascii=False, indent=2))
    atomic_write_text(output_dir / "review.html", '<!doctype html><html><meta charset="utf-8">'
        '<h1>Offline review preview</h1><p>No live model was called.</p>'
        f'<p><a href="cases/{html.escape(case_dir.name, quote=True)}/review.html">Historical case · captured inputs</a></p>'
        '<a href="rgb_mode_preview.html">Separate new RGB mode preview</a></html>')
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preview(args.case_dir, args.output_dir), ensure_ascii=True, indent=2))
