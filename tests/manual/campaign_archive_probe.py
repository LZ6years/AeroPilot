"""Validate post-attempt archiving on copies without calling a model or flying.

Run with the configured flyagent interpreter:
  python tests/manual/campaign_archive_probe.py --output CAMPAIGN --destination DIAGNOSTICS
The destination must be new. Original attempts, scores and frozen files remain
unchanged. Probe records include exact image/request/transcript hashes, HTML
link checks, ZIP restoration checks and measured retained/working sizes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.benchmarks.core.campaign.artifacts import compact_attempt, validate_archive, restore_archive, PROFILE


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def size(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in {"href", "src"} and v)


def inspect_links(root):
    links, missing = 0, []
    for page in root.rglob("*.html"):
        parser = Links();parser.feed(page.read_text(encoding="utf-8"))
        for raw in parser.links:
            value = urlsplit(raw)
            if value.scheme or not value.path:
                continue
            target = (page.parent / unquote(value.path)).resolve()
            links += 1
            if not target.exists():missing.append(dict(page=str(page.relative_to(root)), link=raw))
    return dict(links=links, missing=missing)


def probe(source, destination):
    before_source = hashes(source)
    shutil.copytree(source, destination)
    before = hashes(destination)
    links_before = inspect_links(destination)
    record = compact_attempt(destination, profile=PROFILE)
    validate_archive(destination)
    archived = {v["path"] for v in record["members"]}
    remaining = hashes(destination)
    assert all(remaining.get(name) == digest for name, digest in before.items() if name not in archived)
    assert before_source == hashes(source), "Original attempt changed"
    restored = destination.parent / (destination.name + "-restore")
    restore_archive(destination, restored)
    assert hashes(restored) == {name: before[name] for name in archived}
    after_links = inspect_links(destination)
    assert after_links == links_before, "Archiving changed report links"
    assert not after_links["missing"], after_links["missing"]
    # These exact files are deliberately left outside the archive. They include
    # model-produced intents/calls/results, source observations and wire media
    # hashes/order/timing/usage. Byte equality is stronger than a field subset.
    evidence = {name: digest for name, digest in before.items() if "/memory/" in name or
                "/debug/model_call_metrics/" in name or Path(name).suffix.lower() in {".png", ".jpg", ".webp"}}
    assert all(remaining[name] == digest for name, digest in evidence.items())
    return dict(source=str(source), copy=str(destination), before_bytes=size(source), after_bytes=size(destination),
                archive_bytes=record["archive_bytes"], archived_members=len(archived),
                source_unchanged=True, retained_files_exact=True, restoration_exact=True,
                evidence_files_exact=len(evidence), image_files_exact=sum(Path(n).suffix == ".png" for n in evidence),
                model_request_receipts_exact=sum("/model_call_metrics/" in n for n in evidence),
                html_links=after_links["links"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root, destination = args.output.resolve(), args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    state = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    state_hash = hashlib.sha256((root / "progress.json").read_bytes()).hexdigest()
    rows = []
    for item in state["tasks"].values():
        for attempt in item["attempts"]:
            if attempt.get("logging_profile") != "paper_process_metrics_v1" or attempt.get("cleanup_verified") is not True:
                continue
            task = item["task"]
            path = root / task["dataset"] / task["split"] / "attempts" / task["episode_id"] / f'attempt_{attempt["number"]:02d}'
            rows.append((size(path), path, attempt))
    rows.sort(key=lambda row: (row[0], str(row[1])), reverse=True)
    assert rows, "No closed compact attempts to calibrate"
    selected = list(dict.fromkeys([i for i in range(min(8, len(rows)))] + [len(rows)//2, len(rows)-1]))
    checked, results = set(), []
    while selected:
        index = selected.pop(0)
        if index in checked:continue
        _, source, attempt = rows[index]
        result = probe(source, destination / f'case_{index:04d}')
        result.update(attempt_status=attempt.get("status"), temporary_depth_bytes=attempt.get("temporary_depth_bytes", 0))
        checked.add(index);results.append(result)
        print(json.dumps(result), flush=True)
        if not selected:
            seed = max(r["after_bytes"] for r in results)
            # A smaller untouched original cannot exceed this measured archive
            # seed except for its manifest/ZIP overhead. Verify all originals
            # larger than seed, rather than selecting cases by navigation score.
            selected = [i for i, row in enumerate(rows) if i not in checked and row[0] >= seed]
    assert hashlib.sha256((root / "progress.json").read_bytes()).hexdigest() == state_hash
    seed = max(r["after_bytes"] for r in results)
    report = dict(passed=True, profile=PROFILE, original_state_sha256=state_hash,
                  original_attempts=len(rows), probed_attempts=len(results), cases=results,
                  measured_seed_bytes=seed,
                  measured_peak_bytes=max(r["before_bytes"] + r["temporary_depth_bytes"] for r in results),
                  unprobed_max_original_bytes=max((row[0] for i, row in enumerate(rows) if i not in checked), default=0),
                  interpretation="Offline archive validation; no model calls, simulator actions or score changes")
    (destination / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}), flush=True)


if __name__ == "__main__":
    main()
