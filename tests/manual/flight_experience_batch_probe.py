"""Run a frozen historical subset, then inspect new logs with offline experience.

This manual orchestration owns only experiments. Production runners, safety,
model prompts and the independent experience module remain unchanged. Resume
never overwrites an attempt or repeats a completed navigation/model outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.manual import benchmark_20_case_probe as batch
from plugins.research.flight_experience import read_run

DEFAULT_SELECTION = (1, 4, 5, 7, 10, 12, 13, 17, 18, 19)
TERMINAL = {"completed", "model_error"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(manifest: dict, indices: list[int]) -> list[dict]:
    """Use original one-based positions, retaining exact IDs and metadata."""
    if len(indices) != 10 or len(set(indices)) != 10:
        raise ValueError("Exactly ten distinct source indices required")
    if any(i < 1 or i > len(manifest["cases"]) for i in indices):
        raise ValueError("Source index out of bounds")
    cases = [{**manifest["cases"][i - 1], "source_index": i} for i in indices]
    if Counter(c["benchmark"] for c in cases) != {"aerial": 5, "travel": 5}:
        raise ValueError("Five cases from each benchmark required")
    if len({c["case_id"] for c in cases}) != 10:
        raise ValueError("Duplicate case identity")
    return cases


def hashes() -> dict:
    """Include knowledge bodies/references omitted by the legacy hash helper."""
    return {**batch.source_hashes(), **{
        p.relative_to(ROOT).as_posix(): batch.sha256(p)
        for p in ROOT.glob("plugins/knowledge/**/*.md")}}


def guard(manifest: dict) -> None:
    if hashes() != manifest["complete_source_hashes"]:
        raise ValueError("Source drift: pause before scheduling another case/call")
    if batch.load_model_config(ROOT / ".env").redacted_summary() != manifest["model_config"]:
        raise ValueError("Model configuration drift")


def prepare(root: Path, source: Path, indices: list[int]) -> dict:
    if root.exists():
        raise ValueError("Output exists; frozen evidence must not be replaced")
    if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
        raise ValueError("Less than 10 GiB free")
    old = read(source)
    cases = select_cases(old, indices)
    config = batch.load_model_config(ROOT / ".env").redacted_summary()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model_name"], "model_config": config,
        "cases": cases, "source_manifest": str(source),
        "source_manifest_sha256": batch.sha256(source),
        "dataset_sha256": old["dataset_sha256"],
        "source_hashes": batch.source_hashes(), "complete_source_hashes": hashes(),
        "comparison_limitations": "Current working-tree changes are included; historical differences are not causal attribution."}
    root.mkdir(parents=True)
    snapshot = root / "snapshot"
    snapshot.mkdir()
    # No .env or private routing credentials are copied. Model config is redacted.
    for name in manifest["complete_source_hashes"]:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    for filename, command in (("git_head.txt", ["git", "rev-parse", "HEAD"]),
                              ("git_status.txt", ["git", "status", "--short"]),
                              ("working_tree.patch", ["git", "diff", "--binary", "HEAD"])):
        (snapshot / filename).write_bytes(subprocess.check_output(command, cwd=ROOT))
    batch.write_json(root / "manifest.json", manifest)
    batch.report(root, manifest)
    return manifest


def next_attempt(root: Path, index: int, item: dict, resume: bool) -> Path | None:
    """Classified clean interruptions may retry; unknown ownership always pauses."""
    directory = batch.latest_case_dir(root, index, item)
    if not directory.exists():
        return directory
    status_path = directory / "status.json"
    if not status_path.exists():
        raise ValueError(f"Unclassified existing attempt: {directory}")
    status = read(status_path)
    if status["status"] in TERMINAL and status.get("cleanup_verified"):
        return None
    if not resume or not status.get("cleanup_verified"):
        raise ValueError(f"Inspect interrupted attempt and verify cleanup: {directory}")
    original = batch.case_dir(root, index, item)
    number = len(list(original.parent.glob(original.name + "_recovery_*"))) + 1
    return original.with_name(original.name + f"_recovery_{number:02d}")


def fly(root: Path, resume: bool) -> None:
    manifest = read(root / "manifest.json")
    guard(manifest)
    batch.preflight(root, manifest)
    for index, item in enumerate(manifest["cases"]):
        guard(manifest)
        directory = next_attempt(root, index, item, resume)
        if directory is None:
            continue
        if shutil.disk_usage(ROOT).free < 5 * 1024 ** 3:
            raise ValueError("Less than 5 GiB free")
        print(f"START {index + 1}/10 {item['case_id']}", flush=True)
        status = batch.run_one(root, index, item, directory=directory)
        batch.refresh_reviews(root)
        batch.report(root, manifest)
        print(json.dumps({"case": index + 1, **status}, ensure_ascii=False), flush=True)
        guard(manifest)
        if status["status"] not in TERMINAL or not status.get("cleanup_verified"):
            raise ValueError(f"Paused for infrastructure/cleanup: {directory}")


def uniform_frames(ids: list[str], limit: int = 8) -> list[str]:
    """Evenly sample chronological available images, including both endpoints."""
    if len(ids) <= limit:
        return list(ids)
    if limit < 2:
        return ids[:limit]
    return [ids[round(i * (len(ids) - 1) / (limit - 1))] for i in range(limit)]


def experience(root: Path) -> None:
    manifest = read(root / "manifest.json")
    rows = []
    for index, item in enumerate(manifest["cases"]):
        guard(manifest)
        directory = batch.latest_case_dir(root, index, item)
        transcripts = list((directory / "runs").glob("**/memory/agent_transcript.jsonl"))
        row = {"case": index + 1, "case_id": item["case_id"], "attempt": str(directory)}
        rows.append(row)
        if len(transcripts) != 1:
            row.update(status="unavailable", reason="Expected exactly one task transcript", count=len(transcripts))
            continue
        source = transcripts[0].parent.parent
        bundle = read_run(source)
        row.update(source=str(source), images=len(bundle.images), events=len(bundle.timeline))
        if not bundle.images:
            row.update(status="unavailable", reason="No original RGB")
            continue
        output = root / "experience" / f"{index + 1:02d}"
        for mode, suffix, frames in (("inspect", "inspection", []), ("run", "autonomous", [])):
            invoke_experience(source, output / suffix, mode, frames)
        result = read(output / "autonomous/experiment.json")
        row.update(status=result["status"], candidates=len(result["candidates"]),
                   accepted=len(result["accepted"]), report=str(output / "autonomous/report.html"))
        # A single labelled diagnostic never replaces the autonomous result.
        if result["status"] == "failed":
            frames = uniform_frames([obs for obs in bundle.observations if obs in bundle.images])
            guard(manifest)
            invoke_experience(source, output / "diagnostic", "run", frames)
            diagnostic = read(output / "diagnostic/experiment.json")
            row["diagnostic"] = {"status": diagnostic["status"],
                "accepted": len(diagnostic["accepted"]), "observations": frames}
        batch.write_json(root / "experience_summary.json", rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    batch.write_json(root / "experience_summary.json", rows)


def invoke_experience(source: Path, output: Path, mode: str, frames: list[str]) -> None:
    artifact = output / ("inspection.json" if mode == "inspect" else "experiment.json")
    if artifact.exists():
        if mode == "run" and read(artifact).get("status") == "running":
            raise ValueError(f"Interrupted experience needs inspection: {output}")
        return
    command = [sys.executable, "-u", str(ROOT / "tests/manual/flight_experience_probe.py"),
               mode, "--source", str(source), "--output", str(output)]
    if frames:
        command += ["--observations", *frames]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if not artifact.exists():
        raise ValueError(f"Experience process failed before report ({completed.returncode}): {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fly", "experience"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "tests/manual_outputs/benchmark_20_flash_20260923_v2/manifest.json")
    parser.add_argument("--indices", nargs="+", type=int, default=list(DEFAULT_SELECTION))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        prepare(root, args.source.resolve(), args.indices)
    elif args.command == "fly":
        fly(root, args.resume)
    else:
        experience(root)


if __name__ == "__main__":
    main()
