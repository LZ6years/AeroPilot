"""Import an existing 5x2 experiment batch and optionally maintain its candidates.

Creates a NEW isolated directory. V2 may publish independently approved items in
experiment scope only. Never mutates the source batch or substitutes missing cases.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plugins.research.flight_experience.store import Store
from plugins.research.flight_experience.maintenance import Worker
from plugins.research.flight_experience.experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--reflection-model")
    parser.add_argument("--review-model")
    args = parser.parse_args()
    batch, output = args.batch.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(batch) or batch.is_relative_to(output):
        raise ValueError("Output must be separate from historical batch")
    output.mkdir(parents=True, exist_ok=False)
    store = Store(output / "experience.sqlite")
    store.initialize()
    report = {"source_batch": str(batch), "python": sys.executable, "imports": [], "cases": [],
        "attempts": [], "published": 0, "closed_loop_benefit": "not_evaluated", "estimated_cost_usd": None}
    start = time.monotonic()
    for case in range(1, 11):
        entries = []
        for mode in ("autonomous", "diagnostic"):
            directory = batch / "experience" / f"{case:02d}" / mode
            if not (directory / "experiment.json").exists():
                entries.append({"mode": mode, "status": "missing"})
                continue
            imported = store.import_experiment(directory)
            entries.append({"mode": mode, **imported})
            report["imports"].append({"case": case, "mode": mode, **imported})
            assert store.import_experiment(directory)["duplicate"]
        report["cases"].append({"case": case, "experiments": entries})
    report["source_audit"] = store.audit_sources()
    report["candidate_ids"] = store.list_ids()
    queries = ["inconsistent_depth_surfaces invalid distance", "bridge altitude occlusion", "道路尽头", "目标未见 继续前进", "descending final confirmation"]
    report["retrieval"] = [{"query": q, "maintenance": store.search(q, maintenance=True),
        "flight_experiment": store.search(q, scope="experiment"), "flight_production": store.search(q)} for q in queries]
    write_json(output / "summary.json", report)
    if args.run_models and report["candidate_ids"]:
        from src.llm import create_model_client
        from src.agent.config import load_model_config
        model = load_model_config(args.env)
        reflection = replace(model.to_provider_settings(), model_name=args.reflection_model or model.model_name,timeout_sec=180,provider_timeout_retries=1)
        review = replace(model.to_provider_settings(), model_name=args.review_model or model.model_name,timeout_sec=180,provider_timeout_retries=1)
        report["model_config"] = {"reflection_model": reflection.model_name, "review_model": review.model_name,
            "timeout_sec": reflection.timeout_sec, "timeout_retries": reflection.provider_timeout_retries}
        parameters = model.thinking_parameters()
        store.enqueue(report["candidate_ids"], request_key="initial_batch")
        worker = Worker(store, create_model_client(reflection), create_model_client(review), output / "maintenance", parameters=parameters)
        # Bound the main experiment to one attempt per candidate; no timeout replay.
        for _ in report["candidate_ids"]:
            result = worker.run_once()
            if result is None:
                break
            report["attempts"].append(result)
            write_json(output / "summary.json", report)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    report["proposals"] = store.proposals()
    report['published']=sum(store.get(eid)['state']=='published' for eid in store.list_ids())
    report["jobs"] = store.jobs()
    report["elapsed_seconds"] = time.monotonic()-start
    calls = [call for path in (output / "maintenance").rglob("calls.json") for call in json.loads(path.read_text(encoding="utf-8"))] if (output / "maintenance").exists() else []
    report["usage"] = {"calls": len(calls), "calls_without_usage": sum(not c.get("usage") for c in calls),
        "returned_total_tokens": sum((c.get("usage") or {}).get("total_tokens", 0) for c in calls),
        "elapsed_model_seconds": sum(c.get("elapsed_seconds", 0) for c in calls)}
    write_json(output / "summary.json", report)
    links = []
    for path in sorted((output / "maintenance").rglob("result.json")):
        folder = path.parent.relative_to(output).as_posix()
        links.append(f'<li>{html.escape(folder)}: <a href="{folder}/before.json">before</a> '
            f'<a href="{folder}/proposal.json">proposal (if generated)</a> <a href="{folder}/calls.json">calls/review</a> '
            f'<a href="{folder}/result.json">result</a></li>')
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>Experience library V1 experiment</title>'
        '<h1>Experience library — isolated offline experiment</h1><p>Publication counts are listed below; no flight executed.</p><ul>'
        + ''.join(links) + '</ul><pre style="white-space:pre-wrap">' + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    print(str(output / "report.html"))


if __name__ == "__main__":
    main()
