"""Offline context-read experiment and fail-closed historical input admission.

Historical model_context is an output, not a builder input checkpoint. Never
reverse it into missing FlightMemory/state or fill gaps with final state files.
The reusable compare_builds function accepts fully supplied, verified inputs;
the CLI audits historical availability and exports a partial review if absent.
No simulator, model client, environment credentials or network calls are used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import tracemalloc
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.memory.stores import MemoryRecordReader
from tests.manual.experiment_sources import source_hashes
from tests.manual.gpu_probe_environment import check_interpreter
from tests.manual.task_profile import TaskProfile
from tests.manual.tool_selection_ablation_probe import restore_request, digest, write_json
from tests.manual.export_benchmark_review import validate_bundle
from tests.manual.review_browser_probe import inspect


class LegacyMemoryRecordReader(MemoryRecordReader):
    """Exact pre-optimization reading algorithm, confined to the experiment."""

    def read_scope(self):
        return nullcontext()

    def read_ref(self, ref):
        if ref is None:
            return None
        value = ref.get("ref") if isinstance(ref, dict) else ref
        if not isinstance(value, str) or "#" not in value:
            return None
        relative, line_text = value.split("#", 1)
        try:
            number = int(line_text)
        except ValueError:
            return None
        path = self.run_dir / relative
        if not path.exists():
            return None
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if index == number:
                return json.loads(line)
        return None


def compare_builds(builder, state, memory, capture_request, output, repeats=10):
    """Compare full builds for caller-verified inputs; callback MUST be offline.

    Normal wall timings are uninstrumented. Stage, I/O and allocation sampling
    are separate diagnostic builds, excluded from ordinary warm statistics.
    Each variant uses a distinct reader; the memory manifest itself is retained.
    """
    if repeats < 2:
        raise ValueError("At least two repeats are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    readers = {"baseline": LegacyMemoryRecordReader(memory.memory_reader.run_dir),
               "candidate": MemoryRecordReader(memory.memory_reader.run_dir)}
    rows, reference, wire_reference = [], None, None
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for variant in order:
            source = replace(memory, memory_reader=readers[variant])
            start = time.perf_counter()
            context = builder.build(state, source)
            elapsed = time.perf_counter() - start
            # Capture outside the timing region. Compare actual content, not
            # just a hash or a request with image placeholders removed.
            wire = capture_request(context)
            if reference is None:
                reference, wire_reference = context, wire
            if context != reference or wire != wire_reference:
                write_json(output / "difference.json", {"variant": variant, "repeat": repeat,
                    "context_equal": context == reference, "request_equal": wire == wire_reference})
                raise ValueError("Context or captured request differs")
            rows.append(dict(variant=variant, repeat=repeat + 1, cold=repeat == 0,
                             wall_sec=elapsed, context_sha256=digest(context), request_sha256=digest(wire)))
    diagnostics = {}
    original_read, original_loads = Path.read_text, json.loads
    from src.memory.context.modules import image
    for variant, reader in readers.items():
        counts = {"file_reads": 0, "utf8_bytes": 0, "json_parses": 0, "files": {}}
        profile = TaskProfile(output / f"{variant}_timings.jsonl")
        def read(path, *a, **kw):
            with profile.span("file.read_and_decode"):
                value = original_read(path, *a, **kw)
            counts["file_reads"] += 1
            counts["utf8_bytes"] += len(value.encode("utf-8"))
            counts["files"][str(path)] = counts["files"].get(str(path), 0) + 1
            return value
        def loads(*a, **kw):
            with profile.span("json.parse"):
                value = original_loads(*a, **kw)
            counts["json_parses"] += 1
            return value
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "read_text", read))
            stack.enter_context(patch.object(json, "loads", loads))
            profile.wrap(stack, image, "rgb_data_url", "image.encode")
            for name, module in builder.composer.registry.modules.items():
                profile.wrap(stack, module, "build", f"module.{name}")
            with profile.span("context.build"):
                measured = builder.build(state, replace(memory, memory_reader=reader))
        assert measured == reference
        tracemalloc.start()
        try:
            built = builder.build(state, replace(memory, memory_reader=reader))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert built == reference
        diagnostics[variant] = {**counts, "peak_python_allocated_bytes": peak,
            "memory_scope": "separate build; Python allocations, not total process RSS"}
    from tests.manual.read_only_concurrency_probe import percentile
    summary = {}
    for variant in readers:
        values = [r["wall_sec"] for r in rows if r["variant"] == variant and not r["cold"]]
        summary[variant] = dict(p50_sec=statistics.median(values), p95_sec=percentile(values, .95))
    result = dict(state="passed", repeats=repeats, rows=rows, timings=summary, diagnostics=diagnostics)
    write_json(output / "comparison.json", result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_blockers(decision):
    """Explain only facts of the recorded format; no synthetic recovery fields."""
    # The existing decision format records model_context/model_rounds, but no
    # full builder input. An unknown future format must be explicitly supported
    # and tested before this probe could admit it as a performance sample.
    return ["No supported per-decision builder-input checkpoint: need FlightMemory references/anchors, "
            "MissionState inputs and map/run/module configuration at this decision. "
            "Recorded model_context is already-built output; latest_state/flight_memory are final-only."]


def select_samples(batch):
    """Choose earliest complete audit per benchmark, independent of scores."""
    summary = read_json(batch / "summary.json")
    selections, skipped = [], []
    for benchmark in ("aerial", "travel"):
        chosen = None
        for case in sorted(summary["cases"], key=lambda item: item["index"]):
            if case["benchmark"] != benchmark:
                continue
            runs = sorted((batch / case["attempt_dir"] / "runs").iterdir())
            if len(runs) != 1:
                skipped.append({"case_id": case["case_id"], "reason": "ambiguous run directory"})
                continue
            run = runs[0]
            decisions = sorted((run / "debug/tool_loop").glob("*/decision_*.json"))
            if len(decisions) < 3:
                skipped.append({"case_id": case["case_id"], "reason": "fewer than three decision records"})
                continue
            picked = [decisions[0], decisions[(len(decisions) - 1) // 2], decisions[-1]]
            candidates = []
            try:
                for position, path in zip(("first", "middle", "last"), picked):
                    decision = read_json(path)
                    record = decision["model_rounds"][0]
                    # Existing strict restoration validates paths and SHA256,
                    # and rejects missing or external media without fallback.
                    restored = restore_request(record["request"], record["request_media"], run)
                    candidates.append(dict(benchmark=benchmark, case_id=case["case_id"], position=position,
                        decision_index=decision["decision_index"], decision_path=str(path), run_dir=str(run),
                        request_sha256=digest(restored), state="paused", blockers=checkpoint_blockers(decision)))
            except (KeyError, IndexError, ValueError, OSError) as exc:
                skipped.append({"case_id": case["case_id"], "reason": str(exc)})
                continue
            chosen = candidates
            break
        if chosen is None:
            raise ValueError(f"No complete saved request triple for {benchmark}")
        selections.extend(chosen)
    return selections, skipped


def run(args):
    check_interpreter()
    if args.root.exists() or args.review_root.exists():
        raise ValueError("Use new output directories; never overwrite earlier evidence")
    samples, skipped = select_samples(args.batch.resolve())
    args.root.mkdir(parents=True)
    args.review_root.mkdir(parents=True)
    hashes = source_hashes()
    for relative in hashes:
        target = args.root / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    inputs = {}
    links = []
    for index, sample in enumerate(samples, 1):
        decision_path = Path(sample["decision_path"])
        decision = read_json(decision_path)
        record = decision["model_rounds"][0]
        directory = args.review_root / "cases" / f"{index:02d}"
        directory.mkdir(parents=True)
        inputs[str(decision_path)] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        shutil.copyfile(decision_path, args.root / f"decision_{index:02d}.json")
        # Preserve exact placeholder-bearing requests for detailed human review;
        # images themselves are exported below with short relative names.
        write_json(directory / "request.json", record["request"])
        pictures = []
        for n, media in enumerate(record["request_media"], 1):
            source = Path(sample["run_dir"]) / media["asset_ref"]
            inputs[str(source)] = media["sha256"]
            name = f"i{n:03d}" + {"image/png": ".png", "image/webp": ".webp"}.get(media["mime_type"], ".jpg")
            shutil.copyfile(source, directory / name)
            pictures.append(f'<figure><img width="320" src="{name}"><figcaption>{html.escape(str(media["request_path"]))}</figcaption></figure>')
        body = '<!doctype html><meta charset="utf-8"><h1>Context input admission — paused</h1>'
        body += '<p>Historical input review only. No rebuilt context, speed measurement or navigation score.</p>'
        body += '<pre>' + html.escape(json.dumps(sample, ensure_ascii=False, indent=2)) + '</pre>'
        body += '<div class="phase-timeline">' + ''.join(pictures) + '</div>'
        (directory / "review.html").write_text(body + '<a href="audit_review.html">Saved request audit</a>', encoding="utf-8")
        (directory / "audit_review.html").write_text(body + '<a href="request.json">Original saved request</a><pre>' +
            html.escape(json.dumps(record["request"], ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
        links.append(f'<li>{html.escape(sample["benchmark"])} / {sample["position"]}: '
                     f'<a href="cases/{index:02d}/review.html">images</a> · <a href="cases/{index:02d}/audit_review.html">audit</a></li>')
    manifest = dict(state="paused", reason="historical_builder_inputs_unavailable", samples=samples,
        skipped=skipped, source_hashes=hashes, input_hashes=inputs, python=sys.executable,
        timed_builds=0, model_requests=0, simulator_started=False)
    write_json(args.root / "manifest.json", manifest)
    write_json(args.review_root / "summary.json", manifest)
    page = '<!doctype html><meta charset="utf-8"><h1>上下文读取优化：历史回放暂停</h1>'
    page += '<p>已冻结六个历史请求及图片。缺少逐轮构建输入，不以最终任务状态回填，不报告历史上下文加速比。工程测试与历史回放验收分开。</p><ul>' + ''.join(links) + '</ul><a href="summary.json">冻结清单与原因</a>'
    (args.review_root / "index.html").write_text(page, encoding="utf-8")
    browser_dir = args.review_root / "browser_validation"
    browser_dir.mkdir()
    pages = [f'cases/{i:02d}/{kind}.html' for i in (1, 4) for kind in ('review', 'audit_review')]
    for name, exe in (("chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe"),
                      ("edge", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")):
        write_json(browser_dir / f"{name}.json", inspect(Path(exe), args.review_root, pages, name))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in inputs.items())
    assert all(hashlib.sha256((args.root / "source" / p).read_bytes()).hexdigest() == value for p, value in hashes.items())
    write_json(args.review_root / "resource_validation.json", validate_bundle(args.review_root))
    print(json.dumps({"state": "paused", "samples": len(samples), "review": str(args.review_root / "index.html")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    run(parser.parse_args())
