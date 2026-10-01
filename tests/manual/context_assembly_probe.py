"""Measure Model Context V3 assembly against optional frozen V2 metrics.

This probe never reads V1 runs or imports V1 source. A previous experiment may
provide a small metrics JSON via --v1-metrics; it is treated as numeric baseline
data only. The live measurement always uses V2 contracts and lazy media.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.transcript import AgentTranscript
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.media_refs import image_media_ref
from src.memory.stores import MemoryEventWriter
from src.tools.registry import ToolRegistry


def prefix_boundary(records, canonical, instruction_id):
    """Return the unique durable boundary preceding a canonical turn."""
    identity = canonical.get("turn_id")
    if not identity:
        raise ValueError("No canonical turn identity; prefix cannot be verified")
    matches = [
        index
        for index, event in enumerate(records)
        if event.get("instruction_id") == instruction_id
        and event.get("kind") == "assistant_turn"
        and event.get("payload", {}).get("turn_id") == identity
    ]
    if len(matches) != 1:
        raise ValueError("Canonical turn boundary missing or ambiguous")
    boundary = matches[0]
    if records[boundary].get("payload") != canonical:
        raise ValueError("Canonical turn contents do not match durable history")
    if any(
        event.get("instruction_id") != instruction_id
        for event in records[:boundary]
    ):
        raise ValueError("Prefix contains a different instruction")
    return boundary


def distribution(values):
    if not values:
        return {"p50_ms": None, "p95_ms": None, "n": 0}
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return {
        "p50_ms": statistics.median(ordered) * 1000,
        "p95_ms": p95 * 1000,
        "n": len(ordered),
    }


def alternating(functions, repeats=30):
    """Alternate ordering and require exact nested output equality."""
    rows = []
    for repeat in range(repeats + 1):
        values = {}
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for variant in names:
            started = time.perf_counter()
            values[variant] = functions[variant]()
            rows.append(
                {
                    "variant": variant,
                    "repeat": repeat,
                    "first": repeat == 0,
                    "wall_sec": time.perf_counter() - started,
                }
            )
        baseline = values["baseline"]
        for variant, value in values.items():
            if value != baseline:
                raise ValueError(f"Output differs: {variant}, repeat={repeat}")
    return {
        "equal": True,
        "rows": rows,
        "stats": {
            name: distribution(
                [
                    row["wall_sec"]
                    for row in rows
                    if row["variant"] == name and not row["first"]
                ]
            )
            for name in functions
        },
    }


def diagnostic(function: Callable[[], Any]) -> dict[str, Any]:
    """Keep copy counting and peak-memory collection in separate invocations."""
    original = copy.deepcopy
    copy_calls = 0
    depth = 0

    def counted(value, *args, **kwargs):
        nonlocal copy_calls, depth
        outer = depth == 0
        depth += 1
        try:
            return original(value, *args, **kwargs)
        finally:
            depth -= 1
            if outer:
                copy_calls += 1

    copy.deepcopy = counted
    try:
        function()
    finally:
        copy.deepcopy = original
    tracemalloc.start()
    try:
        function()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return {"copy_calls": copy_calls, "python_peak_bytes": peak}


def _checkpoint(index: int, image_path: Path) -> StateCheckpoint:
    observation_id = f"obs_{index:04d}"
    media = image_media_ref(
        image_path,
        observation_id=observation_id,
        frame_id=f"rgb_{index:04d}",
        role="current_observation",
        usage="Recorded chronological observation.",
        scene_width=32,
        scene_height=24,
    )
    fragment = ContextFragment(
        "image",
        {"sequence": index},
        (media.to_model_input(),),
    )
    fragments = (fragment,)
    reason = "instruction_start" if index == 1 else "physical_window_completed"
    return StateCheckpoint(
        checkpoint_id=f"checkpoint_{index:04d}",
        reason=reason,
        observation_id=observation_id,
        fragments=fragments,
        fingerprint=checkpoint_fingerprint(observation_id, fragments),
    )


def measure_v2(output: Path, history_length: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "fixed_rgb.png"
    Image.new("RGB", (32, 24), (17, 89, 143)).save(image_path)
    transcript = AgentTranscript(MemoryEventWriter(output))
    transcript.start_instruction(
        "Inspect the fixed long history.",
        "probe_instruction",
        InstructionEnvelope(
            instruction_id="probe_instruction",
            protocol_context="offline long-history measurement",
            identity={"name": "FlyAgent"},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )
    build_started = time.perf_counter()
    for index in range(1, history_length + 1):
        transcript.append_state(_checkpoint(index, image_path))
    checkpoint_build_sec = time.perf_counter() - build_started

    hydrator = MediaHydrator(base_dir=output)
    projector = HistoryProjector(assembler=RequestAssembler(hydrator))
    tracemalloc.start()
    started = time.perf_counter()
    projection = projector.project(transcript, ToolRegistry())
    elapsed = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    request = {"messages": projection.messages}
    request_json = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    text_only = copy.deepcopy(request)
    for message in text_only["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "<prepared-media>"}
    text_json = json.dumps(text_only, ensure_ascii=False, separators=(",", ":"))
    return {
        "schema_version": "context_assembly_probe_v2",
        "history_length": history_length,
        "checkpoint_build_count": history_length,
        "checkpoint_build_sec": checkpoint_build_sec,
        "request_text_bytes": len(text_json.encode("utf-8")),
        "provider_request_bytes": len(request_json.encode("utf-8")),
        "media": hydrator.stats(),
        "projection_peak_python_bytes": peak,
        "projection_end_to_end_sec": elapsed,
        "message_count": len(projection.messages),
        "request_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
    }


def _numeric_comparison(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "checkpoint_build_count",
        "request_text_bytes",
        "projection_peak_python_bytes",
        "projection_end_to_end_sec",
    )
    rows = {}
    for field in fields:
        before, after = v1.get(field), v2.get(field)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            rows[field] = {
                "v1": before,
                "v2": after,
                "delta": after - before,
                "ratio": (after / before) if before else None,
            }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=100)
    parser.add_argument(
        "--v1-metrics",
        type=Path,
        help="Optional numeric metrics JSON from a frozen V1 experiment; never a V1 run.",
    )
    args = parser.parse_args()
    if args.history_length < 1:
        raise ValueError("--history-length must be positive")
    result = {"v2": measure_v2(args.output, args.history_length)}
    if args.v1_metrics is not None:
        baseline = json.loads(args.v1_metrics.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise ValueError("--v1-metrics must contain one JSON object")
        result["v1_metrics"] = baseline
        result["comparison"] = _numeric_comparison(baseline, result["v2"])
    target = args.output / "context_assembly_metrics.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
