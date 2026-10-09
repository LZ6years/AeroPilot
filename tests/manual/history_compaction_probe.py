"""Replay saved benchmark transcripts through the image-free compactor.

This diagnostic never mutates a run or starts a simulator. It selects the
largest oldest contiguous prefix that fits the configured 32k semantic budget,
optionally launches one background model call per sample, and writes a
binary-free JSON report. Invoke it with the required ``flyagent`` interpreter:

    F:\conda_env\flyagent\python.exe tests/manual/history_compaction_probe.py ^
      --sample aerial=RUN/memory/agent_transcript.jsonl ^
      --sample travel=RUN/memory/agent_transcript.jsonl --invoke-model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.config import load_model_config
from src.agent.context.compression import (
    HistoryCompactionCoordinator,
    HistoryCompressionConfig,
    HistoryProjectionState,
    ModelHistoryCompactor,
)
from src.agent.context.compression.validator import CompressionValidator
from src.agent.context.projection import HistoryProjector
from src.agent.context.window_manager import ContextWindowManager
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult, TranscriptEvent
from src.llm import create_model_client
from src.tools.registry import default_registry


class SavedTranscript:
    """Read-only snapshot API needed by ``ContextWindowManager``."""

    def __init__(self, events: tuple[TranscriptEvent, ...]) -> None:
        self.events = events

    def snapshot(self) -> tuple[TranscriptEvent, ...]:
        return self.events

    def snapshot_range(
        self, start: int, end: int | None = None
    ) -> tuple[TranscriptEvent, ...]:
        return self.events[start:end]


def _load(path: Path) -> SavedTranscript:
    events: list[TranscriptEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row["kind"]
        payload = row.get("payload") or {}
        if kind == "assistant_turn":
            payload = AssistantTurn(
                turn_id=str(payload.get("turn_id") or "saved_turn"),
                public_intent=str(payload.get("public_intent") or ""),
                public_intent_source=payload.get(
                    "public_intent_source", "provider_content"
                ),
                tool_calls=[AgentToolCall(**item) for item in payload.get("tool_calls", [])],
            )
        elif kind == "tool_result":
            payload = ToolResult(**payload)
        events.append(TranscriptEvent(kind, payload))
    return SavedTranscript(tuple(events))


def _sample(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sample must be LABEL=TRANSCRIPT_PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not label.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid sample: {value}")
    return label.strip(), path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="append", type=_sample, required=True)
    parser.add_argument("--context-config", default="configs/context.json")
    parser.add_argument("--invoke-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = json.loads(Path(args.context_config).read_text(encoding="utf-8-sig"))
    config = HistoryCompressionConfig.from_mapping(root.get("history_projection"))
    provider = load_model_config().provider_config()
    settings = replace(
        provider.to_provider_settings(),
        timeout_sec=config.compactor_timeout_sec,
        provider_timeout_retries=config.compactor_timeout_retries,
        provider_connection_retries=config.compactor_connection_retries,
    )
    compactor = ModelHistoryCompactor(
        create_model_client(settings), summary_max_tokens=config.summary_max_tokens,
        thinking_parameters=provider.compaction_parameters(),
        structured_output_format=provider.structured_output_format,
    )
    manager = ContextWindowManager(HistoryProjector(), config, compactor=compactor)
    registry = default_registry()
    jobs: dict[str, tuple[HistoryCompactionCoordinator, Any]] = {}
    report: dict[str, Any] = {
        "schema_version": "history_compaction_probe_v1",
        "sys_executable": sys.executable,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "dirty_diff_sha256": hashlib.sha256(
            _git_output("diff", "--binary").encode("utf-8")
        ).hexdigest(),
        "invoke_model": args.invoke_model,
        "provider": settings.provider,
        "model_name": settings.model_name,
        "timeout_sec": settings.timeout_sec,
        "timeout_retries": settings.provider_timeout_retries,
        "connection_retries": settings.provider_connection_retries,
        "thinking_parameters": provider.compaction_parameters(),
        "structured_output_format": provider.structured_output_format,
        "compression_config": {
            "prefetch_trigger_tokens": config.prefetch_trigger_tokens,
            "soft_trigger_tokens": config.soft_trigger_tokens,
            "target_tokens": config.target_tokens,
            "max_compactor_input_tokens": config.max_compactor_input_tokens,
            "summary_max_tokens": config.summary_max_tokens,
            "keep_recent_decision_windows": config.keep_recent_decision_windows,
            "max_request_images": config.max_request_images,
            "max_tool_evidence_images": config.max_tool_evidence_images,
            "keep_recent_observations": config.keep_recent_observations,
            "pin_instruction_start": config.pin_instruction_start,
            "pin_current": config.pin_current,
        },
        "samples": {},
    }

    for label, path in args.sample:
        transcript = _load(path)
        candidate, preparation = manager._prepare_candidate(  # diagnostic seam
            transcript, registry, HistoryProjectionState(), {}
        )
        if candidate is None:
            raise RuntimeError(f"{label}: no compression candidate ({preparation})")
        semantic = json.dumps(
            [window.to_dict() for window in candidate.request.windows],
            ensure_ascii=False,
        )
        forbidden = {
            term: semantic.count(term)
            for term in ("model_inputs", "media_refs", "media_ref", "base64", "data:image")
        }
        row = {
            "transcript": str(path),
            "preparation_status": preparation,
            "through_event_index": candidate.through_event_index,
            "decision_window_count": len(candidate.request.windows),
            "estimated_input_tokens": candidate.estimated_input_tokens,
            "max_compactor_input_tokens": config.max_compactor_input_tokens,
            "forbidden_media_term_counts": forbidden,
            "source_digest": candidate.request.source_digest,
            "indexed_prefix_media_count": len(candidate.prefix_media),
            "indexed_prefix_media": [
                {
                    "event_index": item.event_index,
                    "evidence_ref": item.evidence_ref,
                    "observation_id": item.observation_id,
                    "frame_id": item.frame_id,
                    "source_kind": item.source_kind,
                    "checkpoint_reason": item.checkpoint_reason,
                    "tool_call_id": item.tool_call_id,
                    "ref": item.media_ref.get("ref"),
                    "sha256": item.media_ref.get("sha256"),
                }
                for item in candidate.prefix_media
            ],
            "status": "prepared",
        }
        report["samples"][label] = row
        if candidate.estimated_input_tokens > config.max_compactor_input_tokens:
            row["status"] = "oversized"
            continue
        if args.invoke_model:
            coordinator = HistoryCompactionCoordinator()
            if not coordinator.submit(candidate, compactor):
                raise RuntimeError(f"{label}: background submission refused")
            jobs[label] = (coordinator, candidate)
            row["status"] = "pending"

    deadline = time.monotonic() + config.compactor_timeout_sec + 10
    while jobs and time.monotonic() < deadline:
        complete: list[str] = []
        for label, (coordinator, candidate) in jobs.items():
            snapshot = coordinator.poll()
            if snapshot.status == "pending":
                continue
            row = report["samples"][label]
            row["status"] = snapshot.status
            row["errors"] = list(snapshot.errors)
            if snapshot.started_at is not None and snapshot.finished_at is not None:
                row["elapsed_seconds"] = max(
                    0.0, snapshot.finished_at - snapshot.started_at
                )
            if snapshot.status == "ready" and snapshot.result is not None:
                validation = CompressionValidator(config.summary_max_tokens).validate(
                    snapshot.result.draft, candidate.request
                )
                row.update(
                    validation_accepted=validation.accepted,
                    validation_errors=list(validation.errors),
                    elapsed_seconds=snapshot.result.elapsed_seconds,
                    finish_reason=snapshot.result.finish_reason,
                    usage=snapshot.result.usage,
                    summary=snapshot.result.draft.to_dict(),
                )
            complete.append(label)
        for label in complete:
            jobs.pop(label)
        if jobs:
            time.sleep(0.1)
    for label in jobs:
        report["samples"][label]["status"] = "probe_deadline_exceeded"

    output = args.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = Path("runs/experiments/history_compaction") / stamp / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False, indent=2))
    return 0 if all(
        row["status"] in ({"ready"} if args.invoke_model else {"prepared"})
        and all(count == 0 for count in row["forbidden_media_term_counts"].values())
        and (not args.invoke_model or row.get("validation_accepted") is True)
        for row in report["samples"].values()
    ) else 1


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8"
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
