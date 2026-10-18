"""Evaluation-only lightweight persistence with complete cost/media telemetry.

The supervised host installs this hashed logging adapter before constructing
any task. It reuses the full Aerial campaign's binary-free logger and preserves
the original observation dictionaries. Frozen planner, context, tools and
provider code remain untouched; only human records and frame-copy policy change.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from src.runtime.logging.async_run_logger import AsyncRunLogger

PROFILE = "paper_process_metrics_v1"


class MetricsRunLogger(AsyncRunLogger):
    """Use the existing lightweight logger regardless of legacy report profile."""

    def __init__(self, run_dir, **kwargs):
        kwargs["capture_provider_debug"] = False
        super().__init__(run_dir, **kwargs)


def metrics_round(planner, *, turn=None, error_type=None):
    """Retain counters, public turn and selection audit without replay payloads."""
    exchange = getattr(planner, "last_exchange", None) or {}
    response = exchange.get("response") or {}
    return {"response": {"usage": copy.deepcopy(response.get("usage"))},
            "transport": copy.deepcopy(exchange.get("transport") or {}),
            "history_projection": copy.deepcopy(exchange.get("history_projection") or {}),
            "canonical_turn": turn.to_dict() if turn is not None else None,
            "error_type": error_type}


def capture_metrics(journal, planner, session, decision_id, *, turn=None, error_type=None):
    """Write one small per-request receipt and exact media refs, once per hash."""
    record = metrics_round(planner, turn=turn, error_type=error_type)
    exchange = getattr(planner, "last_exchange", None) or {}
    entries = record["history_projection"].get("selected_visual_evidence") or []
    refs = {e["sha256"]: e.get("ref") for e in entries if e.get("sha256")}
    media = []
    for original in exchange.get("media") or []:
        item = {k: copy.deepcopy(v) for k, v in original.items() if k != "data"}
        digest = item.get("sha256")
        if digest in refs and refs[digest]:
            item["asset_ref"] = refs[digest]
        elif isinstance(original.get("data"), bytes) and digest:
            # Keep wire images whose bytes differ from the existing observation
            # asset. Ordinary RGB/maps reuse their original artifact reference.
            relative = f'debug/provider_media/{digest}{item.get("suffix", ".png")}'
            if digest not in journal._persisted_provider_media:
                journal.recorder.save_bytes(relative, original["data"])
                journal._persisted_provider_media.add(digest)
            item["asset_ref"] = relative
        media.append(item)
    record["request_media"] = media
    namespace = hashlib.sha256(str(session.transcript.instruction_id).encode()).hexdigest()[:16]
    journal.recorder.write_json(f"debug/model_call_metrics/{namespace}/{decision_id}.json",
        {"audit_profile": PROFILE, "decision_id": decision_id, "instruction_id": session.transcript.instruction_id,
         "model_rounds": [record]})


def reference_recorded_path(value, run_dir, destination, stem):
    """Human reviews reference the one original media tree instead of copying."""
    if value in (None, ""):
        return None
    source = Path(str(value))
    candidates = [source]
    if not source.is_absolute():
        candidates.extend((run_dir / source, run_dir.parent / source))
    return next((p for p in candidates if p.is_file()), None)


def install(full_episode_ids=None):
    """Install only logging/report bindings in this private execution host."""
    from evaluation.benchmarks.aerial_vln import runner as aerial
    from evaluation.benchmarks.travel_uav import runner as travel
    from evaluation.benchmarks.core import audit
    from src.runtime.task.journal import RuntimeJournal
    if getattr(RuntimeJournal.capture_provider_exchange, "_metrics_adapter", False):
        return
    original = RuntimeJournal.capture_provider_exchange
    def capture(self, planner, session, decision_id, *, turn=None, error_type=None):
        original(self, planner, session, decision_id, turn=turn, error_type=error_type)
        if isinstance(self.recorder, MetricsRunLogger):
            capture_metrics(self, planner, session, decision_id, turn=turn, error_type=error_type)
    capture._metrics_adapter = True
    RuntimeJournal.capture_provider_exchange = capture
    aerial.AsyncRunLogger = travel.AsyncRunLogger = MetricsRunLogger
    if full_episode_ids is not None:
        full_episode_ids = frozenset(full_episode_ids)
        def select_logger(run_dir, **kwargs):
            # Run directories are beneath attempts/<episode_id>/attempt_N/runs.
            # Select by an exact path component, never by a substring in a title.
            logger = AsyncRunLogger if full_episode_ids.intersection(Path(run_dir).parts) else MetricsRunLogger
            return logger(run_dir, **kwargs)
        aerial.AsyncRunLogger = travel.AsyncRunLogger = select_logger
    original_copy = audit._copy_recorded_path
    def copy_for_profile(value, run_dir, destination, stem):
        if full_episode_ids is not None and full_episode_ids.intersection(Path(run_dir).parts):
            return original_copy(value, run_dir, destination, stem)
        return reference_recorded_path(value, run_dir, destination, stem)
    audit._copy_recorded_path = copy_for_profile
