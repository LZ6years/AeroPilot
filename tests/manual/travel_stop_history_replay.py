"""Replay three saved decision boundaries with current guidance, executing no tools.

Requests are explicitly offline reconstructions from the durable transcript,
not claimed to be the exact historical provider requests. The configured model
and endpoint are retained; transport retries are disabled for this small probe.
Raw history/media remain unchanged except declared protocol/knowledge prose.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.manual.aerial_vln_context_module_audit import events_from_records
from src.agent.config import load_model_config
from src.agent.context.media import MediaHydrator
from src.agent.context.projection import HistoryProjector
from src.agent.context.request_assembler import RequestAssembler
from src.agent.planning.model_action_planner import ModelActionPlanner
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT


class RecordedTranscript:
    """Read-only adapter for the shared request projection and planner."""
    def __init__(self, records):
        self.records = records
        self.events = events_from_records(records)
    def snapshot(self):
        return copy.deepcopy(self.events)
    def instruction_envelope(self):
        return next(copy.deepcopy(r["payload"]) for r in self.records if r["kind"] == "instruction_envelope")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-fragment", action="append", help="Restrict the fixed saved-input set")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.campaign / "analysis/travel30_failures_embedded_20261018/manifest.json").read_text(encoding="utf-8"))
    config = replace(load_model_config(), timeout_sec=60, provider_timeout_retries=0,
                     provider_connection_retries=0, retry_deadline_sec=65)
    registry = default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json"))
    results = []
    # These are evaluation-only saved inputs, never production special cases.
    for fragment, boundary in (("dd5dcabf", "terminal"), ("cc390351", "terminal"), ("fb6be330", "historical_read")):
        if args.case_fragment and fragment not in args.case_fragment:
            continue
        row = next(r for r in manifest["cases"] if fragment in r["episode"])
        run = next((Path(row["source"]).parents[3] / "runs").iterdir())
        source = run / "memory/agent_transcript.jsonl"
        records = []
        decisions = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            p = record["payload"]
            if record["kind"] == "assistant_turn":
                decisions += 1
                if ((boundary == "terminal" and any(c["tool_name"] == "runtime.report_completed" for c in p["tool_calls"]))
                    or (boundary == "historical_read" and any(c["tool_call_id"] == "call_7897b568b7f0471f9fc6302bca6537b5" for c in p["tool_calls"]))):
                    break
            if record["kind"] == "instruction_envelope":
                p["protocol_context"] = TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT
            elif record["kind"] == "tool_result" and p.get("tool_name") == "knowledge.load" and p.get("status") == "ok":
                name = p.get("data", {}).get("name")
                path = Path("plugins/knowledge/uav_flight") / str(name) / "SKILL.md"
                if path.is_file():
                    p["data"]["body"] = path.read_text(encoding="utf-8").split("---", 2)[-1].strip()
            records.append(record)
        planner = ModelActionPlanner(config, history_projector=HistoryProjector(
            assembler=RequestAssembler(MediaHydrator(base_dir=run)),
            media_policy={"max_request_images": 30, "max_tool_evidence_images": 4,
                          "keep_recent_observations": 3, "pin_instruction_start": True,
                          "pin_current": True}))
        result = {"episode_id": row["episode"], "boundary": boundary,
                  "mode": "offline_reconstructed_from_transcript_no_execution",
                  "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "decision_index": decisions, "model": config.model_name}
        try:
            turn = planner.plan_turn(RecordedTranscript(records), registry, decision_index=decisions)
            result["turn"] = turn.to_dict()
            result["usage"] = (planner.last_response or {}).get("usage")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(any(r.get("error") for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
