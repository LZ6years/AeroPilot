"""Read-only acceptance of manifest-sized artifacts and actual requests.

No model, scene or score is regenerated. Report-reader changes are separately
hashed so navigation provenance remains the immutable experiment snapshot.
"""
import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

from src.agent.planning.task_plan import TaskPlan
from src.tools.diagnostics import Diagnostic
from tests.manual.aerial_vln_plan_context_probe import digest, read, write
from tests.manual.aerial_vln_plan_mode import admission_passed


def delivered_media(model_round):
    """Join original image positions to their original adjacent labels.

    Older image audits omitted CONTEXT IMAGE labels. This derived reader uses
    captured request text, never a regenerated view or a later observation.
    """
    result = []
    latest = [m for m in (model_round.get("history_projection") or {}).get("selected_visual_evidence", [])
              if m.get("selection_tier") == "latest_group"]
    for raw in model_round.get("request_media") or []:
        item = dict(raw)
        path = raw.get("request_path")
        if isinstance(path, list):
            try:
                content = model_round["request"][path[0]][path[1]][path[2]]
                label = content[path[3] - 1].get("text", "")
            except (KeyError, IndexError, TypeError):
                label = ""
            if label.startswith("CONTEXT IMAGE "):
                item.update(label=label, timeline_role="CONTEXT_IMAGE", label_source="captured_adjacent_text")
                for key in ("updated_at_observation_id", "frame_id"):
                    match = re.search(rf"\b{key}=([^\s]+)", label.splitlines()[0])
                    if match:
                        item[key] = match.group(1)
            # save_place's immediate result in older snapshots uses the tool
            # image caption. Match the actual captured image digest to the
            # captured latest-group selection, and retain its original label.
            # This is image identity recovery, not relabeling the model input.
            matching = [m for m in latest if m.get("sha256") == item.get("sha256")]
            if len(matching) == 1:
                item.update(context_image_classification="captured_latest_group",
                    selected_frame_id=matching[0].get("frame_id"),
                    selected_observation_id=matching[0].get("observation_id"))
                if not label.startswith("CONTEXT IMAGE "):
                    item["context_label_variant"] = "tool_result_caption"
        result.append(item)
    return result


def request_checks(model_round, *, require_map=False, plan_mode="optional", require_place_list=False):
    """Check raw delivered messages, independently of projection metadata."""
    messages = model_round["request"]["messages"]
    failures = []
    states = [m["content"] for m in messages if isinstance(m.get("content"), str)
              and m["content"].startswith("CURRENT TASK STATE\n")]
    if len(states) != 1:
        failures.append("latest_task_state_not_unique")
    elif json.loads(states[0].split("\n", 1)[1]) != model_round.get("current_task_state"):
        failures.append("latest_task_state_mismatch")
    calls = {call["id"] for m in messages for call in m.get("tool_calls", [])}
    image_count = 0
    for message in messages:
        parts = message.get("content")
        if isinstance(parts, list):
            image_count += sum(p.get("type") == "image_url" for p in parts)
            for part in parts:
                text = part.get("text", "")
                if text.startswith("STATE CHECKPOINT") and not re.match(r"STATE CHECKPOINT (CURRENT|HISTORY) ", text):
                    failures.append("checkpoint_time_role_missing")
        if message["role"] == "tool":
            result = json.loads(message["content"])
            if message.get("tool_call_id") in calls and "requested_args" in result:
                failures.append("duplicate_requested_args")
            diagnostic = result.get("diagnostic")
            if diagnostic is not None:
                try:
                    if Diagnostic(**diagnostic).execution is None:
                        failures.append("tool_diagnostic_execution_missing")
                except ValueError:
                    failures.append("invalid_tool_diagnostic")
    if image_count > 20:
        failures.append("image_budget_exceeded")
    if require_map:
        # Labels and bytes are checked together: a module description alone is
        # not evidence that the map or the current camera reached the provider.
        media = delivered_media(model_round)
        current = [m for m in media if m.get("timeline_role") == "CURRENT"]
        maps = [m for m in media if m.get("label", "").startswith("CONTEXT IMAGE ")
                or m.get("context_image_classification") == "captured_latest_group"]
        if len(current) != 1:
            failures.append("current_rgb_not_unique")
        if len(maps) != 1:
            failures.append("latest_context_image_not_unique")
        elif current:
            expected_obs = current[0].get("observation_id")
            if maps[0].get("context_label_variant") == "tool_result_caption":
                if maps[0].get("selected_observation_id") != expected_obs:
                    failures.append("context_image_observation_stale")
                if not re.search(r"\bnot current (?:camera )?visibility\b", maps[0].get("label", "")):
                    failures.append("context_image_visibility_boundary_missing")
            elif f"updated_at_observation_id={expected_obs} " not in maps[0].get("label", ""):
                failures.append("context_image_observation_stale")
        projection = model_round.get("history_projection") or {}
        latest = [m for m in projection.get("selected_visual_evidence", []) if m.get("selection_tier") == "latest_group"]
        if len(latest) != 1:
            failures.append("latest_map_projection_not_unique")
        elif maps and (maps[0].get("frame_id") or maps[0].get("selected_frame_id")) != latest[0].get("frame_id"):
            failures.append("delivered_map_not_latest_projection")
        names = {t["function"]["name"] for t in model_round["request"]["tools"]}
        expected = {"skill_pixel_depth_measure_pixel_depth", "skill_place_memory_save_place", "skill_place_memory_inspect_place"}
        if require_place_list:
            expected.add("skill_place_memory_list_places")
        if not expected <= names or "skill_visual_grounding_measure_visible_depth" in names:
            failures.append("pixel_place_tool_surface_mismatch")
    plans = [t["function"] for t in model_round["request"]["tools"] if t["function"]["name"] == "runtime_update_plan"]
    if plan_mode == "off":
        if plans:
            failures.append("off_mode_plan_tool_exposed")
        if (model_round.get("current_task_state") or {}).get("task_plan") is not None:
            failures.append("off_mode_saved_plan_not_empty")
        texts = "\n".join(m["content"] for m in messages if m["role"] == "system" and isinstance(m.get("content"), str))
        if any(marker in texts for marker in ("When a plan is useful", "If a plan exists", "or one plan update", "committed plan")):
            failures.append("off_mode_plan_prompt_retained")
    elif plan_mode == "optional":
        if len(plans) != 1 or plans[0].get("strict") is not True or set(plans[0]["parameters"]["properties"]) != {"operation", "explanation", "steps"}:
            failures.append("live_plan_contract_mismatch")
    else:
        failures.append("unknown_plan_mode")
    return failures, image_count


def report_link_checks(root):
    """Check generated local evidence links without opening external services."""
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []

        def handle_starttag(self, tag, attrs):
            self.targets.extend(value for key, value in attrs if key in {"href", "src"} and value)

    checked = 0
    failures = []
    for page in [root / "comparison.html", root / "context_modules.html", *root.glob("case_diagnostics_*.html")]:
        if not page.exists():
            failures.append({"source": str(page), "reason": "report_page_missing"})
            continue
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in parser.targets:
            parts = urlsplit(target)
            if not parts.path or parts.scheme not in {"", "file"}:
                continue
            path = Path(url2pathname(parts.path)) if parts.scheme == "file" else page.parent / unquote(parts.path)
            checked += 1
            if not path.exists():
                failures.append({"source": str(page), "target": target, "reason": "report_evidence_link_missing"})
    return {"checked": checked, "failures": failures}


def accept(root):
    manifest = read(root / "manifest.json")
    from tests.manual.aerial_vln_plan_context_probe import read_experiment_progress
    progress = read_experiment_progress(root)
    failures = []
    if manifest.get("execution_mode") == "parallel":
        from evaluation.benchmarks.aerial_vln.parallel_campaign import verify_overlay
        verify_overlay(read(root / "parallel/execution_overlay.json"), manifest)
        gate = read(root / "parallel/trial_check.json")
        if not gate.get("passed") or gate.get("episodes") != 4:
            failures.append({"reason": "parallel_technical_gate_failed"})
        for row in progress["cases"]:
            for attempt in row["attempts"]:
                expected = {"A": (0, 35001, 30001), "B": (2, 35002, 30002)}.get(attempt.get("lane"))
                actual = tuple(attempt.get(key) for key in ("remote_gpu", "local_port", "remote_port"))
                if expected != actual:
                    failures.append({"reason": "lane_isolation_mismatch", "task": row["task"]})
    for relative, expected in manifest["source_hashes"].items():
        if digest(root / "source" / relative) != expected:
            failures.append({"source": relative, "reason": "frozen_source_changed"})
    admission = read(root / "admission.json")
    plan_mode = manifest.get("plan_mode", "optional")
    if not admission_passed(admission, plan_mode):
        failures.append({"reason": "admission_not_passed_first_attempt"})
    if not read(root / "historical_request_replay.json")["passed"]:
        failures.append({"reason": "historical_replay_failed"})
    if progress["status"] != "complete" or len(progress["cases"]) != len(manifest["tasks"]):
        failures.append({"reason": "manifest_cases_not_complete"})
    require_map = "capabilities.place_memory" in manifest.get("plugins", {}).get("plugins", {}).get("enabled", [])
    if require_map and not read(root / "pixel_place_admission.json")["passed"]:
        failures.append({"reason": "pixel_place_admission_failed"})
    if manifest.get("place_memory_admission_required"):
        if read(root / "place_memory_admission.json").get("status") != "passed":
            failures.append({"reason": "place_memory_provider_admission_failed"})
        if read(root / "place_memory_audit.json").get("failures"):
            failures.append({"reason": "place_memory_temporal_audit_failed"})
    cases = []; requests = 0; peak_images = 0; limitations = []
    for index, row in enumerate(progress["cases"], 1):
        terminal = row["status"] in {"completed", "model_error"}
        if not terminal:
            failures.append({"case": index, "reason": "case_not_terminal"})
        for attempt in row["attempts"]:
            directory = Path(attempt["directory"])
            if not read(directory / "cleanup.json")["verified"]:
                failures.append({"case": index, "attempt": attempt["number"], "reason": "cleanup_not_verified"})
            environment = read(directory / "environment.json")
            if Path(environment["sys_executable"]).resolve() != Path("F:/conda_env/flyagent/python.exe").resolve() or not environment["cuda_available"]:
                failures.append({"case": index, "reason": "python_or_cuda_mismatch"})
        result = row["attempts"][-1].get("result") or {}
        if not result.get("run_dir"):
            continue
        if not (root / f"case_diagnostics_{index:02d}.html").is_file():
            failures.append({"case": index, "reason": "case_diagnostics_page_missing"})
        run = Path(result["run_dir"])
        transcript = run / "memory/agent_transcript.jsonl"
        before = digest(transcript)
        events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        calls = {}; plan = TaskPlan(); case_requests = 0
        for event in events:
            p = event["payload"]
            if event["kind"] == "assistant_turn":
                calls.update({c["tool_call_id"]: c for c in p.get("tool_calls", [])})
            elif event["kind"] == "tool_result" and p["tool_name"] == "runtime.update_plan" and p["status"] == "ok":
                if plan_mode == "off":
                    failures.append({"case": index, "reason": "off_mode_plan_committed"})
                candidate = plan.updated(calls[p["tool_call_id"]]["args"])
                if candidate.to_dict() != p["data"]["plan"] or p["data"]["previous_revision"] != plan.revision:
                    failures.append({"case": index, "reason": "applied_plan_snapshot_mismatch"})
                plan = candidate
        for path in sorted(run.glob("debug/tool_loop/*/decision_*.json")):
            for model_round in read(path).get("model_rounds", []):
                issues, images = request_checks(model_round, require_map=require_map, plan_mode=plan_mode,
                    require_place_list=manifest.get("place_memory_admission_required", False))
                for media in model_round.get("request_media") or []:
                    asset = run / media.get("asset_ref", "")
                    if not asset.is_file() or digest(asset) != media.get("sha256"):
                        issues.append("captured_image_bytes_missing_or_changed")
                requests += 1; case_requests += 1; peak_images = max(peak_images, images)
                failures.extend({"case": index, "source": str(path), "reason": reason} for reason in issues)
        from tests.manual.aerial_vln_plan_context_probe import run_stats
        stats = run_stats(result)
        for missing in (stats.get("capability_integrity") or {}).get("unavailable_checks", []):
            limitations.append({"case": index, **missing})
        if require_map and (stats.get("capability_integrity") or {}).get("failures"):
            failures.append({"case": index, "reason": "capability_integrity_failure", "details": stats["capability_integrity"]["failures"]})
        maximum = stats.get("max_calls_in_observed_window")
        if require_map and maximum is not None and maximum > 1:
            failures.append({"case": index, "reason": "physical_window_not_single"})
        unchanged = digest(transcript) == before
        if not unchanged:
            failures.append({"case": index, "reason": "transcript_changed"})
        cases.append({"index": index, "requests": case_requests, "revision": plan.revision,
                      "transcript_sha256": before, "transcript_unchanged": unchanged})
    for filename in ("comparison.md", "comparison.html", "context_modules.html"):
        text = (root / filename).read_text(encoding="utf-8")
        if "\ufffd" in text or "???" in text:
            failures.append({"source": filename, "reason": "encoding_replacement_text"})
    links = report_link_checks(root)
    failures.extend(links["failures"])
    data = {"passed": not failures, "failures": failures, "limitations": limitations, "requests": requests,
            "peak_images": peak_images, "cases": cases, "source_files": len(manifest["source_hashes"]),
            "report_links_checked": links["checked"],
            "reader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(root / "acceptance.json", data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    value = accept(parser.parse_args().root.resolve())
    print(json.dumps({k: v for k, v in value.items() if k != "cases"}, ensure_ascii=False))
    raise SystemExit(0 if value["passed"] else 2)
