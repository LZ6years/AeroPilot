"""Offline contracts for real ten-flight orchestration and historical accounting."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.manual import aerial_vln_plan_context_probe as probe


def tasks():
    return {str(i): {"task": {"split": "val_seen", "episode_id": f"case{i}",
                              "scene_id": str(i), "difficulty": "low"},
                     "status": "scored", "result": {"metrics": {"success": bool(i%2)}}}
            for i in range(10)}


def cohort_fixture():
    """Completed baseline with original cases plus score-independent high pools."""
    progress = {"tasks": tasks()}
    for scene in (2, 5, 10, 12, 14):
        for index in range(4):
            cid = f"high_{scene}_{index}"
            progress["tasks"][cid] = {"status": "scored", "task": {
                "split": "val_seen", "episode_id": cid, "scene_id": str(scene), "difficulty": "high"}}
    hard = probe.select_hard_cases(progress)
    original = [value["task"] for value in tasks().values()]
    previous = {"status": "complete", "cases": [
        {"task": task, "status": "completed", "attempts": [{"result": {"episode_id": task["episode_id"]}}]}
        for task in original + hard]}
    return progress, original, hard, previous


def test_hard_only_selection_and_previous_twenty_case_matching():
    progress, original, hard, previous = cohort_fixture()
    ids = [task["episode_id"] for task in hard]
    selected, cohorts = probe.select_cohorts(None, ids, progress, hard_only=True)
    assert selected == hard and len(selected) == 10
    assert cohorts == [{"name": "hard10", "episode_ids": ids}]
    matched = probe.match_previous_cohort(previous, selected, allow_subset=True)
    assert [row["task"] for row in matched] == hard
    combined, groups = probe.select_cohorts([t["episode_id"] for t in original], ids, progress)
    assert combined == original + hard
    assert [g["name"] for g in groups] == ["original10", "hard10"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "reordered", "unfinished", "no_result"])
def test_previous_cohort_rejects_incomplete_or_ambiguous_matching(problem):
    progress, original, hard, previous = cohort_fixture()
    if problem == "missing":
        previous["cases"].pop()
    elif problem == "duplicate":
        previous["cases"].append(previous["cases"][-1])
    elif problem == "reordered":
        previous["cases"][-2:] = reversed(previous["cases"][-2:])
    elif problem == "unfinished":
        previous["cases"][-1]["status"] = "pending"
    else:
        previous["cases"][-1]["attempts"] = []
    with pytest.raises(ValueError):
        probe.match_previous_cohort(previous, hard, allow_subset=True)


def test_hard_only_requires_locked_file_and_default_matching_stays_exact():
    progress, original, hard, previous = cohort_fixture()
    with pytest.raises(ValueError, match="requires"):
        probe.select_cohorts(None, None, progress, hard_only=True)
    with pytest.raises(ValueError, match="sampling"):
        probe.select_cohorts(None, [t["episode_id"] for t in reversed(hard)], progress, hard_only=True)
    with pytest.raises(ValueError, match="ordered"):
        probe.match_previous_cohort(previous, original)
    previous["cases"] = previous["cases"][:10]
    assert len(probe.match_previous_cohort(previous, original)) == 10


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("subset_indices", [None, [5, 8], [5]])
def test_prepare_hard_only_freezes_only_selected_ids_and_cohort_baseline(tmp_path, monkeypatch, subset_indices, combined):
    progress, original, hard, previous = cohort_fixture()
    work = tmp_path / "work"; baseline = tmp_path / "baseline"; old = tmp_path / "old20"
    root = tmp_path / "candidate"; profile = work / "configs/pixel.json"
    ids_file = work / "hard.json"
    probe.write(ids_file, [task["episode_id"] for task in hard])
    probe.write(profile, {"plugins": {"enabled": ["capabilities.pixel_depth"]}})
    probe.write(baseline / "manifest.json", {"model": {}, "configs": {"val_seen": {}}, "source_hashes": {}})
    probe.write(baseline / "progress.json", progress)
    probe.write(old / "manifest.json", {"source_hashes": {}})
    probe.write(old / "progress.json", previous)
    monkeypatch.setattr(probe, "WORKSPACE", work)
    monkeypatch.setattr(probe, "environment_diagnostics", lambda: {"cuda_available": True})
    monkeypatch.setattr(probe, "load_model_config", lambda _: SimpleNamespace(redacted_summary=lambda: {}))
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: b"")
    monkeypatch.setattr(probe, "report", lambda *args, **kwargs: None)
    reference = tmp_path / "reference"
    probe.write(reference / "manifest.json", {"source_hashes": {}})
    probe.write(reference / "progress.json", previous)
    subset_file = work / "subset.json" if subset_indices is not None else None
    chosen = [hard[i] for i in subset_indices] if subset_indices is not None else (original + hard if combined else hard)
    original_file = work / "original.json"
    probe.write(original_file, [t["episode_id"] for t in original])
    if subset_file:
        probe.write(subset_file, [t["episode_id"] for t in chosen])
    probe.prepare(root, baseline, original_file, old, profile, ids_file, hard_only=not combined,
                  reference_roots=[reference, reference], subset_case_ids_file=subset_file)
    manifest = probe.read(root / "manifest.json")
    assert manifest["tasks"] == chosen
    assert manifest["config"]["episode_ids"] == [t["episode_id"] for t in chosen]
    assert manifest["case_file"] == str(original_file if combined else ids_file)
    assert manifest["selection_mode"] == (("original_plus_hard" if combined and not subset_file else "original_only" if combined else "hard_only") + ("_subset" if subset_file else ""))
    assert manifest["technical_gate_indices"] == ( [0, 1, 10] if combined and not subset_file else list(range(min(2, len(chosen)))))
    group = {"name": "hard10", "episode_ids": [t["episode_id"] for t in chosen],
             "primary_baseline": str(old), "secondary_baseline": str(baseline)}
    if subset_file:
        group["source_episode_ids"] = [t["episode_id"] for t in hard]
    expected_groups = ([{"name": "original10", "episode_ids": [t["episode_id"] for t in original], "primary_baseline": str(old), "secondary_baseline": str(baseline)}] if combined and not subset_file else []) + [group]
    if combined and not subset_file:
        expected_groups[-1]["episode_ids"] = [t["episode_id"] for t in hard]
    assert manifest["cohorts"] == expected_groups
    assert manifest["overall_primary_baseline"] == str(old)
    assert probe.digest(old / "progress.json") == manifest["previous_progress_sha256"]
    assert len(probe.read(root / "progress.json")["cases"]) == len(chosen)
    assert manifest["plan_mode"] == "optional" and manifest["experiment_overrides"] == []
    assert len(manifest["reference_roots"]) == 1
    assert manifest["reference_roots"][0]["progress_sha256"] == probe.digest(reference / "progress.json")
    probe.guard(root)
    if subset_file:
        assert manifest["subset_case_file_sha256"] == probe.digest(subset_file)
        probe.write(subset_file, [t["episode_id"] for t in reversed(chosen)] + ["changed"])
        with pytest.raises(ValueError, match="subset_case_file"):
            probe.guard(root)


def test_subset_preserves_locked_source_identity_and_matches_by_episode():
    progress, _, hard, previous = cohort_fixture()
    tasks, groups = probe.select_cohorts(None, [t["episode_id"] for t in hard], progress, hard_only=True)
    ids = [hard[i]["episode_id"] for i in (5, 8)]
    selected, cohorts = probe.select_subset(tasks, groups, ids)
    assert selected == [hard[5], hard[8]]
    assert cohorts[0]["source_episode_ids"] == groups[0]["episode_ids"]
    assert cohorts[0]["episode_ids"] == ids
    assert [r["task"] for r in probe.match_previous_cohort(previous, selected, allow_subset=True)] == selected
    assert len(groups[0]["episode_ids"]) == 10


@pytest.mark.parametrize("problem", ["empty", "duplicate", "unknown", "reordered", "non_text"])
def test_subset_rejects_invalid_identity_or_order(problem):
    progress, _, hard, _ = cohort_fixture()
    tasks, groups = probe.select_cohorts(None, [t["episode_id"] for t in hard], progress, hard_only=True)
    first, last = hard[0]["episode_id"], hard[-1]["episode_id"]
    ids = {"empty": [], "duplicate": [first, first], "unknown": ["unknown"],
           "reordered": [last, first], "non_text": [None]}[problem]
    with pytest.raises(ValueError):
        probe.select_subset(tasks, groups, ids)


def test_subset_report_pairs_two_identities_without_claiming_ten_flights(tmp_path, monkeypatch):
    from tests.manual import aerial_vln_context_module_audit

    root, previous, baseline = (tmp_path / name for name in ("current", "previous", "baseline"))
    progress, _, hard, old = cohort_fixture()
    selected = [hard[5], hard[8]]
    # Sampling pools need not have report fixtures. Keep only the scored
    # historical rows whose complete results are constructed below.
    progress = {"tasks": {}}
    current = []
    for row in old["cases"]:
        task = row["task"]
        result = {"episode_id": task["episode_id"], "status": "ok", "completed": True,
                  "stopped": False, "error": None, "dataset_split": "val_seen",
                  "metrics": {"success": False, "oracle_success": False, "distance_to_goal": 30},
                  "run_dir": str(previous / task["episode_id"]), "audit_path": str(previous / "audit.json")}
        row["attempts"][0]["result"] = result
        progress["tasks"][task["episode_id"]] = {"task": task, "status": "scored", "result": result}
        if task in selected:
            current.append({**row, "attempts": [{"number": 1, "status": "completed", "result": result,
                                               "directory": str(root / task["episode_id"])}]})
    probe.write(baseline / "progress.json", progress)
    probe.write(previous / "progress.json", old)
    probe.write(root / "progress.json", {"status": "complete", "cases": current})
    probe.write(root / "manifest.json", {"baseline": str(baseline), "previous_root": str(previous),
        "plan_mode": "optional", "limits": {}, "cohorts": [{"name": "hard10",
            "episode_ids": [t["episode_id"] for t in selected],
            "source_episode_ids": [t["episode_id"] for t in hard],
            "primary_baseline": str(previous), "secondary_baseline": str(baseline)}]})
    probe.write(root / "history_inventory.json", {"records": [], "read_errors": []})
    monkeypatch.setattr(probe, "run_stats", lambda _: {})
    monkeypatch.setattr(aerial_vln_context_module_audit, "write_module_report", lambda *args: None)
    before = (previous / "progress.json").read_bytes()
    data = probe.report(root)
    assert data["selected_cases"] == data["terminal_cases"] == 2
    assert [p["task"] for p in data["pairs"]] == selected
    assert [p["baseline"]["episode_id"] for p in data["pairs"]] == [t["episode_id"] for t in selected]
    assert data["cohorts"][0]["selected_cases"] == 2
    for filename in ("comparison.md", "comparison.html"):
        text = (root / filename).read_text(encoding="utf-8")
        assert "有序子集，共 2 例" in text and "2/2" in text
        assert "困难十例：沿用锁定" not in text
    assert (previous / "progress.json").read_bytes() == before

    # A provider error is terminal but unscored; never display it as a failure
    # or compare a one-case current denominator with two historical cases.
    current[-1]["status"] = "model_error"
    current[-1]["attempts"][-1]["result"] = {
        **current[-1]["attempts"][-1]["result"], "metrics": {}, "error": "provider timeout"}
    probe.write(root / "progress.json", {"status": "complete", "cases": current})
    data = probe.report(root)
    assert data["terminal_cases"] == 2 and data["terminal_unscored_cases"] == [2]
    assert data["scored_pair_comparison"]["indices"] == [1]
    assert data["cohorts"][0]["scored_pair_comparison"]["indices"] == [1]
    assert probe.metric_rate(data["current"], "success") == "0/1"
    assert probe.metric_rate(data["scored_pair_comparison"]["baseline"], "success") == "0/1"
    assert "有效配对 1 例" in (root / "comparison.md").read_text(encoding="utf-8")
    assert (previous / "progress.json").read_bytes() == before


def test_selection_preserves_exact_order_without_reading_scores():
    ids = [f"case{i}" for i in reversed(range(10))]
    selected = probe.select_cases(ids, {"tasks": tasks()})
    assert [t["episode_id"] for t in selected] == ids
    assert all("result" not in t for t in selected)


@pytest.mark.parametrize("ids", [["case0"]*10, ["case0"], [f"unknown{i}" for i in range(10)]])
def test_invalid_or_unscored_case_selection_rejected(ids):
    with pytest.raises(ValueError):
        probe.select_cases(ids, {"tasks": tasks()})


def test_deduplication_retains_distinct_attempts_and_aliases():
    def row(run, source, success=False):
        return {"source":source,"result":{"episode_id":"same","run_dir":run,
                                         "metrics":{"success":success}}}
    rows = probe.deduplicate([row("F:/old/run1","one"),row("H:/export/run1","copy"),
                              row("H:/old/run2","retry",True)])
    assert len(rows)==2
    assert rows[0]["source_aliases"]==["one","copy"]
    assert rows[0]["result"]["metrics"]["success"] is False


def test_missing_tokens_stay_unknown_and_actual_zero_is_counted():
    assert probe.token_summary([None,{}])["input_tokens_sum"] is None
    s=probe.token_summary([None,{"prompt_tokens":0},{"input_tokens":20}])
    assert s=={"requests":3,"requests_with_usage":2,"input_tokens_sum":20,
               "input_tokens_mean":10,"input_tokens_peak":20, "requests_with_output_usage":0,
               "output_tokens_sum":None,"output_tokens_mean":None,"output_tokens_peak":None}


@pytest.mark.parametrize("status", ["completed","model_error"])
def test_completed_navigation_and_model_outcomes_never_repeat(status):
    assert probe.next_attempt({"status":status,"attempts":[{}]},True) is None


def test_only_verified_infrastructure_interruption_can_resume():
    row={"status":"infrastructure_error","attempts":[{"cleanup_verified":False}]}
    with pytest.raises(ValueError): probe.next_attempt(row,True)
    row["attempts"][0]["cleanup_verified"]=True
    with pytest.raises(ValueError): probe.next_attempt(row,False)
    assert probe.next_attempt(row,True)==2
    row["attempts"]*=3
    with pytest.raises(ValueError): probe.next_attempt(row,True)


def test_freeze_includes_compaction_prompts_and_omits_secrets(tmp_path):
    workspace=tmp_path/"work";source=tmp_path/"snapshot"
    prompt=workspace/"src/agent/context/compression/prompts/history.txt"
    prompt.parent.mkdir(parents=True);prompt.write_text("uncertainty",encoding="utf-8")
    (workspace/".env").write_text("SECRET=private",encoding="utf-8")
    hashes=probe.freeze_source(workspace,source)
    assert "src/agent/context/compression/prompts/history.txt" in hashes
    assert not (source/".env").exists()


def test_guard_rejects_frozen_source_and_configuration_drift(tmp_path,monkeypatch):
    source=tmp_path/"source/src/a.py";source.parent.mkdir(parents=True);source.write_text("initial")
    case=tmp_path/"ids.json";probe.write(case,["episode"])
    baseline=tmp_path/"baseline";probe.write(baseline/"manifest.json",{})
    manifest={"source_hashes":{"src/a.py":probe.digest(source)},"case_file":str(case),
              "case_file_sha256":probe.digest(case),"baseline":str(baseline),
              "baseline_manifest_sha256":probe.digest(baseline/"manifest.json"),
              "model":{"model_name":"same"},"config":{"env_path":"secret.env"}}
    probe.write(tmp_path/"manifest.json",manifest)
    monkeypatch.setattr(probe,"load_model_config",lambda _:SimpleNamespace(redacted_summary=lambda:{"model_name":"same"}))
    assert probe.guard(tmp_path)==manifest
    source.write_text("changed")
    with pytest.raises(ValueError,match="Frozen source changed"):probe.guard(tmp_path)
    source.write_text("initial")
    monkeypatch.setattr(probe,"load_model_config",lambda _:SimpleNamespace(redacted_summary=lambda:{"model_name":"other"}))
    with pytest.raises(ValueError,match="Provider configuration"):probe.guard(tmp_path)


def test_history_excludes_snapshot_and_current_run_but_retains_failed_attempt(tmp_path):
    old={"episode_id":"real","dataset_split":"val_seen","run_dir":"F:/runs/real_1",
         "status":"error","infrastructure_error":{"stage":"provider"},"metrics":{}}
    probe.write(tmp_path/"old/cases/real/case_result.json",old)
    probe.write(tmp_path/"old/summary.json",{"cases":[old]})
    probe.write(tmp_path/"source/case_result.json",{**old,"run_dir":"fake"})
    excluded=tmp_path/"current"
    probe.write(excluded/"case_result.json",{**old,"run_dir":"new"})
    inventory=probe.history_inventory([tmp_path],excluded.resolve(),{"real"})
    assert len(inventory["records"])==1
    assert inventory["records"][0]["result"]["infrastructure_error"]
    assert len(inventory["records"][0]["source_aliases"])==2


def test_summary_exposes_metric_coverage_and_failure_counts():
    result=probe.aggregate([{"status":"ok","metrics":{"success":True}},
                            {"status":"error","metrics":{}}])
    assert result["cases"]==2
    assert result["success"]=={"count":1,"mean":1,"sum":1}
    assert result["ndtw"]["mean"] is None


@pytest.mark.parametrize("fail_first", [False, True])
def test_admission_requires_three_first_attempt_lifecycles(tmp_path, monkeypatch, fail_first):
    import copy
    import json
    import src.llm
    from tests.manual import aerial_vln_remote_readiness
    root=tmp_path/"probe"; baseline=tmp_path/"old"; run=tmp_path/"run"
    task={"episode_id":"one","split":"val_seen"}
    manifest={"config":{"dataset_root":str(tmp_path),"execution_mode":"managed",
                        "env_path":"private.env"},"baseline":str(baseline),"tasks":[task]}
    probe.write(baseline/"progress.json",{"tasks":{"one":{"task":task,"result":{"run_dir":str(run)}}}})
    image=run/"artifacts/first/rgb/image.png"; image.parent.mkdir(parents=True); image.write_bytes(b"test-image")
    prompt=root/"source/src/agent/planning/prompts/agent_tool_loop_system.txt"
    prompt.parent.mkdir(parents=True); prompt.write_text("test protocol")
    probe.write(root/"source/configs/plugins.json",{})
    model=SimpleNamespace(model_name="test", redacted_summary=lambda:{"model_name":"test"},
                          to_provider_settings=lambda:None,generation_parameters=lambda:{})
    first={"id":"s1","description":"inspect","status":"pending"}
    second={"id":"s2","description":"verify","status":"pending"}
    advanced=[{**first,"status":"completed"},{**second,"status":"in_progress"}]
    corrected=[advanced[0],{**advanced[1],"description":"Verify visible roof appearance"}]
    group=[{"operation":"replace","explanation":None,"steps":[first,second]},
           {"operation":"patch","explanation":None,"steps":[{"id":"s1","description":None,"status":"completed"},{"id":"s2","description":None,"status":"in_progress"}]},
           {"operation":"patch","explanation":None,"steps":[{"id":"s2","description":corrected[1]["description"],"status":None}]},
           {"operation":"replace","explanation":None,"steps":list(reversed(corrected))},
           {"operation":"patch","explanation":None,"steps":[{"id":"s2","description":None,"status":"cancelled"}]}]
    payloads=copy.deepcopy(group*3)
    if fail_first: payloads[0]={"steps":[first,second],"updates":[]}
    calls=[]
    def response(request):
        calls.append(request); args=payloads.pop(0)
        return SimpleNamespace(data={"model":"test"},text="",usage={"prompt_tokens":10},transport_trace={},
            tool_calls=[{"id":str(len(calls)),"type":"function", "function":{"name":"runtime_update_plan","arguments":json.dumps(args)}}])
    monkeypatch.setattr(probe,"guard",lambda _:manifest)
    monkeypatch.setattr(probe,"load_credentials",lambda _:None)
    monkeypatch.setattr(probe,"load_model_config",lambda _:model)
    monkeypatch.setattr(aerial_vln_remote_readiness,"inspect_remote_readiness",lambda _:{"ready":True})
    monkeypatch.setattr(src.llm,"create_model_client",lambda _:SimpleNamespace(request=response))
    if fail_first:
        with pytest.raises(RuntimeError,match="not started"): probe.admit(root)
        record=probe.read(root/"admission.json")
        assert record["status"]=="failed" and len(calls)==1
        assert record["receipts"][0]["plan"]["revision"]==0
        assert "updates" in record["receipts"][0]["args"]
    else:
        record=probe.admit(root)
        assert record["status"]=="passed" and record["groups_completed"]==3 and len(calls)==15
        assert [r["plan"]["revision"] for r in record["receipts"]]==[1,2,3,4,5]*3
        assert all(r["attempt"]==1 for r in record["receipts"])


def test_recorded_plan_gaps_and_usage_do_not_duplicate_background_calls(tmp_path):
    import json
    run=tmp_path/"attempt/runs/native";memory=run/"memory";memory.mkdir(parents=True)
    events=[
        {"kind":"assistant_turn","payload":{"tool_calls":[{"tool_call_id":"p1","tool_name":"runtime.update_plan","args":{"steps":[]}}]}},
        {"kind":"tool_result","payload":{"tool_call_id":"p1","tool_name":"runtime.update_plan","status":"ok","data":{"plan":{"revision":1}}}},
        {"kind":"tool_result","payload":{"tool_name":"uav.move_body","status":"ok"}},
        {"kind":"protocol_feedback","payload":{"message":"invalid velocity"}},
        {"kind":"assistant_turn","payload":{"tool_calls":[{"tool_call_id":"p2","tool_name":"runtime.update_plan","args":{"updates":[{"id":"s","status":"completed"}]}}]}},
        {"kind":"tool_result","payload":{"tool_call_id":"p2","tool_name":"runtime.update_plan","status":"ok","data":{"plan":{"revision":2}}}},
    ]
    (memory/"agent_transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events),encoding="utf-8")
    probe.write(tmp_path/"attempt/process.json",{"elapsed_seconds":12.5})
    probe.write(run/"debug/tool_loop/task/decision_0001.json",{"model_rounds":[{"response":{"usage":{"prompt_tokens":10}}},{"response":{}}]})
    probe.write(run/"debug/history_compaction_calls/id.finished.json",{"call_id":"id","status":"ok","usage":{"input_tokens":44}})
    probe.write(run/"debug/history_compaction_calls/id.started.json",{"call_id":"id","status":"started","usage":None})
    stats=probe.run_stats({"run_dir":str(run)})
    assert stats["plan_commits"]==2 and stats["incremental_commits"]==1
    assert stats["flight_actions_between_updates"]==[1]
    assert stats["final_revision"]==2 and stats["protocol_rejections"]==1
    assert stats["elapsed_seconds"]==12.5
    assert stats["navigation_usage"]["input_tokens_sum"]==10
    assert stats["navigation_usage"]["requests_with_usage"]==1
    assert stats["compression_usage"]["requests"]==1
    assert stats["compression_usage"]["input_tokens_sum"]==44


def test_diagnostic_statistics_distinguish_affected_events_from_one_cause(tmp_path):
    import json
    run = tmp_path / "attempt/runs/native"
    (run / "memory").mkdir(parents=True)
    diagnostic = {"code": "plan_update_invalid", "phase": "validation", "execution": "not_started",
                  "source_tool_call_id": "plan", "scope": "call"}
    events = [
        {"kind": "assistant_turn", "payload": {"turn_id": "turn", "tool_calls": [
            {"tool_call_id": "plan", "tool_name": "runtime.update_plan", "args": {}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "plan", "tool_name": "runtime.update_plan",
            "status": "rejected", "diagnostic": {**diagnostic, "role": "source"}}},
        {"kind": "tool_result", "payload": {"tool_call_id": "read", "tool_name": "knowledge.load",
            "status": "rejected", "diagnostic": {**diagnostic, "role": "affected"}}},
        {"kind": "protocol_feedback", "payload": {"message": "No call executed", "diagnostic": diagnostic}},
    ]
    (run / "memory/agent_transcript.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    stats = probe.run_stats({"run_dir": str(run)})
    assert stats["diagnostic_counts"] == {"plan_update_invalid": 3}
    assert stats["diagnostic_incident_counts"] == {"plan_update_invalid": 1}
    assert stats["knowledge_load_results"] == {"rejected": 1}
    assert probe.stats_overview([])["plan_rejections"] is None


def test_unknown_models_still_keep_experiments_separate(tmp_path):
    for name in ("old_one","old_two"):
        probe.write(tmp_path/name/"manifest.json",{"benchmark":"aerial_vln","dataset_split":"val_seen"})
        probe.write(tmp_path/name/"cases/episode/case_result.json",{
            "episode_id":"episode","dataset_split":"val_seen","run_dir":f"F:/runs/{name}",
            "status":"ok","metrics":{"success":False}})
    inventory=probe.history_inventory([tmp_path],tmp_path/"excluded",{"episode"})
    assert {r["experiment"] for r in inventory["records"]}=={str(tmp_path/"old_one"),str(tmp_path/"old_two")}
    assert all(r["model"] is None for r in inventory["records"])


def test_usage_overview_weights_requests_and_preserves_unknown_baseline():
    observed = probe.stats_overview([
        {"navigation_usage":probe.token_summary([{"prompt_tokens":10},{"prompt_tokens":20}])},
        {"navigation_usage":probe.token_summary([{"input_tokens":60},None])},
        {"navigation_usage":probe.token_summary([])},
    ])
    usage = observed["navigation_usage"]
    assert usage["input_tokens_sum"] == 90 and usage["input_tokens_mean"] == 30
    assert usage["captured_requests"] == 4 and usage["requests_with_usage"] == 3
    assert usage["cases_with_usage"] == 2 and usage["input_tokens_peak"] == 60
    missing = probe.stats_overview([{"navigation_usage":probe.token_summary([])}])
    assert missing["navigation_usage"]["input_tokens_sum"] is None
    assert missing["elapsed_seconds"]["mean"] is None


def test_campaign_summary_is_an_index_not_an_extra_flight(tmp_path):
    probe.write(tmp_path/"old/manifest.json",{"model":{"model_name":"old"}})
    attempts = [{"episode_id":"episode","dataset_split":"val_seen","run_dir":f"F:/runs/attempt_{n}",
                 "status":"ok","metrics":{"success":n==2}} for n in (1,2)]
    probe.write(tmp_path/"old/progress.json",{"tasks":{"episode":{"attempts":[{"result":r} for r in attempts],"result":attempts[-1]}}})
    probe.write(tmp_path/"old/val_seen/summary.json",{"dataset_split":"val_seen","cases":[{
        "episode_id":"episode","status":"ok","metrics":{"success":True}}]})
    inventory = probe.history_inventory([tmp_path],tmp_path/"excluded",{"episode"})
    assert len(inventory["records"]) == 2
    assert {r["result"]["run_dir"] for r in inventory["records"]} == {"F:/runs/attempt_1","F:/runs/attempt_2"}
    assert len(inventory["summary_indexes"]) == 1


def test_declared_legacy_dataset_outside_baseline_is_background_not_a_fixture(tmp_path):
    cid = "32EYX73OY97OEHFZ5WHW4VDH3OSRUM"
    probe.write(tmp_path/"legacy/manifest.json",{"benchmark":"aerial_vln","dataset_version":"AerialVLN-v8"})
    for episode in (cid,"ep_1"):
        probe.write(tmp_path/f"legacy/cases/{episode}/case_result.json",{
            "episode_id":episode,"run_dir":f"F:/runs/{episode}","status":"stopped","metrics":{"success":False}})
    inventory = probe.history_inventory([tmp_path],tmp_path/"excluded",{"primary"})
    assert len(inventory["records"]) == 1
    assert inventory["records"][0]["outside_primary_dataset"] is True
    assert inventory["records"][0]["dataset_version"] == "AerialVLN-v8"
    assert inventory["excluded_fixtures"][0]["episode_id"] == "ep_1"


def arrival_result(distance=22.0, **overrides):
    """Raw mission facts, independent of experiment status and formal SR."""
    return {"completed": True, "stopped": False, "error": None,
            "metrics": {"distance_to_goal": distance, "success": False}, **overrides}


@pytest.mark.parametrize("distance,expected", [
    (0, False), (19.99, False), (20, False), (20.0001, True),
    (20.87, True), (25, True), (25.0001, False),
    (None, None), (float("nan"), None), (float("inf"), None),
    (-1, None), (True, None), ("22", None),
])
def test_near_boundary_distance_is_exclusive_at_twenty_inclusive_at_twenty_five(distance, expected):
    result = arrival_result(distance)
    assert probe.near_boundary_stop(result) is expected
    assert result["metrics"]["success"] is False


@pytest.mark.parametrize("overrides", [
    {"completed": False}, {"stopped": True, "stop_source": "runtime.report_blocked"},
    {"error": "model failed"}, {"infrastructure_error": {"stage": "transport"}},
])
def test_near_boundary_requires_normal_completion(overrides):
    assert probe.near_boundary_stop(arrival_result(**overrides)) is False


@pytest.mark.parametrize("missing", ["completed", "stopped", "error"])
def test_near_boundary_missing_termination_is_unknown_even_for_finished_experiments(missing):
    result = arrival_result(status="completed")
    del result[missing]
    assert probe.near_boundary_stop(result) is None
    result["status"] = "stopped"
    assert probe.near_boundary_stop(result) is None
    assert probe.near_boundary_stop(None) is None


def test_near_boundary_summary_keeps_unknown_coverage_and_formal_outcomes():
    import copy
    rows = [arrival_result(), arrival_result(20, metrics={"success": True, "distance_to_goal": 20}),
            {"status": "error"}, arrival_result(stopped=True), arrival_result(None, error="failure")]
    before = copy.deepcopy(rows)
    annotated = [probe.annotated_result(row) for row in rows]
    summary = probe.aggregate(annotated)
    assert summary["near_boundary_stop"] == {"count": 1, "known_count": 3, "unknown_count": 2}
    assert summary["success"]["sum"] == 1
    assert rows == before and all("near_boundary_stop" not in row for row in rows)


def test_report_labels_both_baselines_all_history_and_preserves_raw_sources(tmp_path, monkeypatch):
    from tests.manual import aerial_vln_context_module_audit

    root = tmp_path / "current"
    previous = tmp_path / "previous"
    baseline = tmp_path / "baseline"
    old_tasks = {}; previous_cases = []; current_cases = []; history = []
    for index in range(10):
        task = {"episode_id": f"case{index}", "split": "val_seen", "difficulty": "low"}
        common = {"episode_id": task["episode_id"], "status": "ok", "dataset_split": "val_seen",
                  "audit_path": str(root / f"case{index}/audit.json"), "run_dir": str(root / f"run{index}")}
        # The first case stays a formal failure despite the new review label.
        current = {**arrival_result(22 if index == 0 else 30), **common}
        old = {**arrival_result(24 if index == 1 else 30), **common}
        secondary = {**arrival_result(25 if index == 2 else 30), **common}
        old_tasks[str(index)] = {"task": task, "status": "scored", "result": secondary}
        previous_cases.append({"task": task, "attempts": [{"result": old}]})
        current_cases.append({"task": task, "status": "completed", "attempts": [{
            "number": 1, "status": "completed", "directory": str(root / f"case{index}"), "result": current}]})
        history.append({"source": str(previous / f"case{index}.json"), "experiment": "history",
                        "source_aliases": [], "model": "same", "result": old})
    history.append({"source": str(previous / "legacy.json"), "experiment": "history", "model": "same",
                    "result": {"episode_id": "unmatched", "status": "completed", "metrics": {"distance_to_goal": 22}}})
    probe.write(baseline / "progress.json", {"tasks": old_tasks})
    probe.write(previous / "progress.json", {"cases": previous_cases})
    probe.write(root / "progress.json", {"status": "complete", "cases": current_cases})
    probe.write(root / "manifest.json", {"baseline": str(baseline), "previous_root": str(previous), "limits": {}})
    probe.write(root / "history_inventory.json", {"records": history, "read_errors": []})
    sources = [baseline / "progress.json", previous / "progress.json", root / "progress.json",
               root / "manifest.json", root / "history_inventory.json"]
    before = {path: path.read_bytes() for path in sources}
    monkeypatch.setattr(probe, "run_stats", lambda _: {})
    monkeypatch.setattr(aerial_vln_context_module_audit, "write_module_report", lambda *args: None)
    data = probe.report(root)
    assert data["current"]["success"]["sum"] == 0
    assert data["success_transitions"] == {"failure_to_failure": 10}
    for key in ("current", "ten_case_baseline", "secondary_ten_case_baseline"):
        assert data[key]["near_boundary_stop"] == {"count": 1, "known_count": 10, "unknown_count": 0}
    assert data["pairs"][0]["current"]["near_boundary_stop"] is True
    group = data["historical_groups"][0]
    assert group["summary"]["near_boundary_stop"] == {"count": 1, "known_count": 10, "unknown_count": 1}
    assert group["arrival_annotations"][-1]["near_boundary_stop"] is None
    assert group["paired_attempts"][0]["current_near_boundary_stop"] is True
    assert group["paired_attempts"][1]["historical_near_boundary_stop"] is True
    assert probe.read(root / "comparison.json")["arrival_annotation_contract"]["changes_formal_scoring"] is False
    for name in ("comparison.md", "comparison.html"):
        text = (root / name).read_text(encoding="utf-8")
        assert "当前临界到达 1 例" in text and "第二对照临界到达" in text
        assert "正式 SR" in text and "冻结请求" in text and "不反向作用于历史请求" in text and "未知" in text
        assert "\ufffd" not in text
    assert {path: path.read_bytes() for path in sources} == before
