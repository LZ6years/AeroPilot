"""Frozen paired seen-100 experiment on the existing supervised controller.

The baseline is read-only. Freeze copies an explicit feature overlay, excluding
unrelated working-tree changes. Sampling never consults navigation outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path("H:/FlyAgent-Eval/avln_pixel_map_100_20261009")
DEFAULT_BASELINE = Path("H:/FlyAgent-Eval/aerialvln_s_20261007")
DEFAULT_PREVIEW_OUTPUT = Path("H:/FlyAgent-Eval/avln_pixel_map_save_preview_4_20261009")
DEFAULT_SIMPLE_OUTPUT = Path("H:/FlyAgent-Eval/avln_simple_place_map_preview_4_20261010")
FEATURES = ["pixel_depth", "task_place_memory", "fixed_camera_viewpoint_guidance", "raw_bgr_to_rgb_decode",
            "proactive_place_annotation", "simple_place_map", "map_only_context", "cumulative_media_labels"]
SEED = 20261009
OVERLAY = (
    "src/adapters/airsim/capture.py",
    "src/runtime/task/turn_executor.py", "src/runtime/control/actions/terminal.py",
    "src/memory/context/contracts.py", "src/memory/context/media_refs.py",
    "src/agent/context/projection.py", "src/plugins/core/discovery.py",
    "src/agent/context/request_assembler.py",
    "plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/SKILL.md",
    "evaluation/benchmarks/aerial_vln/campaign.py",
    "evaluation/benchmarks/aerial_vln/parallel_campaign.py",
    "evaluation/benchmarks/aerial_vln/pixel_map_campaign.py",
    "evaluation/benchmarks/aerial_vln/capture_validation.py",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank(task):
    return hashlib.sha256(f"{SEED}:{task['scene_id']}:{task['difficulty']}:{task['episode_id']}".encode()).hexdigest()


def select_cases(tasks):
    """Hamilton quotas by scene within 34/33/33 difficulty strata."""
    chosen = []
    for difficulty, quota in (("low", 34), ("medium", 33), ("high", 33)):
        cells = defaultdict(list)
        for task in tasks:
            if task["difficulty"] == difficulty and task["split"] == "val_seen":
                cells[task["scene_id"]].append(task)
        total = sum(map(len, cells.values()))
        if total < quota:
            raise ValueError("insufficient difficulty population")
        allocations = {s: len(items) * quota // total for s, items in cells.items()}
        order = sorted(cells, key=lambda s: (-(len(cells[s]) * quota % total), int(s)))
        for scene in order[:quota - sum(allocations.values())]:
            allocations[scene] += 1
        for scene in sorted(cells, key=int):
            chosen.extend(sorted(cells[scene], key=rank)[:allocations[scene]])
    pilot = []
    # Five high-difficulty scenes first; then cover unseen pilot scenes while
    # maintaining five pilot cases per difficulty. This never uses outcomes.
    for difficulty in ("high", "low", "medium"):
        pool = [t for t in chosen if t["difficulty"] == difficulty]
        for scene in sorted({t["scene_id"] for t in pool} - {t["scene_id"] for t in pilot}, key=int):
            if sum(t["difficulty"] == difficulty for t in pilot) == 5:
                break
            pilot.append(min((t for t in pool if t["scene_id"] == scene), key=rank))
        for task in sorted(pool, key=rank):
            if sum(t["difficulty"] == difficulty for t in pilot) == 5:
                break
            if task not in pilot:
                pilot.append(task)
    if len(pilot) != 15 or {t["scene_id"] for t in pilot} != {t["scene_id"] for t in chosen}:
        raise ValueError("pilot coverage incomplete")
    return pilot + [t for t in chosen if t not in pilot], pilot


def select_preview_cases(tasks):
    """Select low/high/medium/high from the immutable ordering, never scores."""
    if len(tasks) != 100 or len({t["episode_id"] for t in tasks}) != 100:
        raise ValueError("preview requires the original 100 unique selected cases")
    low = next(t for t in tasks if t["difficulty"] == "low")
    medium = next(t for t in tasks if t["difficulty"] == "medium")
    high = [t for t in tasks if t["difficulty"] == "high"][:2]
    return [low, high[0], medium, high[1]]


def prepare(output, baseline, *, preview_from=None):
    from evaluation.benchmarks.core.report_io import write_json
    from evaluation.benchmarks.aerial_vln.campaign import environment_diagnostics
    from evaluation.benchmarks.aerial_vln.parallel_recovery import ParallelRecoveryQueue
    from src.agent.config import load_model_config
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output == baseline or baseline in output.parents or (output / "manifest.json").exists():
        raise ValueError("use a new independent experiment directory")
    base, progress = read(baseline / "manifest.json"), read(baseline / "progress.json")
    valid = [v for v in progress["tasks"].values() if v["task"]["split"] == "val_seen" and v["status"] == "scored"]
    if len(valid) != 333:
        raise ValueError("expected 333 immutable baseline scores")
    for relative, digest in base["source_hashes"].items():
        if hashlib.sha256((baseline / "source" / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("baseline source changed: " + relative)
    tasks, pilot = select_cases([v["task"] for v in valid])
    preview = None
    if preview_from is not None:
        origin = Path(preview_from).resolve()
        selected = read(origin / "selected_cases.json")
        original = read(origin / "manifest.json")
        if selected["seed"] != SEED or selected["cases"] != tasks or original["tasks"] != tasks:
            raise ValueError("original 100-case selection changed")
        tasks = select_preview_cases(selected["cases"])
        pilot = tasks
        preview = {"source_selection": str(origin / "selected_cases.json"),
                   "source_fingerprint": original["fingerprint"], "case_limit": 4,
                   "min_autonomous_save_episodes": 2}
    config = dict(base["configs"]["val_seen"])
    model = load_model_config(config["env_path"]).redacted_summary()
    if model != base["model"]:
        raise ValueError("current model differs from paired baseline configuration")
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    shutil.copytree(baseline / "source", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in OVERLAY:
        target = source / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for plugin in ("pixel_depth", "place_memory"):
        shutil.copytree(ROOT / "plugins/capabilities" / plugin, source / "plugins/capabilities" / plugin,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "configs/plugins.pixel_map.json", source / "configs/plugins.json")
    config.update(output_dir=str(output / "val_seen"), run_root=str(output / "val_seen/runs"),
                  episode_ids=[t["episode_id"] for t in tasks], max_cases=None,
                  map_config_path=str(source / "configs/maps/aerial_vln_s_remote.example.json"),
                  context_config_path=str(source / "configs/context.json"),
                  enable_action_planner_debug=False, generate_html_review=True, report_profile="paper_process")
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    manifest = {"version": 1, "experiment": "pixel_place_memory", "seed": SEED,
                "baseline": str(baseline), "baseline_fingerprint": base["fingerprint"],
                "configs": {"val_seen": config}, "tasks": tasks, "pilot": pilot,
                "technical_trial": pilot, "model": model, "source_hashes": hashes,
                "counts": {"val_seen": len(tasks)}, "policy": base["policy"],
                "changes": FEATURES,
                "environment": environment_diagnostics()}
    if preview is not None:
        manifest["preview"] = preview
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_hashes.json", hashes)
    write_json(output / "selected_cases.json", {"seed": SEED, "cases": tasks, "pilot": pilot})
    ParallelRecoveryQueue(output / "progress.json", tasks, manifest["fingerprint"])
    write_json(output / "environment.json", manifest["environment"])
    print(json.dumps({"prepared": str(output), "cases": len(tasks), "pilot": len(pilot), "scenes": dict(Counter(t["scene_id"] for t in tasks))}), flush=True)


def refreeze_unstarted(output):
    """Refresh explicit overlay before any attempt; refuse mixed flight versions."""
    from evaluation.benchmarks.core.report_io import write_json
    from evaluation.benchmarks.aerial_vln.campaign import campaign_lock
    output = Path(output).resolve()
    with campaign_lock(output):
        progress = read(output / "progress.json")
        if any(v["attempts"] or v.get("result") for v in progress["tasks"].values()):
            raise ValueError("an experiment with flight attempts cannot be refrozen")
        source = output / "source"
        for relative in OVERLAY:
            shutil.copy2(ROOT / relative, source / relative)
        for name in ("pixel_depth", "place_memory"):
            shutil.copytree(ROOT / "plugins/capabilities" / name, source / "plugins/capabilities" / name,
                dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest = read(output / "manifest.json")
        manifest["changes"] = FEATURES
        manifest["source_hashes"] = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        manifest.pop("fingerprint")
        manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        progress["fingerprint"] = manifest["fingerprint"]
        write_json(output / "manifest.json", manifest)
        write_json(output / "source_hashes.json", manifest["source_hashes"])
        write_json(output / "progress.json", progress)
        # Previous capture/extension gates belong to an older code identity.
        for relative in ("capture_validation.json", "parallel/execution_overlay.json", "parallel/trial_check.json"):
            (output / relative).unlink(missing_ok=True)


def redraw_preview(source, output):
    """Read archived snapshots only; never rebuild/annotate the original store.

    The renderer is shared with production. Photos remain links to original
    files, so this diagnostic adds no RGB copies and makes no model requests.
    """
    from types import SimpleNamespace
    from plugins.capabilities.place_memory.layout import draw_map, RENDER_VERSION
    from plugins.capabilities.place_memory.render import _write_html
    from evaluation.benchmarks.core.report_io import write_json, atomic_write_text
    source, output = Path(source).resolve(), Path(output).resolve() / "offline_preview"
    if output == source or source in output.parents:
        raise ValueError("offline preview must be outside the original experiment")
    state = read(source / "progress.json")
    output.mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    for item in state["tasks"].values():
        if item["status"] != "scored":
            continue
        run = Path(item["result"]["run_dir"]).resolve()
        paths = sorted((run / "contexts/place_memory").glob("*/map_*.json"), key=lambda p: p.stat().st_mtime_ns)
        if not paths:
            raise ValueError("source episode has no archived map snapshots")
        episode = item["task"]["episode_id"]
        for label, snapshot in zip(("start", "middle", "final"), (paths[0], paths[len(paths)//2], paths[-1])):
            graph = read(snapshot)
            layout = draw_map(graph)
            directory = output / episode / label
            directory.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256((RENDER_VERSION + snapshot.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()[:20]
            image = directory / f"map_{digest}.png"
            layout.image.save(image)
            write_json(image.with_suffix(".json"), graph)
            _write_html(SimpleNamespace(directory=directory), graph, image, layout, photo_run_dir=run, offline=True)
            record = {"episode_id": episode, "stage": label, "source_snapshot": str(snapshot),
                      "source_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                      "renderer": RENDER_VERSION, "image": str(image), "html": str(directory / "index.html")}
            records.append(record)
            relative = image.relative_to(output).as_posix()
            link = (directory / "index.html").relative_to(output).as_posix()
            cards.append(f'<article><h2>{html.escape(episode)} · {label}</h2><a href="{link}"><img src="{relative}"></a><p><a href="{link}">查看地点、原始照片和实际动作</a></p></article>')
    write_json(output / "manifest.json", {"source": str(source), "offline_only": True, "renderer": RENDER_VERSION, "snapshots": records})
    atomic_write_text(output / "index.html", '<!doctype html><meta charset="utf-8"><title>简洁地图：真实轨迹离线重绘</title><style>body{font:17px/1.6 system-ui;margin:24px;color:#17324f}img{max-width:100%;border-radius:12px}article{margin:32px 0}a{color:#1878f5}</style><h1>真实轨迹离线重绘</h1><p>只读取原实验存档；新版图片未用于旧模型决策。点击地图查看原始照片、标注及实际动作。</p>' + ''.join(cards))
    return {"html": str(output / "index.html"), "snapshots": len(records)}


def attach_place_report(page, result):
    """Append the interactive graph to the original full review, idempotently."""
    page = Path(page)
    graphs = list((Path(result["run_dir"]) / "contexts/place_memory").glob("*/index.html"))
    if len(graphs) != 1:
        raise ValueError("expected one task-local place archive")
    text = page.read_text(encoding="utf-8")
    if 'id="place-memory"' in text:
        return
    target = html.escape(Path(os.path.relpath(graphs[0], page.parent)).as_posix(), quote=True)
    section = f'<section id="place-memory"><h2>地点拓扑与存档</h2><p><a href="{target}">打开完整地点地图</a></p><iframe title="地点拓扑" src="{target}" style="width:100%;height:1150px;border:0"></iframe></section>'
    from evaluation.benchmarks.core.report_io import atomic_write_text
    atomic_write_text(page, text.replace('</body>', section + '</body>') if '</body>' in text else text + section)


def write_comparison(output, state=None, *, pilot=False):
    """Pair the same valid episodes; keep infrastructure attempts out of SR."""
    from evaluation.benchmarks.core.report_io import write_json, atomic_write_text
    output = Path(output)
    manifest = read(output / "manifest.json")
    state = state or read(output / "progress.json")
    baseline = read(Path(manifest["baseline"]) / "progress.json")["tasks"]
    eligible = {t["episode_id"] for t in manifest["pilot"] if pilot} if pilot else {t["episode_id"] for t in manifest["tasks"]}
    records, rows = [], []
    for key, item in state["tasks"].items():
        if item["task"]["episode_id"] not in eligible:
            continue
        result = item.get("result") if item["status"] == "scored" else None
        old = baseline[key]["result"]
        measurements, tool_results = [], []
        if result:
            page = Path(result["run_dir"]).parents[1] / "evaluation/cases" / result["episode_id"] / "review.html"
            if page.exists() and not result.get("report_error"):
                attach_place_report(page, result)
            transcript = Path(result["run_dir"]) / "memory/agent_transcript.jsonl"
            tool_results = [r["payload"] for r in map(json.loads, transcript.read_text(encoding="utf-8").splitlines() if transcript.exists() else []) if r["kind"] == "tool_result"]
            measurements = [r for r in tool_results if r["tool_name"] == "skill.pixel_depth.measure_pixel_depth"]
        record = {"episode_id": item["task"]["episode_id"], "status": item["status"],
                  "baseline": old.get("metrics", {}), "metrics": result.get("metrics", {}) if result else None,
                  "completed": bool(result and result.get("completed")),
                  "attempts": len(item["attempts"]), "pixel_calls": len(measurements),
                  "save_calls": sum(r["tool_name"] == "skill.place_memory.save_place" for r in tool_results),
                  "save_ok": sum(r["tool_name"] == "skill.place_memory.save_place" and r["status"] == "ok" for r in tool_results),
                  "inspect_calls": sum(r["tool_name"] == "skill.place_memory.inspect_place" for r in tool_results),
                  "termination_reason": result.get("termination_reason") if result else None,
                  "report_pending": bool(result and (result.get("report_error") or not page.exists())),
                  "pixel_valid": sum(m.get("data", {}).get("distance_status") == "ok" for m in measurements)}
        records.append(record)
        label = html.escape(record["episode_id"])
        if result and page.exists():
            link = html.escape(Path(os.path.relpath(page, output)).as_posix(), quote=True)
            label = f'<a href="{link}">{label}</a>'
        rows.append(f'<tr><td>{label}</td><td>{html.escape(item["status"])}</td><td>{old.get("metrics", {}).get("success")}</td><td>{record["metrics"].get("success") if result else "-"}</td><td>{record["metrics"].get("oracle_success") if result else "-"}</td><td>{record["save_ok"]}/{record["save_calls"]}</td><td>{record["inspect_calls"]}</td></tr>')
    scored = [r for r in records if r["metrics"] is not None]
    def metrics(items, field):
        return {k: sum(float(r[field].get(k, 0)) for r in items) / len(items) if items else None
                for k in ("success", "oracle_success", "ndtw", "sdtw")}
    completed = [r for r in scored if r["completed"]]
    report = {"requested": len(eligible), "valid_scored": len(scored), "completed": len(completed),
              "blocked_or_navigation_failure": len(scored) - len(completed),
              "counts": dict(Counter(r["status"] for r in records)),
              "metrics": metrics(scored, "metrics"), "completed_only": metrics(completed, "metrics"),
              "paired_baseline": metrics(scored, "baseline"), "cases": records}
    calls = sum(r["pixel_calls"] for r in scored)
    report["pixel_depth"] = {"calls": calls, "valid": sum(r["pixel_valid"] for r in scored),
        "valid_rate": sum(r["pixel_valid"] for r in scored) / calls if calls else None}
    report["place_tools"] = {k: sum(r[k] for r in scored) for k in ("save_calls", "save_ok", "inspect_calls")}
    oracle_only = [r for r in scored if r["metrics"].get("oracle_success") and not r["metrics"].get("success")]
    report["oracle_without_success"] = {"count": len(oracle_only),
        "left_goal_region": sum(r["metrics"].get("distance_to_goal", 0) > 20 for r in oracle_only),
        "final_inside_but_not_completed": sum(r["metrics"].get("distance_to_goal", 0) <= 20 and not r["completed"] for r in oracle_only)}
    report["infrastructure_error_attempts"] = sum(bool((a.get("result") or {}).get("infrastructure_error"))
        for item in state["tasks"].values() if item["task"]["episode_id"] in eligible for a in item["attempts"])
    name = "pilot_report" if pilot else "comparison"
    write_json(output / (name + ".json"), report)
    def percent(value):
        return f"{value:.2%}" if value is not None else "—"
    def number(value):
        return f"{value:.3f}" if value is not None else "—"
    summary = (f'<p>有效评分：{len(scored)}/{len(eligible)}；正常报告完成：{len(completed)}；导航失败或 blocked：{len(scored)-len(completed)}；基础设施错误尝试：{report["infrastructure_error_attempts"]}。</p>'
               f'<p>SR：<strong>{percent(report["metrics"]["success"])}</strong> · OSR：<strong>{percent(report["metrics"]["oracle_success"])}</strong> · nDTW：{number(report["metrics"]["ndtw"])}。</p>'
               f'<p>完成子集 SR / OSR：{percent(report["completed_only"]["success"])} / {percent(report["completed_only"]["oracle_success"])}；同批旧基线 SR / OSR：{percent(report["paired_baseline"]["success"])} / {percent(report["paired_baseline"]["oracle_success"])}。</p>'
               f'<p>像素测距：{report["pixel_depth"]["valid"]}/{calls} 有效；存档：{report["place_tools"]["save_ok"]}/{report["place_tools"]["save_calls"]} 成功；查阅：{report["place_tools"]["inspect_calls"]} 次。</p>')
    body = '<!doctype html><meta charset="utf-8"><title>简洁地图四条复测</title><style>body{font:17px/1.7 system-ui;margin:32px;color:#17324f}td,th{padding:10px;border-bottom:1px solid #ddd}a{color:#1878f5}</style><h1>地点记忆与像素测距：' + str(len(eligible)) + ' 条对照</h1><p>指标采用有效评分，包括导航失败及 blocked；基础设施错误排除。四条试验用于验证功能，不能判断单项能力贡献。</p>' + summary + '<p><a href="save_place_check.html">地图与自主存档检查</a> · <a href="offline_preview/index.html">旧轨迹离线重绘</a></p><table><tr><th>完整任务过程</th><th>状态</th><th>旧 SR</th><th>新 SR</th><th>新 OSR</th><th>存档成功/调用</th><th>查阅</th></tr>' + ''.join(rows) + '</table>'
    atomic_write_text(output / (name + ".html"), body)
    if manifest.get("preview"):
        atomic_write_text(output / "report.html", body)
    if manifest.get("preview"):
        write_place_check(output, state)
    return report


def write_place_check(output, state):
    """Audit model-originated saves; never add notes or fly to repair a report."""
    from evaluation.benchmarks.core.report_io import write_json, atomic_write_text
    output = Path(output)
    cases, cards = [], []
    for item in state["tasks"].values():
        record = {"episode_id": item["task"]["episode_id"], "status": item["status"],
                  "autonomous_saves": 0, "evidence_consistent": False}
        result = item.get("result") if item["status"] == "scored" else None
        if result:
            run = Path(result["run_dir"])
            transcript = run / "memory/agent_transcript.jsonl"
            events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()] if transcript.exists() else []
            calls = {c["tool_call_id"]: c for event in events if event["kind"] == "assistant_turn"
                     for c in event["payload"].get("tool_calls", [])}
            saves = [event["payload"] for event in events if event["kind"] == "tool_result"
                     and event["payload"]["tool_name"] == "skill.place_memory.save_place"
                     and event["payload"]["status"] == "ok"
                     and event["payload"]["tool_call_id"] in calls
                     and calls[event["payload"]["tool_call_id"]]["tool_name"] == "skill.place_memory.save_place"]
            consistent = []
            for saved in saves:
                call = calls[saved["tool_call_id"]]
                data, args = saved["data"], call["args"]
                refs = [entry["media_ref"] for entry in saved.get("model_inputs", []) if "media_ref" in entry]
                snapshot = run / refs[0]["ref"] if refs else None
                evidence_ok = False
                if snapshot and snapshot.is_file() and snapshot.with_suffix(".json").is_file():
                    graph = read(snapshot.with_suffix(".json"))
                    node = next((n for n in graph["nodes"] if n["id"] == args["place_id"]), None)
                    if node:
                        evidence_ok = node["summary"] == args["summary"] == data["summary"] and node["landmarks"] == args["landmarks"] == data["landmarks"]
                        for photo in node["photos"]:
                            path = (run / photo["ref"]).resolve()
                            evidence_ok = evidence_ok and run.resolve() in path.parents and path.is_file() and photo["observation_id"] in node["visits"]
                consistent.append(bool(evidence_ok))
            record.update(autonomous_saves=len(saves), evidence_consistent=bool(saves) and all(consistent))
            directories = list((run / "contexts/place_memory").glob("*"))
            if len(directories) == 1:
                directory = directories[0]
                maps = sorted(directory.glob("map_*.png"), key=lambda path: path.stat().st_mtime_ns)
                if maps:
                    record["map_image"] = str(maps[-1])
                    ref = html.escape(Path(os.path.relpath(maps[-1], output)).as_posix(), quote=True)
                    archive = html.escape(Path(os.path.relpath(directory / "index.html", output)).as_posix(), quote=True)
                    review = run.parents[1] / "evaluation/cases" / record["episode_id"] / "review.html"
                    review_ref = html.escape(Path(os.path.relpath(review, output)).as_posix(), quote=True)
                    cards.append(f'<article><h2>{html.escape(record["episode_id"])}</h2><p>模型自主有效存档：{len(saves)} 次；证据对应核验：{record["evidence_consistent"]}</p><img src="{ref}"><p><a href="{archive}">地点、照片与访问动作</a> · <a href="{review_ref}">完整过程 HTML</a></p></article>')
        cases.append(record)
    completed = len(cases) == 4 and all(c["status"] == "scored" for c in cases)
    verified = sum(c["autonomous_saves"] > 0 and c["evidence_consistent"] for c in cases)
    report = {"complete": completed, "passed": completed and verified >= 2,
              "required_autonomous_save_episodes": 2, "verified_save_episodes": verified,
              "status": "passed" if completed and verified >= 2 else "not_verified" if completed else "running",
              "semantic_grounding": "Requires human review of the original photos; reference consistency is not a semantic truth judge.",
              "cases": cases}
    write_json(output / "save_place_check.json", report)
    summary = f'<p>状态：{html.escape(report["status"])}；有自主存档且引用一致的任务：{verified}/{len(cases)}。</p>'
    atomic_write_text(output / "save_place_check.html", '<!doctype html><meta charset="utf-8"><title>主动地点存档：4 条实测</title><style>body{font:17px/1.6 system-ui;margin:24px;color:#172b4d}img{max-width:100%}article{margin:32px 0}</style><h1>主动地点存档：4 条实测</h1><p>只统计有效评分尝试中的模型自主存档。图片与引用一致不代表语义已被验证，需结合原始照片查看标注。未通过时不自动补写标注。</p>' + summary + ''.join(cards))
    return report


def controller_entry(output):
    """Use the separately frozen, dependency-complete execution extension.

    Calibration freezes the external controller with its recovery helpers.
    Runtime/model imports still bootstrap from output/source in the child.
    Launching source/parallel_campaign.py directly loses those helpers because
    they intentionally do not belong to the original Runtime source overlay.
    """
    output = Path(output).resolve()
    manifest = read(output / "manifest.json")
    overlay = read(output / "parallel/execution_overlay.json")
    if overlay["base_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("controller extension does not match the frozen experiment")
    script = Path(overlay["controller_script"]).resolve()
    directory = output / "parallel/controller_source"
    if directory not in script.parents:
        raise ValueError("controller must be inside the experiment extension archive")
    for name, digest in overlay["source_hashes"].items():
        path = (script.parent / name).resolve()
        if path.parent != script.parent or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("controller extension file changed")
    if script.name not in overlay["source_hashes"]:
        raise ValueError("controller entry is not included in extension hashes")
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-preview", "refreeze", "calibrate", "run", "watch", "report", "redraw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--selection-from", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--source", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    args = parser.parse_args()
    args.output = args.output or (DEFAULT_SIMPLE_OUTPUT if args.command == "redraw" else DEFAULT_PREVIEW_OUTPUT if args.command == "prepare-preview" else DEFAULT_OUTPUT)
    if args.command == "redraw":
        print(json.dumps(redraw_preview(args.source, args.output), ensure_ascii=False), flush=True)
    elif args.command in {"prepare", "prepare-preview"}:
        prepare(args.output, args.baseline, preview_from=args.selection_from if args.command == "prepare-preview" else None)
    elif args.command == "refreeze":
        refreeze_unstarted(args.output)
    elif args.command == "report":
        write_comparison(args.output)
    elif args.command == "calibrate":
        from evaluation.benchmarks.aerial_vln.capture_validation import validate_campaign
        if not validate_campaign(args.output)["passed"]:
            raise SystemExit(1)
    else:
        script = controller_entry(args.output)
        command = [sys.executable, "-u", "-X", "utf8", str(script), args.command,
                   "--output", str(args.output), "--workers", str(args.workers)]
        raise SystemExit(subprocess.call(command, cwd=args.output / "source"))


if __name__ == "__main__":
    main()
