"""Supervise frozen AirSim-only experiments, with short offline review exports.

No language model is constructed. Each group owns a new process and scene.
Case 16 is a diagnostic reproduction, never included in speed averages.
"""

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.manual.process_diagnostics import supervise, write_json
from tests.manual.supervised_flight_probe import cleanup_owned
from tests.manual.experiment_sources import source_hashes
from src.runtime.configs.map_config import load_map_config
from src.sim_server.remote.control import RemoteSceneControl

BASELINE = ROOT / "tests/manual_outputs/benchmark_20_flash_20260923_v2"
MAPS = ROOT / "configs/maps/travel_uav_remote.local"


def experiment_cases(mode, repeats):
    """Identity/order come from the frozen historical manifest, never outcomes."""
    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    if mode == "case16":
        if not 1 <= repeats <= 3:
            raise ValueError("case16 permits at most three independent scenes")
        item = manifest["cases"][15]
        if item["scene"] != "NewYorkCity":
            raise ValueError("Historical case 16 identity changed")
        return [{**item, "reuse": False, "snapshots": True} for _ in range(repeats)], [
            {"action": "move_body", "args": {"vx": 0, "vy": 0, "vz": 2, "duration": 3}}]
    item = manifest["cases"][12]
    if item["scene"] != "ModernCityMap":
        raise ValueError("Historical case 13 identity changed")
    if mode == "timeout":
        return [{**item, "reuse": False, "snapshots": True, "yaw_timeout": .001}], [
            {"action": "turn_relative", "args": {"yaw_deg": 90}},
            {"action": "move_body", "args": {"vx": 1, "vy": 0, "vz": 0, "duration": 1}}]
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [{**item, "reuse": reuse, "snapshots": False}
            for _ in range(repeats) for reuse in (False, True)], [
        {"action": "turn_relative", "args": {"yaw_deg": 90}},
        {"action": "move_body", "args": {"vx": 1, "vy": 0, "vz": 0, "duration": 1}}]


def verify_cleanup(directory, item, descendants, map_config=MAPS):
    """Reuse ownership checks and additionally require the local tunnel port closed."""
    runs = directory / "runs"
    # Recover a launch interrupted before the child could publish its SceneInstance.
    # The isolated local launch spec and remote registry must agree before stopping.
    control = RemoteSceneControl()
    packages, _ = load_map_config(map_config)
    package = next(p for p in packages if p.scene_id == item["scene"])
    for spec_path in runs.glob("*/scenes/*/remote-scene-spec.json"):
        run = spec_path.parents[2]
        if (run / "state/scene_instance.json").exists():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        scene_id = spec_path.parent.name
        status = control.status(package, scene_id)
        if status.get("scene_instance_id") != scene_id or spec.get("scene_instance_id") != scene_id:
            raise RuntimeError("Interrupted launch ownership not verified")
        settings = json.loads((spec_path.parent / "settings.json").read_text(encoding="utf-8"))
        (run / "state").mkdir(exist_ok=True)
        write_json(run / "state/scene_instance.json", {
            "process_ownership": "remote_managed", "map_id": package.map_id,
            "scene_instance_id": scene_id, "remote_process_id": status["pid"],
            "control_port": settings["ApiServerPort"]})
    cleanup = cleanup_owned(runs, map_config, descendants, control=control)
    if not cleanup or not all(row["remote_stop"].get("cleanup_verified") and
                              row["endpoint"].get("status") == "unreachable" for row in cleanup):
        raise RuntimeError("Remote process group/endpoint cleanup not verified")
    ports = []
    for path in runs.glob("*/state/scene_instance.json"):
        scene = json.loads(path.read_text(encoding="utf-8"))
        port = scene["control_port"]
        with socket.socket() as sock:
            sock.settimeout(.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Owned tunnel port remains open: {port}")
        ports.append({"port": port, "released": True})
    return {"remote": cleanup, "local_ports": ports, "verified": True}


def export_review(directory, destination, status):
    """Export actual frames and full diagnostic audit; interrupted runs stay readable."""
    destination.mkdir(parents=True, exist_ok=True)
    path = directory / "runs/run/report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    write_json(destination / "audit.json", {"supervision": status, "report": report})
    images = []
    for index, source in enumerate(sorted(directory.glob("runs/*/artifacts/*/rgb/*.png")), 1):
        target = destination / f"rgb{index:03d}.png"
        if len(str(target.resolve())) > 240:
            raise ValueError("Review resource path exceeds 240 characters")
        shutil.copyfile(source, target)
        from PIL import Image
        with Image.open(target) as image:
            image.verify()
        images.append(f'<figure class="phase-timeline"><img width="384" src="{target.name}"><figcaption>{html.escape(source.name)}</figcaption></figure>')
    prefix = '<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:auto;font:16px sans-serif}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    title = f'<h1>{html.escape(report.get("case_id") or "Interrupted AirSim probe")}</h1><p>无最终评分 · 无模型动作诊断</p>'
    summary = {k: v for k, v in report.items() if k not in {"first_observation", "final_observation", "windows"}}
    summary["windows"] = [{k: v for k, v in w.items() if k != "observation"} for w in report.get("windows", [])]
    body = title + '<a href="audit_review.html">完整审计</a><h2>过程图片</h2>' + ''.join(images)
    body += '<h2>动作与状态</h2><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>'
    (destination / "review.html").write_text(prefix + body, encoding="utf-8")
    (destination / "audit_review.html").write_text(prefix + title + '<a href="audit.json">原始审计 JSON</a>' + ''.join(images) +
        '<pre>' + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + '</pre>' +
        '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    if any(len(str(p.resolve())) > 240 for p in destination.iterdir()):
        raise ValueError("Review path exceeds 240 characters")
    return {"images": len(images), "path": str(destination / "review.html")}


def performance_summary(rows):
    """Only matched full successful sequences support timing comparisons."""
    import numpy as np
    result = {}
    for reuse in (False, True):
        groups = [r for r in rows if r["reuse"] == reuse]
        windows = [w for r in groups for w in r.get("windows", [])]
        elapsed = [w["action_to_observation_seconds"] for w in windows if w.get("result", {}).get("ok")]
        rpc = [w["rpc_count"] for w in windows if w.get("result", {}).get("ok")]
        result[str(reuse)] = {
            "groups": len(groups), "windows": len(windows),
            "failed_windows": sum(not w.get("result", {}).get("ok") for w in windows),
            "p50_seconds": float(np.percentile(elapsed, 50)) if elapsed else None,
            "p95_seconds": float(np.percentile(elapsed, 95)) if elapsed else None,
            "rpc_counts": rpc,
            "startup_seconds": [r.get("startup_seconds") for r in groups],
            "residuals": [(w["result"].get("execution_state", {}).get("action_diagnostics") or
                           w["result"].get("data", {}).get("action_diagnostics", {})).get("tracking") for w in windows],
        }
    result["recommend_enable"] = False  # Human correctness review required.
    return result


def stage_summary(directory):
    """Group inclusive spans and subtract direct children for exclusive time."""
    totals = {}
    for path in directory.glob("runs/*/artifacts/*/performance.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            spans = record["spans"]
            for index, span in enumerate(spans):
                key = record["operation"] + "/" + span["name"]
                entry = totals.setdefault(key, {"count": 0, "inclusive_sec": 0., "exclusive_sec": 0., "errors": 0})
                children = sum(s["elapsed_sec"] for s in spans if s["parent"] == index)
                entry["count"] += 1
                entry["inclusive_sec"] += span["elapsed_sec"]
                entry["exclusive_sec"] += max(0., span["elapsed_sec"] - children)
                entry["errors"] += int("error_type" in span)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("case16", "normal", "timeout"), required=True)
    parser.add_argument("--repetitions", type=int, default=10, help="Pairs for normal; independent scenes for case16")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args()
    items, sequence = experiment_cases(args.mode, args.repetitions)
    args.root.mkdir(parents=True, exist_ok=False)
    args.review_root.mkdir(parents=True, exist_ok=False)
    args.root = args.root.resolve()
    args.review_root = args.review_root.resolve()
    # Freeze executable inputs outside the shared editing workspace. Do not
    # copy credentials, models, datasets other than the compact case index, or
    # historical outputs. This is an experiment snapshot, not a second checkout.
    snapshot = args.root / "source"
    sources = set(source_hashes())
    sources.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "plugins").rglob("*.md"))
    sources.update({"tests/__init__.py", str(MAPS.relative_to(ROOT)).replace("\\", "/"),
                    ".external_refs/TravelUAV/data/travel_uav_index.json"})
    hashes = {}
    for relative in sorted(sources):
        original = ROOT / relative
        if not original.is_file():
            continue
        data = original.read_bytes()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    frozen_maps = snapshot / MAPS.relative_to(ROOT)
    manifest = {"mode": args.mode, "cases": items, "sequence": sequence, "source_hashes": source_hashes(),
                "snapshot_hashes": hashes, "source_snapshot": str(snapshot),
                "map_config_sha256": hashlib.sha256(MAPS.read_bytes()).hexdigest(),
                "dataset_sha256": hashlib.sha256((ROOT / ".external_refs/TravelUAV/data/travel_uav_index.json").read_bytes()).hexdigest()}
    write_json(args.root / "manifest.json", manifest)
    write_json(args.root / "sequence.json", sequence)
    packages, _ = load_map_config(frozen_maps)
    try:
        preflight = {p.scene_id: RemoteSceneControl().preflight(p) for p in packages if p.scene_id in {i["scene"] for i in items}}
        preflight["disk_free_bytes"] = shutil.disk_usage(args.root).free
        write_json(args.root / "preflight.json", preflight)
    except Exception as exc:
        write_json(args.root / "preflight.json", {"error": str(exc)})
        raise
    rows, links = [], []
    for index, item in enumerate(items, 1):
        if any(hashlib.sha256((snapshot / p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
            raise RuntimeError("Frozen source snapshot changed; remaining scenes not started")
        directory = args.root / f"{index:02d}"
        package = next(p for p in packages if p.scene_id == item["scene"])
        command = [sys.executable, "-u", "-X", "faulthandler", "-c",
                   "import os,runpy,sys; os.chdir(sys.argv.pop(1)); runpy.run_path(sys.argv.pop(1), run_name='__main__')",
                   str(snapshot), str(snapshot / "tests/manual/airsim_action_regression_probe.py"),
                   "--config", str(frozen_maps), "--map-id", package.map_id, "--case-id", item["case_id"],
                   "--sequence-file", str(args.root / "sequence.json"), "--output-dir", str(directory / "runs/run")]
        if item["reuse"]:
            command.append("--reuse-verified-post-action-state")
        if item["snapshots"]:
            command.append("--diagnostic-snapshots")
        if item.get("yaw_timeout"):
            command.extend(["--yaw-timeout", str(item["yaw_timeout"])])
        status = supervise(command, directory)
        try:
            status["cleanup"] = verify_cleanup(directory, item, status["descendants"], frozen_maps)
        except Exception as exc:
            status["cleanup_error"] = str(exc)
        write_json(directory / "supervision.json", status)
        status["review"] = export_review(directory, args.review_root / "cases" / f"{index:02d}", status)
        report_path = directory / "runs/run/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        rows.append({"index": index, "reuse": item["reuse"], "supervision": status,
                     "startup_seconds": report.get("startup_seconds"), "error": report.get("error"),
                     "initialization_to_observation_seconds": report.get("initialization_to_observation_seconds"),
                     "stage_timings": stage_summary(directory),
                     "windows": [{k: v for k, v in w.items() if k != "observation"} for w in report.get("windows", [])]})
        links.append(f'<li><a href="cases/{index:02d}/review.html">{index:02d} {html.escape(item["case_id"])} reuse={item["reuse"]}</a></li>')
        summary = {"mode": args.mode, "planned_groups": len(items), "rows": rows,
                   "pending_indices": list(range(index + 1, len(items) + 1)), "performance": performance_summary(rows)}
        write_json(args.root / "summary.json", summary)
        (args.review_root / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>AirSim 第二批无模型验证</h1>' +
            '<p>无最终导航评分；案例 16 异常与性能对照分开报告。</p><ul>' + ''.join(links) + '</ul><pre>' +
            html.escape(json.dumps(summary["performance"], ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
        print(json.dumps({"index": index, "exit_code": status["exit_code"], "error": report.get("error"),
                          "cleanup": status.get("cleanup", {}).get("verified")}), flush=True)
        if status.get("cleanup_error") or not report or (report.get("error") and not
                (args.mode == "case16" and report.get("requires_scene_reset"))):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
