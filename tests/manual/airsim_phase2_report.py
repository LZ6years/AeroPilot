"""Render the completed AirSim diagnostics offline; never fabricate missing runs."""

import argparse
import html
import json
from pathlib import Path
import sys
from html.parser import HTMLParser

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def diagnostics(result):
    return result.get("execution_state", {}).get("action_diagnostics") or result.get("data", {}).get("action_diagnostics", {})


def stats(values):
    return {"count": len(values), "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def build(root):
    summaries = {}
    for mode in ("case16", "normal_frozen", "timeout"):
        path = root / mode / "summary.json"
        summaries[mode] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": [], "pending_indices": ["not started"]}
    normal = summaries["normal_frozen"]
    performance = []
    for reuse in (False, True):
        groups = [r for r in normal["rows"] if r["reuse"] == reuse]
        for action in ("turn_relative", "move_body"):
            windows = [w for r in groups for w in r["windows"] if w["action"] == action]
            success = [w for w in windows if w["result"]["ok"]]
            tracking = [diagnostics(w["result"]).get("tracking", {}) for w in success]
            residuals = [abs(t["target_yaw_error_deg"]) if action == "turn_relative" else
                         abs(t["expected_translation_m"] - t["observed_translation_m"]) for t in tracking]
            performance.append({"reuse": reuse, "action": action, "completed_windows": len(windows),
                                "failed_windows": len(windows) - len(success),
                                "rpc_count": stats([w["rpc_count"] for w in success]),
                                "action_to_observation_seconds": stats([w["action_to_observation_seconds"] for w in success]),
                                "absolute_residual": stats(residuals),
                                "residual_unit": "degree" if action == "turn_relative" else "meter",
                                "reused_windows": sum(bool(w.get("stabilization", {}).get("reused_post_action_state")) for w in windows)})
    case16 = []
    for path in sorted((root / "case16").glob("*/runs/run/report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        physical = [a for a in data["actions"] if a["action"] == "move_body"]
        case16.append({"path": str(path), "error": data.get("error"), "cleanup": data.get("cleanup"),
                       "initial_position_error_m": data.get("initial_position_error_m"),
                       "actions": [{"ok": a["ok"], "diagnostics": diagnostics(a)} for a in physical]})
    return {"diagnostic_only": True, "final_score_available": False, "performance": performance,
            "completed_groups": len(normal["rows"]), "pending_groups": normal.get("pending_indices", []),
            "startup_seconds": stats([r["startup_seconds"] for r in normal["rows"] if r.get("startup_seconds") is not None]),
            "first_startup_seconds": normal["rows"][0].get("startup_seconds") if normal["rows"] else None,
            "case16": case16, "timeout": summaries["timeout"],
            "cleanup_verified": all(r["supervision"].get("cleanup", {}).get("verified") for s in summaries.values() for r in s["rows"]),
            "recommend_default_enable": False,
            "limitations": ["No navigation task or model requests were run; this does not measure navigation success.",
                            "Case16 snapshot reads add RPCs and can affect timing; no initialization/controller fix is claimed.",
                            "The interrupted normal/01 pilot is retained but excluded from the frozen comparison.",
                            "All paired normal groups use the same immutable source snapshot. Later workspace edits are separate.",
                            "Short translation residuals include acceleration/braking; a successful action is not exact path tracking."]}


def render(report):
    rows = []
    for row in report["performance"]:
        elapsed = row["action_to_observation_seconds"]
        def number(value):
            return "缺失" if value is None else f"{value:.3f}"
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            "开启" if row["reuse"] else "关闭", row["action"], row["completed_windows"], row["failed_windows"],
            number(row["rpc_count"]["p50"]), number(elapsed["p50"]), number(elapsed["p95"]),
            number(row["absolute_residual"]["p50"]) + " " + row["residual_unit"])) + '</tr>')
    links = ''.join(f'<li><a href="{name}/index.html">{label}</a></li>' for name, label in (
        ("case16", "案例 16：3 次独立场景"), ("normal_frozen", "案例 13：关闭／开启各 10 组"),
        ("timeout", "有限转向超时及下一步边界")))
    return '<!doctype html><meta charset="utf-8"><style>body{max-width:1150px;margin:30px auto;font:16px sans-serif}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>' + \
        '<h1>AirSim 第二批：故障恢复与 RPC 对照</h1><p>无模型诊断，无最终导航评分。复用开关仍默认关闭。</p><ul>' + links + \
        '</ul><table><tr><th>复用</th><th>动作</th><th>样本</th><th>失败</th><th>RPC P50</th><th>耗时 P50/s</th><th>耗时 P95/s</th><th>残差 P50</th></tr>' + \
        ''.join(rows) + '</table><p>每个动作保留 RGB／深度采集。启动成本单列；父子计时不重复相加。</p><a href="report.json">完整对照数据</a><pre>' + \
        html.escape(json.dumps({k: v for k, v in report.items() if k not in {"timeout", "case16"}}, ensure_ascii=False, indent=2)) + '</pre>'


def validate_review(root):
    """Check the exported local links, decoded images and Windows path ceiling."""
    from PIL import Image
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            self.links.extend(value for name, value in attrs if name in {"src", "href"} and value)
    failures, count, images = [], 0, 0
    pages = list(root.rglob("*.html"))
    for page in pages:
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in parser.links:
            if target.startswith(("http:", "https:", "#", "data:")):
                continue
            count += 1
            path = page.parent / target.split("#")[0]
            if not path.is_file():
                failures.append(str(path))
            elif path.suffix.lower() == ".png":
                images += 1
                with Image.open(path) as image:
                    image.verify()
    longest = max((len(str(p.resolve())) for p in root.rglob("*") if p.is_file()), default=0)
    if longest > 240:
        failures.append("path exceeds 240 characters")
    result = {"html_pages": len(pages), "local_links": count, "image_refs": images,
              "max_path": longest, "failures": failures}
    (root / "asset_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"Review validation failed: {failures}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.root)
    args.review_root.mkdir(parents=True, exist_ok=True)
    (args.review_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.review_root / "index.html").write_text(render(report), encoding="utf-8")
    validate_review(args.review_root)
    print(args.review_root / "index.html")
