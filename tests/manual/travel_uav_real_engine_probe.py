"""Run one real TravelUAV case through the production FlyAgent runner.

This is not a fixed-action smoke test.  It selects one compact-index case,
spawns its target, runs the configured TaskParser/ModelActionPlanner, and emits
the same case/evaluation HTML as a normal benchmark run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.travel_uav.config import TravelUavEvalConfig
from evaluation.benchmarks.travel_uav.runner import TravelUavBenchmarkRunner


def _write_action_diagnostics(result: dict[str, Any]) -> dict[str, str]:
    """Extract adapter-owned physical-action diagnostics beside each case.

    The complete records remain in ``audit.json``.  This small projection makes
    RPC, collision, controller-state, and tracking failures easy to inspect
    during real-engine diagnosis without changing Benchmark Core reports.
    """

    paths: dict[str, str] = {}
    for case_result in result.get("results", []):
        case_id = str(case_result.get("episode_id", "unknown"))
        audit_path = Path(str(case_result.get("audit_path", "")))
        if not audit_path.is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        actions = []
        for action in audit.get("actions", []):
            action_result = action.get("result") or {}
            data = action_result.get("data") or {}
            diagnostics = action_result.get("execution_state", {}).get("action_diagnostics") or data.get("action_diagnostics")
            if not isinstance(diagnostics, dict):
                continue
            actions.append(
                {
                    "action_id": action.get("id"),
                    "timestamp": action.get("timestamp"),
                    "tool": action.get("tool"),
                    "args": action.get("args"),
                    "ok": action_result.get("ok"),
                    "message": action_result.get("message"),
                    "diagnostics": diagnostics,
                }
            )
        output_path = Path(str(case_result["case_dir"])) / "action_diagnostics.json"
        output_path.write_text(
            json.dumps({"case_id": case_id, "actions": actions}, indent=2),
            encoding="utf-8",
        )
        paths[case_id] = str(output_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--eval-id", default="travel_uav_real_engine_probe")
    args = parser.parse_args()
    data = TravelUavEvalConfig.from_file(args.config).to_dict()
    data.update(
        {
            "execution_mode": "managed",
            "case_ids": [args.case_id],
            "save_artifacts": True,
            "generate_html_review": True,
            "enable_action_planner_debug": True,
        }
    )
    result = TravelUavBenchmarkRunner(TravelUavEvalConfig.from_dict(data)).run(
        eval_id=args.eval_id
    )
    result["action_diagnostics_paths"] = _write_action_diagnostics(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["summary"]["batch_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
