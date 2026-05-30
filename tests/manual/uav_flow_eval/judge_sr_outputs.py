"""Run or summarize SR judge outputs for an existing UAV-Flow eval directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    args = _parser().parse_args()
    eval_dir = Path(args.eval_dir)
    cases = sorted(eval_dir.glob("cases/*/sr_judge.json"))
    summary = {"eval_dir": str(eval_dir), "case_count": len(cases), "statuses": {}}
    for path in cases:
        data = json.loads(path.read_text(encoding="utf-8"))
        status = str(data.get("status", "unknown"))
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize UAV-Flow SR judge outputs")
    parser.add_argument("--eval-dir", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
