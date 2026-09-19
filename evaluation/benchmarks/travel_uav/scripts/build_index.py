"""Build the compact TravelUAV episode index from official raw metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.travel_uav.dataset import build_travel_uav_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--seen-split", required=True)
    parser.add_argument("--unseen-split", required=True)
    parser.add_argument("--spawn-areas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-version", default="official")
    args = parser.parse_args()
    result = build_travel_uav_index(
        dataset_root=args.dataset_root,
        seen_split_path=args.seen_split,
        unseen_split_path=args.unseen_split,
        map_spawn_area_path=args.spawn_areas,
        output_path=args.output,
        dataset_version=args.dataset_version,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "episode_count": len(result["episodes"]),
                "sources": result["sources"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
