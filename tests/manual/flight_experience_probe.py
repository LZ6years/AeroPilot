"""Inspect saved evidence or run independent offline reflection and review.

No flight/tool action is executed. Existing model configuration is loaded only
by this manual entry point; the experiment module accepts generic model clients.
Each invocation requires a new output directory outside the source run.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plugins.research.flight_experience import ExperimentSettings, read_run, run_experiment
from plugins.research.flight_experience.experiment import prepare_output, write_json
from src.llm import create_model_client
from src.agent.config import load_model_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction-id")
    parser.add_argument("--instruction", help="Original instruction for legacy action-only logs")
    parser.add_argument("--env", type=Path, default=ROOT / ".env")
    parser.add_argument("--reflection-model")
    parser.add_argument("--review-model")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--checkpoints',type=Path)
    parser.add_argument('--database',type=Path)
    parser.add_argument("--observations", nargs="+", help="Explicit initial observation IDs; reviewer still independently selects original counterevidence")
    for role in ("reflection", "review"):
        for direction in ("input", "output"):
            parser.add_argument(f"--{role}-{direction}-price", type=float, help="USD per million tokens; omit if unknown")
    args = parser.parse_args()
    bundle = read_run(args.source, instruction_id=args.instruction_id, instruction=args.instruction)
    if args.mode == "inspect":
        output = prepare_output(bundle, args.output)
        write_json(output / "inspection.json", {"timeline_events": len(bundle.timeline),
            "rgb_available": len(bundle.images), "observations": len(bundle.observations),
            "source": str(bundle.source), "model_calls": 0})
        print(f"Indexed {len(bundle.timeline)} events, {len(bundle.images)} original RGBs: {output}")
        return 0
    model = load_model_config(args.env)
    reflection = replace(model.to_provider_settings(), model_name=args.reflection_model or model.model_name,timeout_sec=args.timeout,provider_timeout_retries=1)
    review = replace(model.to_provider_settings(), model_name=args.review_model or model.model_name,timeout_sec=args.timeout,provider_timeout_retries=1)
    settings = ExperimentSettings(max_images=args.max_images, max_candidates=args.max_candidates,checkpoint_directory=args.checkpoints,library_path=args.database,
        reflection_model=reflection.model_name, review_model=review.model_name,
        observation_ids=tuple(args.observations) if args.observations is not None else None,
        **{f"{role}_{direction}_price": getattr(args, f"{role}_{direction}_price")
           for role in ("reflection", "review") for direction in ("input", "output")})
    parameters = model.thinking_parameters()
    result = run_experiment(bundle, args.output, reflection=create_model_client(reflection),
        reviewer=create_model_client(review), settings=settings, parameters=parameters)
    print(f"{result['status']}: {args.output.resolve() / 'report.html'}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
