"""Run the JSON-configured visual-grounding capability against saved frames.

This probe never starts a simulator or calls the language model. It uses the
same capability.json settings and handler service as Runtime, writes annotated
candidate evidence, and retains the complete internal depth audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.capabilities.visual_grounding.locate.scripts.service import (
    service_from_settings,
)


CAPABILITY_JSON = (
    REPO_ROOT
    / "plugins"
    / "capabilities"
    / "visual_grounding"
    / "locate"
    / "capability.json"
)


def main() -> None:
    args = _parser().parse_args()
    if args.repeat < 1:
        raise SystemExit('--repeat must be positive')
    if args.trace_dir:
        from tests.manual.process_diagnostics import install_grounding_trace
        install_grounding_trace(args.trace_dir)
    if args.sam_attention:
        # One-variable diagnostic override; never changes capability settings.
        from transformers import SamModel
        original_load = SamModel.from_pretrained
        def diagnostic_load(*values, **kwargs):
            return original_load(*values, **kwargs, attn_implementation=args.sam_attention)
        SamModel.from_pretrained = diagnostic_load
    rgb_path = Path(args.rgb).resolve()
    depth_path = Path(args.depth).resolve() if args.depth else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rgb_path.is_file():
        raise SystemExit(f"RGB frame does not exist: {rgb_path}")
    if args.include_distance and (depth_path is None or not depth_path.is_file()):
        raise SystemExit("--include-distance requires an existing --depth metric-depth .npy file")

    capability = json.loads(CAPABILITY_JSON.read_text(encoding="utf-8"))
    settings = capability.get("settings")
    if not isinstance(settings, dict):
        raise SystemExit("capability.json settings must be an object")
    settings = dict(settings)
    if args.grounding_model:
        settings["grounding_model"] = str(Path(args.grounding_model).resolve())
    if args.segmentation_model:
        settings["segmentation_model"] = str(
            Path(args.segmentation_model).resolve()
        )
    if args.device:
        settings["device"] = args.device
    if args.sam_encoder_only:
        from PIL import Image
        from plugins.capabilities.visual_grounding.locate.scripts.backends.sam import SamBackend
        backend = SamBackend(settings['segmentation_model'], settings['device'])
        processor, model = backend._load()
        with Image.open(rgb_path) as source:
            inputs = processor(images=source.convert('RGB'), return_tensors='pt')
        print('sam.encoder begin', flush=True)
        with backend._torch().no_grad():
            embeddings = model.get_image_embeddings(inputs['pixel_values'].to(backend.device))
        print(json.dumps({'stage': 'sam.encoder end', 'shape': list(embeddings.shape)}), flush=True)
        return
    frames = {
        "rgb": {"id": "probe_rgb", "path": str(rgb_path)},
    }
    if depth_path is not None:
        frames["depth"] = {
            "id": "probe_depth",
            "path": str(depth_path),
            "depth_unit": "meter",
        }
    service = service_from_settings(settings)
    for repeat in range(args.repeat):
        result = service.locate(
            observation={"observation_id": args.observation_id, "frames": frames},
            observation_id=args.observation_id,
            query=args.query,
            include_distance=args.include_distance,
            run_dir=str(output_dir),
            tool_call_id="offline_visual_grounding_probe",
        )
        print(json.dumps({'repeat': repeat + 1, 'status': 'returned'}), flush=True)
    output = {
        "status": "ok",
        "call_name": capability.get("call_name"),
        "input": {
            "observation_id": args.observation_id,
            "query": args.query,
            "include_distance": args.include_distance,
            "rgb": str(rgb_path),
            "depth": str(depth_path) if depth_path else None,
        },
        "result": result["data"],
        "artifact_refs": list(result["artifact_refs"]),
    }
    output_path = output_dir / "visual_grounding_probe.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_count": len(result["data"]["candidates"]),
                "output": str(output_path),
                "artifact_refs": list(result["artifact_refs"]),
            },
            ensure_ascii=False,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run visual grounding on one saved RGB/depth observation."
    )
    parser.add_argument("--rgb", required=True, help="Saved RGB frame path.")
    parser.add_argument("--depth", default=None, help="Matching metric-depth .npy path.")
    parser.add_argument("--query", required=True, help="Natural-language target query.")
    parser.add_argument("--observation-id", default="obs_probe_0001")
    parser.add_argument("--include-distance", action="store_true")
    parser.add_argument(
        "--grounding-model",
        default=None,
        help="Probe-only local model override; does not modify capability.json.",
    )
    parser.add_argument(
        "--segmentation-model",
        default=None,
        help="Probe-only local model override; does not modify capability.json.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trace-dir", help="Child-local stage trace; no production instrumentation")
    parser.add_argument("--repeat", type=int, default=1, help="Calls in the same process for warm-path diagnosis")
    parser.add_argument("--sam-attention", choices=("eager", "sdpa"), help="Diagnostic-only SAM attention override")
    parser.add_argument("--sam-encoder-only", action="store_true", help="Isolate SAM image encoder without GroundingDINO or depth processing")
    return parser


if __name__ == "__main__":
    main()
