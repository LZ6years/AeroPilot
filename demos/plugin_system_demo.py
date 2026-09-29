"""Inspect enabled tools and explicitly load knowledge without a model or UE.

Run: python demos/plugin_system_demo.py --knowledge <catalog-name>
Omit --knowledge to inspect the catalog. Results go into a new timestamped
directory; bodies are explicit load results, never automatically injected state.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plugins import PluginRegistry
from src.plugins.skills.knowledge_service import KnowledgeService
from src.tools.registry import default_registry


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge", help="Exact enabled knowledge catalog name")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output-root", default="runs/experiments/plugin_system_demo")
    return parser


def inspect_plugins(name=None, reference_paths=None):
    plugins = PluginRegistry.from_config()
    service = KnowledgeService(plugins.skill_catalog)
    before = service.loaded_index()
    result = service.load(name, reference_paths or []) if name is not None else None
    if name is None and reference_paths:
        raise ValueError("--reference requires --knowledge")
    return {
        "mode": "offline_explicit_knowledge_load",
        "catalog": service.catalog(),
        "tools": default_registry(plugins).names(),
        "loaded_before": before, "load_result": result,
        "loaded_after": service.loaded_index(),
    }


def main():
    args = build_parser().parse_args()
    result = inspect_plugins(args.knowledge, args.reference)
    output = Path(args.output_root).resolve() / (
        "plugin_knowledge_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    path = output / "knowledge_demo.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(path), "knowledge_count": len(result["catalog"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
