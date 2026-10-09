"""Thin manual launcher for the external frozen seen parallel controller.

Use the explicit flyagent interpreter. The implementation bootstraps the
original frozen source before importing any Runtime, model or plugin module.
Keeping this launcher free of project imports prevents accidental execution
against the changing working tree.
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] /
        "evaluation/benchmarks/aerial_vln/parallel_campaign.py"), run_name="__main__")
