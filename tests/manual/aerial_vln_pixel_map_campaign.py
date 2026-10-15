"""Thin flyagent launcher for prepare/calibrate/run/watch/report commands."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks.aerial_vln.pixel_map_campaign import main
if __name__ == "__main__":
    main()
