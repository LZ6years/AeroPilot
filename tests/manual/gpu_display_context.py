"""Classify NVML graphics-only initialization separately from execution devices.

Only explicitly allowed display devices with known, bounded per-process memory
may be exempted. Original observations remain available for operator audit.
"""
from copy import deepcopy


def classify_gpu_sample(sample, display_gpus=(1,), max_display_mib=64):
    result = deepcopy(sample)
    for scene in result.get("owned", []):
        scene["all_observed_gpus"] = list(scene["observed_gpus"])
        display = []
        for gpu in result.get("gpus", []):
            processes = [p for p in gpu["processes"] if p["pgid"] == scene["pid"]]
            if gpu["index"] not in display_gpus or not processes or gpu["index"] == scene["expected_gpu"]:
                continue
            try:
                memory = sum(float(p["used_memory"].split()[0]) for p in processes)
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
            if all(p["type"] == "G" for p in processes) and 0 <= memory <= max_display_mib:
                display.append(gpu["index"])
        scene["display_only_gpus"] = display
        scene["observed_gpus"] = [i for i in scene["all_observed_gpus"] if i not in display]
    return result
