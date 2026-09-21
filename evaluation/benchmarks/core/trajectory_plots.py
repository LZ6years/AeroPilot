"""Human-only trajectory comparisons; callers own units and frame conversion.

Neither reference paths nor rendered plots belong in agent context. Both
benchmarks use these plotting primitives without changing their metric inputs.
"""
from __future__ import annotations

import math
from pathlib import Path


def plot_trajectory(plt, points, target, path, reference_points=None, *, dimension=2, unit="m"):
    """Draw equal-scale paths with explicit starts/ends; close even on failure."""
    reference = reference_points or []
    fig = plt.figure(figsize=(8, 7))
    try:
        ax = fig.add_subplot(111, projection="3d" if dimension == 3 else None)
        for rows, color, name in ((reference, "#b97715", "reference"), (points, "#246ac4", "actual")):
            if not rows:
                continue
            coords = ([p[1] for p in rows], [p[0] for p in rows])
            if dimension == 3:
                coords += ([p[2] for p in rows],)
            ax.plot(*coords, color=color, label=name, linestyle="--" if name == "reference" else "-")
            ax.scatter(*[[c[0]] for c in coords], color=color, marker="o", label=f"{name} start")
            ax.scatter(*[[c[-1]] for c in coords], color=color, marker="X", label=f"{name} end")
        if target:
            coords = ([target[1]], [target[0]])
            if dimension == 3:
                coords += ([target[2]],)
            ax.scatter(*coords, color="red", marker="*", label="target")
        ax.set_xlabel(f"Y ({unit})")
        ax.set_ylabel(f"X ({unit})")
        if dimension == 3:
            ax.set_zlabel(f"Z up ({unit})")
            rows = list(points) + list(reference) + ([target] if target else [])
            if rows:
                centers = [(max(p[i] for p in rows) + min(p[i] for p in rows)) / 2 for i in range(3)]
                radius = max(1.0, max(max(p[i] for p in rows) - min(p[i] for p in rows) for i in range(3)) / 2)
                ax.set_xlim(centers[1] - radius, centers[1] + radius)
                ax.set_ylim(centers[0] - radius, centers[0] + radius)
                ax.set_zlim(centers[2] - radius, centers[2] + radius)
                ax.set_box_aspect((1, 1, 1))
        else:
            ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        if points or reference or target:
            ax.legend()
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)


def ned_display_points(rows):
    """Copy NED metres to positive-up display coordinates, rejecting corruption."""
    result = []
    for row in rows:
        if len(row) < 3 or not all(math.isfinite(float(v)) for v in row[:3]):
            raise ValueError("Trajectory contains missing or non-finite coordinates")
        result.append([float(row[0]), float(row[1]), -float(row[2])])
    return result


def write_ned_comparison(case_dir: Path, actual, reference):
    """Create portable assets, or an explicit unavailable reason (never zeros)."""
    if not actual:
        return {"status": "unavailable", "reason": "empty_actual_trajectory"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        points, refs = ned_display_points(actual), ned_display_points(reference)
        output = case_dir / "review_assets" / "trajectory"
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for dimension in (2, 3):
            path = output / f"trajectory_{dimension}d.png"
            plot_trajectory(plt, points, None, path, refs, dimension=dimension)
            paths[f"path_{dimension}d"] = str(path)
        return {"status": "ok", "has_reference": bool(refs), **paths}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
