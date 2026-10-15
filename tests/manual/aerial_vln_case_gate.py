"""Read-only per-case gates for manual AerialVLN experiments.

Reuse the Native transcript/trace auditor and window tracking calculator. This
does not alter flight execution, official metrics, or model behavior judgments.
"""

import html
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from evaluation.benchmarks.uav_flow.metric_suite import _backend_action_tracking
from src.plugins import PluginRegistry
from src.uav.config import load_uav_action_config
from src.sim_server.remote.control import RemoteSceneControl
from src.tools.registry import default_registry
from src.uav.frames import pose_value, yaw_deg
from tests.manual.native_tool_loop_real_audit import _audit_transcript, _audit_physical_trace


def _audit_registry():
    """Build the same plugin-aware tool surface used by RuntimeFactory."""

    return default_registry(PluginRegistry.from_config())


def audit_case(case_dir: Path, result: dict, package, *, output_path: Path | None = None) -> dict:
    """Fail closed on missing evidence; behavior failure is not a gate failure."""
    report = {"passed": False, "case_id": case_dir.name}
    try:
        read = lambda name: json.loads((case_dir / name).read_text(encoding="utf-8"))
        audit, trace = read("audit.json"), read("trace.json")
        windows = []
        for raw in trace["physical_windows"]:
            window = dict(raw)
            for key in ("observation_before_pose", "observation_after_pose"):
                pose = window[key]
                if not pose or not any(k in pose for k in ("yaw_deg", "euler_deg", "euler_rad")):
                    raise ValueError("Missing observed window pose/yaw")
                window[key] = {**{axis: pose_value(pose, axis) for axis in ("x", "y", "z")},
                               "yaw_deg": yaw_deg(pose)}
            window["ordered_steps"] = [SimpleNamespace(**step) for step in window["ordered_steps"]]
            windows.append(SimpleNamespace(**window))
        tracking = _backend_action_tracking(
            SimpleNamespace(**{**trace, "physical_windows": windows}),
            action_config=load_uav_action_config(), completed=result.get("completed", False))
        combined = {**result, "trace": trace, "backend_action_tracking": tracking}
        violations = []
        registry = _audit_registry()
        coverage = _audit_transcript(audit["agent_transcript"], audit, combined,
                                    case_dir, registry, violations, case_dir.name)
        _audit_physical_trace(combined, audit["agent_transcript"], audit,
                              registry, violations, case_dir.name)
        page = (case_dir / "review.html").read_text(encoding="utf-8")
        refs = re.findall(r'<img[^>]+src="([^"]+)"', page)
        bad_images = [ref for ref in refs if not ref.startswith("review_assets/")
                      or not (case_dir / html.unescape(ref)).is_file()]
        sizes = set()
        for frame in audit["frames"]:
            modality = frame.get("modality")
            if modality not in {"rgb", "depth"}:
                continue
            path = Path(frame.get("metadata", {}).get("artifact_path") or frame.get("path") or "")
            if modality == "rgb":
                with Image.open(path) as image:
                    size = image.size
            else:
                size = tuple(np.load(path, mmap_mode="r").shape[:2][::-1])
            sizes.add((modality, *size))
        remote = RemoteSceneControl()
        cleanup = []
        for scene in (Path(result["run_dir"]) / "scenes").iterdir():
            if not (scene / "lifecycle.jsonl").is_file():
                continue
            state = remote.status(package, scene.name)
            endpoint = remote.probe_prepared_port(package, scene.name,
                str(package.airsim.get("remote_host", "127.0.0.1")), int(package.airsim["port"]))
            cleanup.append({"scene": scene.name, "state": state, "endpoint": endpoint,
                            "verified": state.get("cleanup_verified") is True
                            and state.get("status") == "stopped"
                            and endpoint.get("status") == "unreachable"})
        report.update(coverage=coverage, tracking=tracking,
                      violations=[item.to_dict() for item in violations],
                      bad_image_refs=bad_images, image_dimensions=sorted(sizes), cleanup=cleanup)
        report["passed"] = bool(
            not violations and not bad_images and refs
            and sizes == {("rgb", 256, 256), ("depth", 256, 256)}
            and trace.get("trace_integrity") == "ok"
            and tracking["tracking_valid"]
            and (not windows or tracking["all_observed_within_tolerance"] is True)
            and cleanup and all(item["verified"] for item in cleanup)
            and (not result.get("infrastructure_error")
                 or result["infrastructure_error"].get("stage") == "provider")
            and not result.get("cleanup_errors"))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (output_path or case_dir / "manual_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
