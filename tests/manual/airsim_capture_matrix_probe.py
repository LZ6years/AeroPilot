"""Diagnose AirSim image transports while EnvClient owns scene lifecycle.

This developer probe deliberately calls the AirSim SDK directly after the
public EnvClient has started and connected a scene.  It is not a production
path: its purpose is to compare request flags without adding experimental
fallbacks to ``AirSimUavAdapter``.  Process ownership and cleanup still remain
inside EnvClient.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs.map_config import load_map_config
from src.sim_server.execution import SimulationExecutionMode
from src.sim_server.server import SimulationServer


def main() -> int:
    """Run a non-moving capture matrix and always release the owned scene."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument(
        "--run-root", default="runs/manual_airsim_capture_matrix"
    )
    args = parser.parse_args()

    packages, policy = load_map_config(args.config)
    package = next(item for item in packages if item.map_id == args.map_id)
    run_dir = (
        Path(args.run_root)
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    server = SimulationServer(
        packages,
        policy,
        run_dir=run_dir,
        execution_mode=SimulationExecutionMode.MANAGED,
        save_artifacts=False,
    )
    client = SimulatorEnvClient(server)
    scene = None
    report: dict[str, Any] = {"status": "starting", "captures": {}}
    try:
        scene = client.start_scene(package.map_id, package.scene_id)
        connected = client.execute_action(
            scene.scene_instance_id, "uav_1", "connect", {}
        )
        if not connected.ok:
            raise RuntimeError(connected.message)
        report["scene"] = scene.to_dict()
        report["captures"] = _capture_matrix(
            scene.control_host,
            scene.control_port,
            str(package.airsim["vehicle_name"]),
            str(package.airsim["camera_name"]),
            run_dir,
        )
        report["status"] = "completed"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise
    finally:
        if scene is not None:
            try:
                ownership = scene.process_ownership
                client.stop_scene(scene.scene_instance_id)
                report["cleanup"] = {
                    "ok": True,
                    "process_ownership": ownership,
                }
            except Exception as cleanup_error:
                report["cleanup"] = {
                    "ok": False,
                    "error": str(cleanup_error),
                }
        (run_dir / "capture_matrix.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(run_dir), **report}, indent=2))


def _capture_matrix(
    host: str,
    port: int,
    vehicle_name: str,
    camera_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Request each transport separately so one failure cannot hide another."""

    import airsim

    cases = (
        ("scene_raw", airsim.ImageType.Scene, False, False),
        ("scene_compressed", airsim.ImageType.Scene, False, True),
        ("depth_vis_float_compressed", airsim.ImageType.DepthVis, True, True),
        ("depth_vis_uint8_compressed", airsim.ImageType.DepthVis, False, True),
        ("depth_vis_float_raw", airsim.ImageType.DepthVis, True, False),
        (
            "depth_perspective_float_raw",
            airsim.ImageType.DepthPerspective,
            True,
            False,
        ),
    )
    results: dict[str, Any] = {}
    for name, image_type, pixels_as_float, compress in cases:
        rpc = airsim.VehicleClient(ip=host, port=port, timeout_value=15)
        try:
            response = rpc.simGetImages(
                [
                    airsim.ImageRequest(
                        camera_name,
                        image_type,
                        pixels_as_float,
                        compress,
                    )
                ],
                vehicle_name=vehicle_name,
            )[0]
            results[name] = _save_response(name, response, run_dir)
        except Exception as exc:
            results[name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            close = getattr(rpc, "close", None)
            if callable(close):
                close()
    return results


def _save_response(name: str, response: Any, run_dir: Path) -> dict[str, Any]:
    """Record raw response sizes and decode whichever payload is populated."""

    width = int(response.width)
    height = int(response.height)
    uint8_payload = bytes(response.image_data_uint8 or b"")
    float_payload = np.asarray(
        response.image_data_float or [], dtype=np.float32
    )
    result: dict[str, Any] = {
        "ok": bool(uint8_payload or float_payload.size),
        "width": width,
        "height": height,
        "bytes": len(uint8_payload),
        "float_count": int(float_payload.size),
    }
    if float_payload.size:
        path = run_dir / f"{name}.npy"
        np.save(path, float_payload)
        result["float_path"] = str(path)
        result["float_min"] = float(float_payload.min())
        result["float_max"] = float(float_payload.max())
        result["float_std"] = float(float_payload.std())
    if uint8_payload:
        encoded_path = run_dir / f"{name}.bin"
        encoded_path.write_bytes(uint8_payload)
        result["uint8_path"] = str(encoded_path)
        try:
            with Image.open(BytesIO(uint8_payload)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            preview = run_dir / f"{name}.png"
            Image.fromarray(array, mode="RGB").save(preview)
            result["preview_path"] = str(preview)
            result["preview_std"] = float(array.std())
        except Exception:
            pixel_count = width * height
            channels, remainder = divmod(len(uint8_payload), pixel_count or 1)
            if width > 0 and height > 0 and not remainder and channels in {3, 4}:
                array = np.frombuffer(uint8_payload, dtype=np.uint8).reshape(
                    height, width, channels
                )
                preview = run_dir / f"{name}.png"
                Image.fromarray(
                    array, mode="RGBA" if channels == 4 else "RGB"
                ).save(preview)
                result["preview_path"] = str(preview)
                result["preview_std"] = float(array.std())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
