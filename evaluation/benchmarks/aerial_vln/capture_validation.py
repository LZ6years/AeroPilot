"""Evaluation-only capture calibration; EnvClient owns every scene and cleanup."""
from io import BytesIO
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def capture_checks(rpc, camera, vehicle, directory):
    """Compare raw/PNG orientation and paired perspective/planar depth."""
    import airsim
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    paused = rpc.simIsPause()
    try:
        rpc.simPause(True)
        responses = rpc.simGetImages([
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True),
            airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, True, False),
            airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False)], vehicle_name=vehicle)
        raw, png, ray, axis = responses
        h, w = raw.height, raw.width
        from src.adapters.airsim.capture import AirSimCapture
        from src.perception.frame_payload import load_rgb_frame
        capture = AirSimCapture(camera, vehicle)
        decoded = capture._frame_metadata("rgb", "scene", capture._capture_spec("scene"), raw)
        color = np.asarray(load_rgb_frame(decoded))
        reference = np.asarray(Image.open(BytesIO(bytes(png.image_data_uint8))).convert("RGB"))
        if reference.shape != color.shape or any((r.height, r.width) != (h, w) for r in responses):
            raise ValueError(f"capture channel dimensions do not match: raw={color.shape}, png={reference.shape}, channels={[(r.height, r.width) for r in responses]}")
        errors = {"identity": float(np.abs(color.astype(float) - reference).mean()),
                  "swap_rb": float(np.abs(color[:, :, ::-1].astype(float) - reference).mean()),
                  "flip_y": float(np.abs(color[::-1].astype(float) - reference).mean()),
                  "flip_y_swap_rb": float(np.abs(color[::-1, :, ::-1].astype(float) - reference).mean())}
        best = min(errors, key=errors.get)
        Image.fromarray(color).save(directory / "raw_decoder.png")
        Image.fromarray(reference).save(directory / "compressed_reference.png")
        ranges, depths = (np.asarray(r.image_data_float).reshape(h, w) for r in (ray, axis))
        fov = float(rpc.simGetCameraInfo(camera, vehicle_name=vehicle).fov)
        fx = w / (2 * math.tan(math.radians(fov) / 2))
        y, x = np.mgrid[:h, :w]
        factor = np.sqrt(1 + ((x - (w - 1) / 2) / fx) ** 2 + ((y - (h - 1) / 2) / fx) ** 2)
        valid = np.isfinite(ranges) & np.isfinite(depths) & (depths > 0) & (ranges < 10000)
        relative = np.abs(ranges - depths * factor) / np.maximum(ranges, 1e-6)
        median_error = float(np.median(relative[valid])) if valid.any() else None
        positions = [np.array([r.camera_position.x_val, r.camera_position.y_val, r.camera_position.z_val]) for r in responses]
        same_pose = all(np.linalg.norm(p - positions[0]) < 1e-4 for p in positions)
        camera_names = [getattr(r, "camera_name", camera) for r in responses]
        passed = best == "identity" and errors[best] < 2 and same_pose and len(set(camera_names)) == 1 and median_error is not None and median_error < .03
        return {"passed": passed, "scene_size": [w, h], "camera": camera, "camera_names": camera_names,
                "same_camera_position": same_pose, "timestamps": [r.time_stamp for r in responses],
                "raw_png_errors": errors, "best_raw_transform": best, "fov_deg": fov,
                "metric_valid_fraction": float(valid.mean()), "perspective_planar_median_relative_error": median_error}
    finally:
        rpc.simPause(paused)


def validate_campaign(output):
    """Validate both configured lanes without consuming any flight attempts."""
    from evaluation.benchmarks.aerial_vln import parallel_campaign as parallel
    from evaluation.benchmarks.aerial_vln.campaign import environment_diagnostics
    from evaluation.benchmarks.core.supervision import cleanup_owned, cleanup_verified
    from src.runtime.configs.map_config import load_map_config
    from src.sim_server.server import SimulationServer
    from src.env_client.simulator_client import SimulatorEnvClient
    from src.adapters.airsim.client import close_airsim_client
    from src.sim_server.airsim_settings import AirSimSettingsRenderer
    from dataclasses import replace
    import hashlib
    import airsim
    class CalibrationRenderer(AirSimSettingsRenderer):
        """Add only a diagnostic DepthPlanar channel to the owned settings copy."""
        def render(self, package, destination, *args, **kwargs):
            rendered = super().render(package, destination, *args, **kwargs)
            settings = json.loads(rendered.path.read_text(encoding="utf-8"))
            camera = settings["Vehicles"][package.airsim["vehicle_name"]]["Cameras"][package.airsim["camera_name"]]
            captures = camera["CaptureSettings"]
            perspective = next(c for c in captures if c["ImageType"] == 2)
            if not any(c["ImageType"] == 1 for c in captures):
                captures.append({**perspective, "ImageType": 1})
            payload = json.dumps(settings, ensure_ascii=False).encode("utf-8")
            rendered.path.write_bytes(payload)
            return replace(rendered, sha256=hashlib.sha256(payload).hexdigest())
    output = Path(output).resolve()
    manifest = parallel.read_json(output / "manifest.json")
    parallel.verify_frozen(output, manifest)
    with parallel.campaign_lock(output):
        overlay = parallel.prepare_extension(output, manifest)
        results = []
        for lane in ("A", "B"):
            packages, policy = load_map_config(overlay["lanes"][lane]["map_config_path"])
            map_id = manifest["configs"]["val_seen"]["scene_map"][manifest["pilot"][0]["scene_id"]]
            package = next(p for p in packages if p.map_id == map_id)
            directory = output / "calibration" / lane
            server = SimulationServer(packages, policy, run_dir=directory / "run", execution_mode="managed", show_display=False,
                                      settings_renderer=CalibrationRenderer())
            client = SimulatorEnvClient(server)
            scene = rpc = None
            result = {"lane": lane, "remote_gpu": overlay["lanes"][lane]["remote_gpu"]}
            try:
                scene = client.start_scene(map_id)
                parallel.write_json(directory / "run/state/scene_instance.json", scene.to_dict())
                connected = client.execute_action(scene.scene_instance_id, package.airsim["vehicle_name"], "connect", {})
                if not connected.ok:
                    raise RuntimeError(connected.message)
                rpc = airsim.VehicleClient(ip=scene.control_host, port=scene.control_port, timeout_value=30)
                result.update(capture_checks(rpc, package.airsim["camera_name"], package.airsim["vehicle_name"], directory))
            except Exception as exc:
                result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
            finally:
                try:
                    close_airsim_client(rpc)
                except Exception as exc:
                    result["transport_close_error"] = str(exc)
                if scene:
                    try:
                        client.stop_scene(scene.scene_instance_id)
                    except Exception as exc:
                        result["cleanup_error"] = str(exc)
                receipts = cleanup_owned(directory, overlay["lanes"][lane]["map_config_path"], [])
                result["cleanup_verified"] = cleanup_verified(receipts)
                result["passed"] = result.get("passed", False) and result["cleanup_verified"]
                parallel.write_json(directory / "capture_check.json", result)
            results.append(result)
        report = {"passed": all(r["passed"] for r in results), "lanes": results, "environment": environment_diagnostics()}
        parallel.write_json(output / "capture_validation.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return report
