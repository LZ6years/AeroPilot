"""EnvClient-compatible UAV-Flow benchmark scenario adapter.

The adapter is deliberately outside ``src`` because it is a benchmark harness,
not a core runtime backend. It can run in dry mode for tests, and imports
``gym_unrealcv`` only when real UnrealCV execution is requested.
"""

from __future__ import annotations

import math
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.trajectory import world_pose_to_model_pose
from src.env_client import ActionResult, Observation, SceneInstance
from src.runtime.configs.action_config import UavActionConfig
from src.uav import UNREALCV_VERTICAL_SIGN, body_velocity_to_world_delta, move_body_velocity_from_args


class UavFlowBenchmarkEnvClient:
    """Drive one UAV-Flow case through the UAV-Agent EnvClient protocol."""

    def __init__(
        self,
        case: UavFlowCase,
        *,
        run_dir: str | Path,
        env_id: str,
        uav_flow_repo_path: str | Path | None = None,
        real_unrealcv: bool = False,
        time_dilation: int = 10,
        resolution: list[int] | tuple[int, int] = (256, 256),
        seed: int = 0,
        action_config: UavActionConfig | None = None,
        save_artifacts: bool = True,
        enable_depth_capture: bool = False,
        enable_research_frame_capture: bool = False,
    ) -> None:
        self.case = case
        self.run_dir = Path(run_dir)
        self.env_id = env_id
        self.uav_flow_repo_path = Path(uav_flow_repo_path) if uav_flow_repo_path else None
        self.real_unrealcv = real_unrealcv
        self.time_dilation = int(time_dilation)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.seed = int(seed)
        self.action_config = action_config or UavActionConfig()
        self.save_artifacts = save_artifacts
        self.enable_depth_capture = bool(enable_depth_capture)
        self.enable_research_frame_capture = bool(enable_research_frame_capture)
        self.env: Any | None = None
        self.scene: SceneInstance | None = None
        self.world_pose_cm = list(case.initial_pos)
        self.pose_history: list[dict[str, Any]] = []
        self._frame_index = 0
        self._case_objects_initialized = False

    def start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        instance_id = f"uav_flow_{uuid.uuid4().hex[:8]}"
        self.scene = SceneInstance(
            scene_instance_id=instance_id,
            map_id=map_id,
            scene_id=scene_id or self.case.case_id,
            machine_id="local",
            control_host="127.0.0.1",
            control_port=0,
            settings_path="",
            process_id=None,
            status="running",
        )
        if self.real_unrealcv:
            self._start_real_env()
        self._record_pose()
        return self.scene

    def get_or_start_scene(self, map_id: str, scene_id: str | None = None) -> SceneInstance:
        if self.scene is not None and self.scene.status == "running":
            return self.scene
        return self.start_scene(map_id, scene_id)

    def stop_scene(self, scene_instance_id: str) -> None:
        self.close_real_env()
        if self.scene is not None:
            self.scene.status = "stopped"

    def close_real_env(self) -> None:
        """Close the underlying simulator process when this client owns one."""

        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
            self.env = None
            self._case_objects_initialized = False
        if self.scene is not None:
            self.scene.status = "stopped"

    def prepare_case(self, case: UavFlowCase, run_dir: str | Path, scene_id: str | None = None) -> SceneInstance | None:
        """Switch a running benchmark env to a new case without restarting UE."""

        self.case = case
        self.run_dir = Path(run_dir)
        self.world_pose_cm = list(case.initial_pos)
        self.pose_history.clear()
        self._frame_index = 0
        if self.scene is not None:
            self.scene.scene_id = scene_id or case.case_id
            self.scene.status = "running"
        if self.real_unrealcv and self.env is not None:
            self._place_case_objects()
            self._apply_pose_to_real_env()
        if self.scene is not None or self.env is not None:
            self._record_pose()
        return self.scene

    def reset_scene(self, scene_instance_id: str) -> SceneInstance:
        self.world_pose_cm = list(self.case.initial_pos)
        self.pose_history.clear()
        self._record_pose()
        if self.scene is None:
            raise RuntimeError("UAV-Flow scene has not been started")
        self.scene.status = "running"
        return self.scene

    def health_check(self) -> bool:
        return True

    def get_observation(self, scene_instance_id: str, agent_id: str, modalities: list[str]) -> Observation:
        frames: dict[str, Any] = {}
        wants_rgb = "rgb" in modalities
        wants_depth = "depth" in modalities
        can_capture_depth = self.enable_depth_capture or self.enable_research_frame_capture
        if wants_rgb and wants_depth and self.real_unrealcv and self.env is not None and can_capture_depth:
            frames.update(self._rgb_depth_frames())
        else:
            if wants_rgb:
                frames["rgb"] = self._rgb_frame()
            if wants_depth:
                frames["depth"] = self._depth_frame()
        if "mask" in modalities:
            frames["mask"] = self._mask_frame()
        data: dict[str, Any] = {}
        if "pose" in modalities:
            data["pose"] = self._model_pose()
        if frames:
            data["frames"] = frames
        return Observation(scene_instance_id, agent_id, modalities, data)

    def execute_action(self, scene_instance_id: str, agent_id: str, action: str, args: dict) -> ActionResult:
        if action == "connect":
            return ActionResult(True, action, "UAV-Flow benchmark scene connected")
        if action in {"takeoff", "hover", "land"}:
            return ActionResult(True, action, f"UAV-Flow benchmark {action} treated as no-op", {"pose": self._model_pose()})
        if action == "turn_relative":
            yaw_delta = float(args.get("yaw_deg", self.action_config.yaw_step_deg))
            self.world_pose_cm[4] = _normalize_yaw_world(self.world_pose_cm[4] + yaw_delta)
            self._apply_pose_to_real_env()
            pose = self._record_pose()
            return ActionResult(True, action, "UAV-Flow benchmark yaw updated", {"pose": pose})
        if action == "move_body":
            vx, vy, vz, duration = move_body_velocity_from_args(
                args,
                self.action_config,
                vertical_sign=UNREALCV_VERTICAL_SIGN,
            )
            delta_m = body_velocity_to_world_delta(vx, vy, vz, duration, self.world_pose_cm[4])
            self.world_pose_cm[0] += delta_m["dx"] * 100.0
            self.world_pose_cm[1] += delta_m["dy"] * 100.0
            self.world_pose_cm[2] += delta_m["dz"] * 100.0
            self._apply_pose_to_real_env()
            pose = self._record_pose()
            return ActionResult(True, action, "UAV-Flow benchmark pose updated", {"pose": pose})
        raise ValueError(f"Unsupported UAV-Flow benchmark action: {action}")

    def _start_real_env(self) -> None:
        self._prepare_uav_flow_import_path()
        try:
            import gym
            import gym_unrealcv  # noqa: F401
            from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "UAV-Flow real UnrealCV mode requires gym, gym_unrealcv, and the UAV-Flow-Eval repo "
                "on PYTHONPATH. Check uav_flow_repo_path and install UAV-Flow-Eval in editable mode."
            ) from exc

        try:
            env = gym.make(self.env_id)
        except Exception as exc:
            raise RuntimeError(f"Could not create UAV-Flow gym env {self.env_id!r}: {exc}") from exc
        try:
            if self.time_dilation > 0:
                env = time_dilation.TimeDilationWrapper(env, self.time_dilation)
            env.unwrapped.agents_category = ["drone"]
            env = configUE.ConfigUEWrapper(env, resolution=self.resolution)
            env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
            env.seed(self.seed)
            env.reset()
            env.unwrapped.unrealcv.set_viewport(env.unwrapped.player_list[0])
            env.unwrapped.unrealcv.set_phy(env.unwrapped.player_list[0], 0)
            self.env = env
            self._init_case_objects_in_real_env()
            self._place_case_objects()
            self._apply_pose_to_real_env()
        except Exception as exc:
            raise RuntimeError(f"UAV-Flow UnrealCV scene initialization failed: {exc}") from exc

    def _prepare_uav_flow_import_path(self) -> None:
        if self.uav_flow_repo_path is None:
            return
        repo = self.uav_flow_repo_path
        if not repo.exists():
            raise RuntimeError(f"UAV-Flow repo path does not exist: {repo}")
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

    def _init_case_objects_in_real_env(self) -> None:
        """Create UAV-Flow's benchmark objects before per-case placement.

        The upstream runner spawns these objects once after ``env.reset`` and
        then repositions or hides them for each case. Real UnrealCV runs need
        the same setup; dry mode intentionally skips it.
        """

        if self.env is None or not self.real_unrealcv or self._case_objects_initialized:
            return
        unrealcv = self.env.unwrapped.unrealcv
        try:
            unrealcv.new_obj("bp_character_C", "BP_Character_21", [0, 0, 0])
            unrealcv.set_appearance("BP_Character_21", 0)
            unrealcv.set_obj_rotation("BP_Character_21", [0, 0, 0])
            unrealcv.new_obj("BP_BaseCar_C", "BP_Character_22", [1000, 0, 0])
            unrealcv.set_appearance("BP_Character_22", 2)
            unrealcv.set_obj_rotation("BP_Character_22", [0, 0, 0])
            unrealcv.set_phy("BP_Character_22", 0)
        except Exception as exc:
            raise RuntimeError(f"UAV-Flow case object initialization failed: {exc}") from exc
        self._case_objects_initialized = True

    def _place_case_objects(self) -> None:
        if self.env is None:
            return
        unrealcv = self.env.unwrapped.unrealcv
        use_obj = self.case.use_obj
        obj_id = self.case.obj_id
        if not self.case.target_pos or use_obj not in {1, 2}:
            self._hide_unused_case_object("BP_Character_21")
            self._hide_unused_case_object("BP_Character_22")
            return
        obj_pos = self.case.target_pos[:3]
        obj_rot = self.case.target_pos[3:] if len(self.case.target_pos) >= 6 else [0.0, 0.0, 0.0]
        if use_obj == 1:
            try:
                if obj_id is not None:
                    unrealcv.set_appearance("BP_Character_21", obj_id)
                unrealcv.set_obj_location("BP_Character_21", obj_pos)
                unrealcv.set_obj_rotation("BP_Character_21", obj_rot)
                self._hide_unused_case_object("BP_Character_22")
                # Match upstream's second placement call after hiding the other object.
                unrealcv.set_obj_location("BP_Character_21", obj_pos)
            except Exception as exc:
                raise RuntimeError(f"UAV-Flow object placement failed for use_obj=1: {exc}") from exc
        elif use_obj == 2:
            try:
                unrealcv.set_appearance("BP_Character_22", 2)
                unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
                unrealcv.set_obj_rotation("BP_Character_22", obj_rot)
                unrealcv.set_phy("BP_Character_22", 0)
                self._hide_unused_case_object("BP_Character_21")
                # Match upstream's second placement call after hiding the other object.
                unrealcv.set_obj_location("BP_Character_22", [obj_pos[0], obj_pos[1], 0])
            except Exception as exc:
                raise RuntimeError(f"UAV-Flow object placement failed for use_obj=2: {exc}") from exc

    def _hide_unused_case_object(self, name: str) -> None:
        if self.env is None:
            return
        self.env.unwrapped.unrealcv.set_obj_location(name, [0, 0, -1000])

    def _apply_pose_to_real_env(self) -> None:
        if self.env is None:
            return
        name = self.env.unwrapped.player_list[0]
        unrealcv = self.env.unwrapped.unrealcv
        unrealcv.set_obj_location(name, self.world_pose_cm[:3])
        # UAV-Flow original runner subtracts 180 when setting drone rotation.
        unrealcv.set_rotation(name, self.world_pose_cm[4] - 180.0)
        self._sync_camera_to_player()

    def _sync_camera_to_player(self) -> None:
        """Mirror UAV-Flow's ``set_cam(env)`` before image capture."""

        if self.env is None:
            return
        unrealcv = self.env.unwrapped.unrealcv
        name = self.env.unwrapped.player_list[0]
        x, y, z = unrealcv.get_obj_location(name)
        roll, yaw, pitch = unrealcv.get_obj_rotation(name)
        unrealcv.set_cam(0, [x, y, z], [roll, pitch, yaw])

    def _rgb_frame(self) -> dict[str, Any]:
        self._frame_index += 1
        path = self.run_dir / "artifacts" / "rgb" / f"frame_{self._frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.real_unrealcv and self.env is not None:
            self._sync_camera_to_player()
            image = self.env.unwrapped.unrealcv.get_image(0, "lit")
            Image.fromarray(image).save(path)
        else:
            self._write_dry_rgb(path)
        return {
            "path": str(path),
            "width": 224,
            "height": 224,
            "mock": False,
            "source": "uav_flow_benchmark",
            "case_id": self.case.case_id,
        }

    def _rgb_depth_frames(self) -> dict[str, dict[str, Any]]:
        """Capture aligned RGB and depth from camera 0 in one UnrealCV batch.

        Animated objects can move between independent image requests. A
        pixel-wise RGB-depth overlay is therefore valid only when both arrays
        come from the same batch and camera pose.
        """

        if self.env is None or not self.real_unrealcv:
            raise RuntimeError("Synchronized RGB-depth capture requires a real UnrealCV environment")
        self._sync_camera_to_player()
        self._frame_index += 1
        rgb_path = self.run_dir / "artifacts" / "rgb" / f"frame_{self._frame_index:06d}.png"
        depth_path = self._next_research_frame_path("depth", "npy")
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        unrealcv = self.env.unwrapped.unrealcv
        commands = [
            "vget /camera/0/lit png",
            "vget /camera/0/depth npy",
        ]
        decoders = [
            unrealcv.decoder.decode_map[unrealcv.decoder.cmd2key(command)]
            for command in commands
        ]
        rgb, depth = unrealcv.batch_cmd(commands, decoders)
        Image.fromarray(rgb).save(rgb_path)
        self._write_depth_array_meters(depth_path, depth)
        width, height = self.resolution
        return {
            "rgb": {
                "path": str(rgb_path),
                "width": width,
                "height": height,
                "mock": False,
                "source": "uav_flow_benchmark",
                "case_id": self.case.case_id,
                "capture_group": f"rgb_depth_{self._frame_index:06d}",
                "synchronized_with": "depth",
            },
            "depth": {
                "path": str(depth_path),
                "width": width,
                "height": height,
                "mock": False,
                "source": "uav_flow_unrealcv_research_capture",
                "case_id": self.case.case_id,
                "fov_degrees": 90,
                "depth_unit": "meter",
                "raw_depth_unit": "centimeter",
                "command": commands[1],
                "capture_group": f"rgb_depth_{self._frame_index:06d}",
                "synchronized_with": "rgb",
            },
        }

    def _depth_frame(self) -> dict[str, Any]:
        if not (self.enable_depth_capture or self.enable_research_frame_capture):
            return {"mock": True, "reason": "uav_flow_v1_rgb_pose_only"}
        if not self.real_unrealcv or self.env is None:
            return {
                "mock": True,
                "reason": "research_frame_capture_requires_real_unrealcv",
                "research_frame_capture": "unavailable",
            }
        self._sync_camera_to_player()
        path = self._next_research_frame_path("depth", "npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self._request_unrealcv_frame("vget /camera/0/depth npy")
        self._write_depth_response(path, response)
        return {
            "path": str(path),
            "mock": False,
            "source": "uav_flow_unrealcv_research_capture",
            "case_id": self.case.case_id,
            "fov_degrees": 90,
            "depth_unit": "meter",
            "raw_depth_unit": "centimeter",
            "command": "vget /camera/0/depth npy",
        }

    def _mask_frame(self) -> dict[str, Any]:
        if not self.enable_research_frame_capture:
            return {
                "mock": True,
                "reason": "research_frame_capture_disabled",
                "research_frame_capture": "disabled",
            }
        if not self.real_unrealcv or self.env is None:
            return {
                "mock": True,
                "reason": "research_frame_capture_requires_real_unrealcv",
                "research_frame_capture": "unavailable",
            }
        self._sync_camera_to_player()
        path = self._next_research_frame_path("mask", "png")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self._request_unrealcv_frame("vget /camera/0/object_mask png")
        self._write_bytes_response(path, response)
        return {
            "path": str(path),
            "mock": False,
            "source": "uav_flow_unrealcv_research_capture",
            "case_id": self.case.case_id,
            "fov_degrees": 90,
            "command": "vget /camera/0/object_mask png",
        }

    def _next_research_frame_path(self, folder: str, suffix: str) -> Path:
        while True:
            self._frame_index += 1
            path = self.run_dir / "artifacts" / folder / f"frame_{self._frame_index:06d}.{suffix}"
            if not path.exists():
                return path

    def _request_unrealcv_frame(self, command: str) -> Any:
        if self.env is None:
            raise RuntimeError("UAV-Flow real UnrealCV environment is not started")
        unrealcv = self.env.unwrapped.unrealcv
        client = getattr(unrealcv, "client", None)
        request = getattr(client, "request", None)
        if callable(request):
            return request(command)
        request = getattr(unrealcv, "request", None)
        if callable(request):
            return request(command)
        raise RuntimeError("UAV-Flow UnrealCV object does not expose a request API for research frame capture")

    def _write_depth_response(self, path: Path, response: Any) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("Depth research capture requires numpy") from exc

        if isinstance(response, (bytes, bytearray, memoryview)):
            array = np.load(BytesIO(bytes(response)))
            self._write_depth_array_meters(path, array)
            return
        if isinstance(response, str):
            source_path = Path(response)
            if source_path.exists():
                array = np.load(source_path)
                self._write_depth_array_meters(path, array)
                return
        array = np.asarray(response, dtype=np.float32)
        self._write_depth_array_meters(path, array)

    def _write_depth_array_meters(self, path: Path, array: Any) -> None:
        import numpy as np

        depth_m = np.asarray(array, dtype=np.float32) * 0.01
        buffer = BytesIO()
        np.save(buffer, depth_m)
        path.write_bytes(buffer.getvalue())

    def _write_bytes_response(self, path: Path, response: Any) -> None:
        if isinstance(response, (bytes, bytearray, memoryview)):
            path.write_bytes(bytes(response))
            return
        if isinstance(response, str):
            source_path = Path(response)
            if source_path.exists():
                path.write_bytes(source_path.read_bytes())
                return
            raise RuntimeError(f"UnrealCV image command returned text instead of image bytes: {response[:120]}")
        raise RuntimeError(f"Unsupported UnrealCV image response type for {path.suffix}: {type(response).__name__}")

    def collect_research_mask_label_map(
        self,
        visible_colors: set[tuple[int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Collect conservative object-mask labels for standalone custom_map runs.

        UnrealCV masks expose colors, not semantic classes. This method queries
        the simulator's object inventory and only assigns a semantic label when
        the object name supports a conservative category. Unclassified colors
        stay unknown in custom_map instead of being guessed.
        """

        if not self.enable_research_frame_capture or not self.real_unrealcv or self.env is None:
            return {"colors": {}, "unmapped_objects": [], "source": "research_frame_capture_unavailable"}
        color_dict = self._research_object_color_dict()
        visible = set(visible_colors or set())
        colors: dict[str, dict[str, Any]] = {}
        unmapped_objects: list[dict[str, Any]] = []
        for object_name in sorted(color_dict):
            color = color_dict[object_name]
            if visible and color not in visible:
                continue
            semantic = _semantic_label_for_object_name(object_name)
            if semantic is None:
                unmapped_objects.append(
                    {
                        "object_name": object_name,
                        "color": list(color),
                        "reason": "object_name_not_classified",
                    }
                )
                continue
            colors[_color_key(color)] = {
                "id": semantic["id"],
                "label": semantic["label"],
                "category": semantic["category"],
                "object_name": object_name,
                "source": "uav_flow_unrealcv_object_inventory",
                "confidence": "name_heuristic",
            }
        return {
            "colors": colors,
            "unmapped_objects": unmapped_objects,
            "visible_color_count": len(visible),
            "object_count": len(color_dict),
            "source": "uav_flow_unrealcv_object_inventory",
        }

    def _research_object_color_dict(self) -> dict[str, tuple[int, int, int]]:
        if self.env is None:
            return {}
        unrealcv = self.env.unwrapped.unrealcv
        get_objects = getattr(unrealcv, "get_objects", None)
        objects = get_objects() if callable(get_objects) else []
        if not isinstance(objects, list):
            return {}
        build_color_dict = getattr(unrealcv, "build_color_dict", None)
        raw_colors = build_color_dict(objects) if callable(build_color_dict) else {}
        if not isinstance(raw_colors, dict):
            raw_colors = {}
        colors: dict[str, tuple[int, int, int]] = {}
        for object_name in objects:
            color = _normalize_color(raw_colors.get(object_name))
            if color is None:
                get_obj_color = getattr(unrealcv, "get_obj_color", None)
                if callable(get_obj_color):
                    try:
                        color = _normalize_color(get_obj_color(object_name))
                    except Exception:
                        color = None
            if color is not None:
                colors[str(object_name)] = color
        return colors

    def _write_dry_rgb(self, path: Path) -> None:
        image = Image.new("RGB", (224, 224), (242, 242, 238))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 40, 90, 150), fill=(210, 48, 48))
        draw.ellipse((132, 70, 184, 122), fill=(40, 90, 210))
        draw.line((20, 190, 204, 190), fill=(40, 40, 40), width=3)
        image.save(path)

    def _model_pose(self) -> dict[str, Any]:
        return world_pose_to_model_pose(self.world_pose_cm, self.case.initial_pos)

    def _record_pose(self) -> dict[str, Any]:
        pose = self._model_pose()
        self.pose_history.append(pose)
        return pose


def _normalize_yaw_world(value: float) -> float:
    return float(value) % 360.0


def _semantic_label_for_object_name(object_name: str) -> dict[str, Any] | None:
    name = object_name.lower()
    categories = [
        (10, "person", "dynamic_actor", ("character", "human", "person", "bp_character")),
        (11, "vehicle", "dynamic_actor", ("car", "vehicle", "bus", "truck", "bp_basecar")),
        (12, "sunshade", "landmark", ("sunshade", "umbrella", "parasol")),
        (20, "route_surface", "navigation_surface", ("road", "street", "asphalt", "lane", "sidewalk", "crosswalk")),
        (30, "building_structure", "structure", ("building", "house", "wall", "roof", "window", "door")),
        (40, "vegetation", "natural", ("tree", "plant", "grass", "bush")),
        (50, "street_fixture", "fixture", ("pole", "sign", "traffic", "lamp", "light")),
    ]
    for semantic_id, label, category, keywords in categories:
        if any(keyword in name for keyword in keywords):
            return {"id": semantic_id, "label": label, "category": category}
    return None


def _normalize_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = []
        for token in value.replace(",", " ").replace("(", " ").replace(")", " ").split():
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
    else:
        try:
            values = [int(float(item)) for item in value]
        except (TypeError, ValueError):
            return None
    if len(values) < 3:
        return None
    return tuple(max(0, min(255, value)) for value in values[:3])


def _color_key(color: tuple[int, int, int]) -> str:
    return f"{color[0]},{color[1]},{color[2]}"
