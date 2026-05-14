"""Windows 本地 UAV Client 调用示例。

这个文件不是 pytest 单元测试，而是给你手动运行、观察调用流程的 demo。

推荐先运行这条命令：它会启动 Windows CityEnviron 地图，然后依次演示连接、
起飞、悬停、直接放置 set_pose、前进、转向、上升、下降、取 pose、取 RGB/depth，最后降落。

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch

如果你想打开 UE 窗口并看到 AirSim FPV 画面，可以加 --show-display：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --show-display

如果你想保存本次 demo 的日志、观测 JSON 和图片 artifact，可以加 --save-artifacts：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --show-display --save-artifacts

如果你想保持无显示/后台模式，可以加 --hide-display：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --hide-display

如果你想测试把无人机直接放到某个位置，可以指定 set_pose 的目标位姿：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --show-display --set-pose-x 0 --set-pose-y 0 --set-pose-z -3 --set-pose-yaw-deg 90

如果你想让前进动作更明显，可以延长移动时间：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --show-display --move-duration 8

默认动作尺度来自 configs/uav_action_config.json；你也可以临时指定另一个配置：

    python test_uav_client_demo.py --map-id cityenviron_win64 --launch --action-config configs/uav_action_config.json

如果你已经手动打开了 Windows CityEnviron，并且 AirSim 端口是 41451，可以运行：

    python test_uav_client_demo.py --remote-host 127.0.0.1 --remote-port 41451 --vehicle-name Drone1 --camera-name front_center --connect-existing

如果你只想看底层 AirSimUavAdapter 怎么直接调用，不经过 EnvClient，可以运行：

    python test_uav_client_demo.py --path direct-adapter --remote-host 127.0.0.1 --remote-port 41451 --vehicle-name Drone1 --camera-name front_center

如果是 AirVLN ComputerVision 这种不能飞控、只能取观测的模式，可以运行：

    python test_uav_client_demo.py --remote-host 127.0.0.1 --remote-port 30001 --mode ComputerVision --safe-observe-only

注意：
- Windows CityEnviron 的 vehicle_name 通常是 Drone1，camera_name 通常是 front_center。
- AirVLN ComputerVision 的 vehicle_name 通常是 Drone_1，camera_name 通常是 front_0。
- 不确定当前无人机状态时，可以加 --safe-observe-only，只连接和取图，不起飞。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.adapters.airsim_uav import AirSimUavAdapter
from src.env_client.local_client import LocalEnvClient
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.runtime.configs.map_config import build_remote_airsim_config, load_map_config
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.logging.run_store import RunStore
from src.sim_server.server import LocalSimulationServer
from src.tools.dispatcher import ToolDispatcher


def build_parser() -> argparse.ArgumentParser:
    # 定义命令行参数；这个 demo 通过参数选择“启动地图”或“连接已启动地图”。
    parser = argparse.ArgumentParser(description="FlyAgent UAV client demo")
    # 地图配置文件。默认读取项目里正式维护的地图配置。
    parser.add_argument("--config", default="configs/maps/airvln_maps.json")
    # UAV 原子动作尺度配置，统一速度、移动时长、上升步长和转向角，方便 Agent 理解动作效果。
    parser.add_argument("--action-config", default="configs/uav_action_config.json")
    # 每次 demo 的日志和 artifact 根目录；默认落到 runs/。
    parser.add_argument("--run-root", default="runs")
    # 保存 RGB/depth/pose 等本地调试 artifact；默认关闭，避免产生太多文件。
    parser.add_argument("--save-artifacts", action="store_true")
    # 要启动/连接的地图 id。Windows 本地飞控测试默认用 cityenviron_win64。
    parser.add_argument("--map-id", default="cityenviron_win64")
    # Agent/UAV 的逻辑 id。当前 demo 里只是记录和路由用，不是 AirSim vehicle_name。
    parser.add_argument("--agent-id", default="uav_1")
    # 启动配置里的 UE packaged exe，例如 CityEnviron.exe。
    parser.add_argument("--launch", action="store_true", help="Launch the configured packaged UE scene")
    # 不启动 UE，只连接已经打开的 AirSim/UE 进程。
    parser.add_argument("--connect-existing", action="store_true", help="Connect to an already running scene")
    # 临时构造一个远端/本机已启动 AirSim 配置；本地已启动 CityEnviron 可写 127.0.0.1。
    parser.add_argument("--remote-host", default=None, help="Build a temporary remote AirSim map package")
    # AirSim RPC 端口。Windows AirSim 默认常见 41451；我们动态启动 CityEnviron 时可能用 30001。
    parser.add_argument("--remote-port", type=int, default=41451)
    # AirSim 里的真实 vehicle 名称。Windows CityEnviron 通常是 Drone1。
    parser.add_argument("--vehicle-name", default="Drone_1")
    # AirSim camera 名称。Windows CityEnviron 通常是 front_center。
    parser.add_argument("--camera-name", default="front_0")
    # set_pose 是“直接放置无人机”的能力，适合快速切换起点或批量采集数据。
    # 注意 AirSim 使用 NED 坐标系：z 是负数表示离地更高，例如 -3 表示大约上方 3 米。
    parser.add_argument("--set-pose-x", type=float, default=0.0)
    parser.add_argument("--set-pose-y", type=float, default=0.0)
    parser.add_argument("--set-pose-z", type=float, default=-3.0)
    parser.add_argument("--set-pose-yaw-deg", type=float, default=45.0)
    # demo 里前进动作的持续时间；不传时使用 action config 里的 move_duration_sec。
    parser.add_argument("--move-duration", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["Multirotor", "ComputerVision"],
        default="Multirotor",
        help="AirSim mode used for temporary remote configs and direct adapter demos",
    )
    parser.add_argument(
        "--path",
        choices=["env-client", "direct-adapter"],
        default="env-client",
        help="Use FlyAgent EnvClient boundary or call AirSimUavAdapter directly",
    )
    parser.add_argument(
        "--safe-observe-only",
        action="store_true",
        help="Only connect and read pose/rgb/depth; useful for ComputerVision scenes",
    )
    # 控制 AirSim settings 里的 ViewMode。show=Fpv，hide=NoDisplay。
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--show-display", action="store_true", help="Use AirSim ViewMode=Fpv")
    display_group.add_argument("--hide-display", action="store_true", help="Use AirSim ViewMode=NoDisplay")
    # 有时你只想测试起飞/移动/取图，不想最后自动降落，就打开这个开关。
    parser.add_argument("--skip-land", action="store_true", help="Do not call land at the end")
    return parser


def main() -> None:
    # 读取参数，并根据 --path 选择演示哪一种调用方式。
    args = build_parser().parse_args()
    if args.path == "direct-adapter":
        run_direct_adapter_demo(args)
    else:
        run_env_client_demo(args)


def run_env_client_demo(args: argparse.Namespace) -> None:
    # 推荐路径：Agent/工具层 -> EnvClient -> SimulationServer -> backend -> AirSim adapter。
    # 这条路径最接近 FlyAgent 后续真正运行时的调用方式。
    run_dir = RunStore(args.run_root).create_run_dir("uav_client_demo")
    logger = AsyncRunLogger(run_dir)
    maps, port_policy = load_demo_config(args)
    action_config = load_demo_action_config(args)
    print_json("uav_action_config", action_config.to_dict())
    selected_map_id = "remote_linux_airsim" if args.remote_host else args.map_id
    selected_map = next(pkg for pkg in maps if pkg.map_id == selected_map_id)
    logger.info("demo started", path="env-client", map_id=selected_map_id, run_dir=str(run_dir))
    logger.write_json("config/map_package.json", selected_map.to_dict())
    logger.write_json("config/uav_action_config.json", action_config.to_dict())
    # 如果传了 --remote-host，就临时生成 remote_linux_airsim；否则使用 --map-id。
    map_id = selected_map_id
    server = LocalSimulationServer(
        maps,
        port_policy,
        # demo 运行记录目录；这里只给 SimulationServer 放 settings/process log 用。
        run_dir=run_dir,
        # --launch=True 时由 FlyAgent 启动 UE exe；False 时不启动进程。
        launch_processes=args.launch,
        # 只要要连接真实 AirSim，就 real_control=True。
        real_control=args.launch or args.connect_existing or bool(args.remote_host),
        # show_display=True 会让后端写 ViewMode=Fpv；False 会写 NoDisplay；None 使用地图默认配置。
        show_display=display_override(args),
        # 统一飞行动作尺度，让 Agent 后续不用自己猜速度和步长。
        action_config=action_config,
        save_artifacts=args.save_artifacts,
    )
    # EnvClient 是 C/S 架构里的 Client，Agent 不应该直接碰 backend。
    env_client = LocalEnvClient(server)
    # ToolDispatcher 演示 Agent 后续会怎么调用 uav.connect/uav.takeoff 这些工具。
    dispatcher = ToolDispatcher(env_client)
    # start_scene 会启动地图或连接已有地图，并返回 SceneInstance。
    scene = None
    try:
        scene = env_client.start_scene(map_id)
        print_json("scene", scene.to_dict())
        logger.write_json("scene_instance.json", scene.to_dict())
        logger.event("scene_started", scene.to_dict())
        # 1. 连接 AirSim。Multirotor 模式会 enableApiControl + arm。
        call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.connect", {}, logger)
        if not args.safe_observe_only:
            # 2. 起飞。ComputerVision 模式不能执行这个动作。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.takeoff", {}, logger)
            # 3. 悬停。让无人机稳定住。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.hover", {}, logger)
            # 4. 直接设置无人机的绝对位置和朝向，类似 AirVLN 的 setPoses，用于快速放置起点。
            #    x/y/z 是 AirSim NED 坐标，yaw_deg 是机头朝向角；ignore_collision=True 表示忽略碰撞直接摆放。
            call_tool(
                dispatcher,
                scene.scene_instance_id,
                args.agent_id,
                "uav.set_pose",
                {
                    "x": args.set_pose_x,
                    "y": args.set_pose_y,
                    "z": args.set_pose_z,
                    "yaw_deg": args.set_pose_yaw_deg,
                    "ignore_collision": True,
                },
                logger,
            )
            # 5. set_pose 后再悬停一次，让飞控进入稳定状态。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.hover", {}, logger)
            # 6. 机体系速度移动：vx 前进，vy 横移，vz 上下，duration 持续时间。
            call_tool(
                dispatcher,
                scene.scene_instance_id,
                args.agent_id,
                "uav.move_body",
                {"direction": "forward"},
                logger,
            )
            # 7. 相对转向；不传 yaw_deg 时会使用 action config 里的 yaw_step_deg。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.turn_relative", {}, logger)
            # 8. 上升一步；不传 meters 时会使用 action config 里的 vertical_step_m。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.ascend", {}, logger)
            # 9. 下降一步；不传 meters 时会使用 action config 里的 vertical_step_m。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.descend", {}, logger)
            
        # 10. 推荐的观测入口：一次性取 rgb/depth/pose。
        observation = dispatcher.observe(scene.scene_instance_id, args.agent_id, ["rgb", "depth", "pose"])
        print_json("observe(rgb/depth/pose)", observation.to_dict())
        logger.write_json("artifacts/observations/step_0001.json", observation.to_dict())
        logger.event("observation_updated", observation.to_dict())
        # 11. 单独取 pose，展示 uav.get_pose 这个 primitive。
        call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.get_pose", {}, logger)
        # 12. 单独取图，展示 uav.get_frames 这个 primitive。
        call_tool(
            dispatcher,
            scene.scene_instance_id,
            args.agent_id,
            "uav.get_frames",
            {"modalities": ["rgb", "depth"]},
            logger,
        )
        if not args.safe_observe_only:
            # 13. 结束前再悬停一次，避免动作后状态不稳。
            call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.hover", {}, logger)
            if not args.skip_land:
                # 14. 默认最后降落；如果你想保持飞行状态，可以加 --skip-land。
                call_tool(dispatcher, scene.scene_instance_id, args.agent_id, "uav.land", {}, logger)
        logger.info("demo completed", run_dir=str(run_dir))
    except Exception as exc:
        logger.event("run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        logger.error("demo failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # 无论中间是否报错，都释放 SceneInstance。
        # 如果是 --launch 启动的本地 UE，这里也会尝试停止 UE 进程。
        if scene is not None:
            env_client.stop_scene(scene.scene_instance_id)
            logger.event("scene_stopped", scene.to_dict())
            print_json("scene_stopped", scene.to_dict())
        logger.close()
        print(f"\nRun record: {run_dir}")


def run_direct_adapter_demo(args: argparse.Namespace) -> None:
    # 底层路径：直接调用 AirSimUavAdapter。
    # 这条路径绕过 EnvClient/SimulationServer，只适合你快速理解 adapter 能力。
    run_dir = RunStore(args.run_root).create_run_dir("uav_client_demo")
    logger = AsyncRunLogger(run_dir)
    host = args.remote_host or "127.0.0.1"
    action_config = load_demo_action_config(args)
    print_json("uav_action_config", action_config.to_dict())
    logger.info("demo started", path="direct-adapter", host=host, port=args.remote_port, run_dir=str(run_dir))
    logger.write_json("config/uav_action_config.json", action_config.to_dict())
    adapter = AirSimUavAdapter(
        host=host,
        port=args.remote_port,
        vehicle_name=args.vehicle_name,
        camera_name=args.camera_name,
        mode=args.mode,
        artifact_dir=run_dir / "artifacts",
        save_artifacts=args.save_artifacts,
    )
    try:
        # 直接调用 adapter.connect。Multirotor 会 enableApiControl，ComputerVision 只连接。
        call_adapter("connect", adapter.connect, logger)
        if not args.safe_observe_only:
            # 以下动作和 EnvClient 路径里的动作一一对应。
            call_adapter("takeoff", adapter.takeoff, logger)
            call_adapter("hover", adapter.hover, logger)
            call_adapter(
                "set_pose",
                lambda: adapter.set_pose(
                    args.set_pose_x,
                    args.set_pose_y,
                    args.set_pose_z,
                    yaw_deg=args.set_pose_yaw_deg,
                    ignore_collision=True,
                ),
                logger,
            )
            call_adapter("hover", adapter.hover, logger)
            call_adapter(
                "move_body",
                lambda: adapter.move_body(
                    action_config.linear_velocity_mps,
                    0.0,
                    0.0,
                    action_config.move_duration_sec,
                ),
                logger,
            )
            call_adapter("turn_relative", lambda: adapter.turn_relative(action_config.yaw_step_deg), logger)
            call_adapter(
                "ascend",
                lambda: adapter.ascend(action_config.vertical_velocity_mps, action_config.vertical_duration_sec),
                logger,
            )
            call_adapter(
                "descend",
                lambda: adapter.descend(action_config.vertical_velocity_mps, action_config.vertical_duration_sec),
                logger,
            )
        # 取 pose 和图像是所有模式下最常用的观测能力。
        call_adapter("get_pose", adapter.get_pose, logger)
        frames = call_adapter("get_frames(rgb/depth)", lambda: adapter.get_frames(["rgb", "depth"]), logger)
        logger.write_json("artifacts/observations/direct_adapter_frames.json", frames.data or {})
        if not args.safe_observe_only:
            call_adapter("hover", adapter.hover, logger)
            if not args.skip_land:
                call_adapter("land", adapter.land, logger)
        logger.info("demo completed", run_dir=str(run_dir))
    except Exception as exc:
        logger.event("run_failed", {"error_type": type(exc).__name__, "error": str(exc)})
        logger.error("demo failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        logger.close()
        print(f"\nRun record: {run_dir}")


def load_demo_config(args: argparse.Namespace):
    # 如果传 --remote-host，就不读正式地图配置，而是临时构造一个 existing_process 配置。
    # 这样你可以连接一个已经打开的 AirSim，不需要把它写进 configs/maps。
    if args.remote_host:
        return build_remote_airsim_config(
            args.remote_host,
            args.remote_port,
            args.vehicle_name,
            args.camera_name,
            mode=args.mode,
        )
    # 默认读取 configs/maps/airvln_maps.json，其中已经包含 cityenviron_win64。
    return load_map_config(args.config)


def display_override(args: argparse.Namespace) -> bool | None:
    # 把 demo 参数转换成后端 show_display 开关。
    if args.show_display:
        return True
    if args.hide_display:
        return False
    return None


def load_demo_action_config(args: argparse.Namespace) -> UavActionConfig:
    # 读取统一动作尺度配置，并允许 --move-duration 临时覆盖演示时长。
    config = load_uav_action_config(args.action_config)
    if args.move_duration is None:
        return config
    return UavActionConfig(
        linear_velocity_mps=config.linear_velocity_mps,
        vertical_velocity_mps=config.vertical_velocity_mps,
        move_duration_sec=args.move_duration,
        vertical_duration_sec=config.vertical_duration_sec,
        vertical_step_m=config.vertical_step_m,
        yaw_step_deg=config.yaw_step_deg,
    )


def call_tool(
    dispatcher: ToolDispatcher,
    scene_instance_id: str,
    agent_id: str,
    tool_name: str,
    args: dict[str, Any],
    logger: AsyncRunLogger | None = None,
) -> None:
    # 调用 FlyAgent 工具名，例如 uav.takeoff，然后打印标准 ActionResult。
    start = time.monotonic()
    if logger is not None:
        logger.event("action_started", {"action": tool_name, "args": args})
    result = dispatcher.call(scene_instance_id, agent_id, tool_name, args)
    duration_ms = round((time.monotonic() - start) * 1000, 3)
    print_json(tool_name, result.to_dict())
    if logger is not None:
        payload = result.to_dict()
        payload["duration_ms"] = duration_ms
        payload["args"] = args
        logger.event("action_completed", payload)
    if not result.ok:
        raise RuntimeError(f"{tool_name} failed: {result.message}")


def call_adapter(label: str, fn, logger: AsyncRunLogger):
    # 调用底层 adapter 方法，并记录动作耗时。
    start = time.monotonic()
    logger.event("action_started", {"action": label, "args": {}})
    result = fn()
    duration_ms = round((time.monotonic() - start) * 1000, 3)
    print_result(label, result)
    payload = {
        "ok": result.ok,
        "action": label,
        "message": result.message,
        "data": result.data or {},
        "duration_ms": duration_ms,
    }
    logger.event("action_completed", payload)
    if not result.ok:
        raise RuntimeError(f"{label} failed: {result.message}")
    return result


def print_result(label: str, result: Any) -> None:
    # 打印底层 adapter 返回的 PrimitiveResult，格式和 ActionResult 类似。
    payload = {
        "ok": result.ok,
        "message": result.message,
        "data": result.data or {},
    }
    print_json(label, payload)


def print_json(label: str, payload: Any) -> None:
    # 统一用 JSON 打印结果，方便你观察每一步返回了什么字段。
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
