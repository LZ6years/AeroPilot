"""Shared constants for the multimodal image caption context module."""

from __future__ import annotations

SCHEMA_VERSION = "phase_visual_summary_v1"
STRATEGY = "qwen_vl_phase_caption_v3"
DEFAULT_VISUAL_DESCRIPTION_MODEL = "qwen3-vl-flash"
MAX_CAPTION_TOKENS = 800

ALLOWED_POSITIONS = {
    "foreground-left",
    "foreground-center",
    "foreground-right",
    "midground-left",
    "midground-center",
    "midground-right",
    "background-left",
    "background-center",
    "background-right",
    "unknown",
}
ALLOWED_ROLES = {"target_candidate", "landmark", "boundary", "free_space", "vehicle", "obstacle", "unknown"}
BANNED_ACTION_TERMS = [
    "turn_left",
    "turn_right",
    "move_forward",
    "recommended_next_observation",
    "左转",
    "右转",
    "前进",
    "靠近",
    "飞向",
    "悬停",
    "适合",
    "飞行",
    "机动",
    "通行",
    "绕行",
    "扫描",
    "航路规划",
    "路径方向",
]
VEHICLE_TERMS = {"车", "车辆", "校车", "轿车", "货车", "厢式车", "SUV", "van", "bus", "car", "truck"}
ENTRANCE_TARGET_TERMS = {"入口", "出入口", "门", "门廊", "门洞", "开口", "入口标识", "建筑底层", "entry", "entrance", "door"}
SOFT_ACTION_REPLACEMENTS = {
    "可通行": "可见空隙",
    "通行使用": "实际使用",
    "通行路径": "可见空隙",
    "通行条件良好": "可见空隙明显",
    "通行条件": "可见空隙",
    "通行设施": "入口附属结构",
    "通行空间": "可见空隙",
    "通行/绕行": "空间间隔",
    "绕行路径": "空间间隔",
    "路径方向": "空间延伸线索",
    "航路规划": "空间布局判断",
    "适合低空悬停": "画面中无遮挡",
    "适合悬停": "画面中无遮挡",
    "适合飞行机动": "画面中无遮挡",
    "适合飞行器空间变化": "画面中可见开阔区域",
    "适合": "呈现为",
    "飞行或移动观察": "开阔可见",
    "飞行或观察": "开阔可见",
    "无人机开阔可见或空间变化扫描": "开阔可见",
    "空间变化扫描": "空间开阔",
    "移动观察": "视觉观察",
    "低空悬停": "开阔可见",
    "飞行机动": "空间开阔",
    "机动观察": "继续作为视觉参照",
    "机动": "空间变化",
    "进一步扫描": "当前画面未确认",
    "进一步观察": "当前画面未确认",
    "需扫描": "当前画面未确认",
    "需要扫描": "当前画面未确认",
    "应扫描": "当前画面未确认",
    "扫描": "视觉确认",
    "需进一步": "当前画面未确认",
    "靠近": "邻近",
    "飞向": "朝向",
    "通行": "可见空隙",
    "飞行": "空间",
    "悬停": "开阔可见",
    "绕行": "空间间隔",
}
