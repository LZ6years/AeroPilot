"""Default context modules."""

from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.modules.motion import DefaultMotionContextModule
from src.memory.context.modules.pose import DefaultPoseContextModule
from src.memory.context.modules.skills import DefaultSkillsContextModule

DEFAULT_MODULES = {
    "skills": DefaultSkillsContextModule,
    "image": DefaultImageContextModule,
    "pose": DefaultPoseContextModule,
    "motion": DefaultMotionContextModule,
    "map": DefaultMapContextModule,
}
