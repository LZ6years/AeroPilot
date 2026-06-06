"""Default context modules."""

from src.memory.context.modules.action import DefaultActionContextModule
from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.modules.instruction import DefaultInstructionContextModule
from src.memory.context.modules.map import DefaultMapContextModule
from src.memory.context.modules.pose import DefaultPoseContextModule
from src.memory.context.modules.skills import DefaultSkillsContextModule

DEFAULT_MODULES = {
    "instruction": DefaultInstructionContextModule,
    "skills": DefaultSkillsContextModule,
    "image": DefaultImageContextModule,
    "pose": DefaultPoseContextModule,
    "action": DefaultActionContextModule,
    "map": DefaultMapContextModule,
}
