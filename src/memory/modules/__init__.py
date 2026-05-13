"""Default context modules."""

from src.memory.modules.action import DefaultActionContextModule
from src.memory.modules.image import DefaultImageContextModule
from src.memory.modules.instruction import DefaultInstructionContextModule
from src.memory.modules.map import DefaultMapContextModule
from src.memory.modules.pose import DefaultPoseContextModule

DEFAULT_MODULES = {
    "instruction": DefaultInstructionContextModule,
    "image": DefaultImageContextModule,
    "pose": DefaultPoseContextModule,
    "action": DefaultActionContextModule,
    "map": DefaultMapContextModule,
}
