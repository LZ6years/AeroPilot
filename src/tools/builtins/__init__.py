"""Built-in ToolSpec factories, independent of Runtime and model clients.

The application ToolRegistry owns registration. These factories do not start
services, discover plugins, or execute the tools they describe.
"""

from src.tools.builtins.knowledge import knowledge_tool
from src.tools.builtins.planning import planning_tool
from src.tools.builtins.terminal import terminal_tools


def builtin_tools():
    return [planning_tool(), knowledge_tool(), *terminal_tools()]
