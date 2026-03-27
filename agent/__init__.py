"""FireRed Agent package.

Exports:
1. ``AgentPipeline`` – legacy lightweight pre-processing agent
2. ``AgentRuntime`` – comprehensive planning/tool/workspace/critic agent
"""

from agent.pipeline import AgentPipeline
from agent.runtime import AgentRuntime
from agent.runtime_types import AgentRuntimeOptions
from agent.tools import FireRedEditConfig, FireRedEditTool

__all__ = [
    "AgentPipeline",
    "AgentRuntime",
    "AgentRuntimeOptions",
    "FireRedEditConfig",
    "FireRedEditTool",
]
