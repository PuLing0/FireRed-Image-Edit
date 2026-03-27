"""Tool router for the comprehensive FireRed agent runtime."""

from __future__ import annotations

from agent.runtime_types import PlanStep
from agent.tools import AgentTool
from agent.workspace import Workspace


class ToolRouter:
    """Resolve a planned step to a concrete tool implementation."""

    def __init__(self, registry: dict[str, AgentTool]) -> None:
        self.registry = registry

    def resolve(self, step: PlanStep, workspace: Workspace) -> AgentTool:
        """Return the tool that should execute *step*."""
        try:
            tool = self.registry[step.tool_name]
        except KeyError as exc:
            raise KeyError(f"No tool registered for {step.tool_name!r}.") from exc

        if step.tool_name == "image_edit" and not workspace.current_image_ids:
            raise RuntimeError("Image edit step requested without active images.")
        return tool


__all__ = ["ToolRouter"]
