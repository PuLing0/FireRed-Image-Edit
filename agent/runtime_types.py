"""Shared runtime types for the comprehensive FireRed agent."""

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import Any

from PIL import Image


def _make_id(prefix: str) -> str:
    """Return a short runtime-scoped identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclasses.dataclass
class Artifact:
    """A workspace artifact produced by an agent step."""

    artifact_id: str
    kind: str
    data: Any
    label: str
    step_id: str
    parents: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of the artifact."""
        info: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "label": self.label,
            "step_id": self.step_id,
            "parents": list(self.parents),
            "metadata": dict(self.metadata),
        }
        if isinstance(self.data, Image.Image):
            info["image"] = {
                "size": self.data.size,
                "mode": self.data.mode,
            }
        elif isinstance(self.data, str):
            info["text_preview"] = self.data[:200]
        elif isinstance(self.data, dict):
            info["record_keys"] = sorted(self.data.keys())
        else:
            info["repr"] = repr(self.data)[:200]
        return info


@dataclasses.dataclass
class PlanStep:
    """A single executable step in an execution plan."""

    step_id: str
    kind: str
    tool_name: str
    purpose: str
    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    success_checks: list[str] = dataclasses.field(default_factory=list)
    allow_repair: bool = True


@dataclasses.dataclass
class ExecutionPlan:
    """Planner output for one runtime iteration."""

    plan_id: str
    goal: str
    repair_round: int
    steps: list[PlanStep]
    success_checks: list[str]
    budget: dict[str, int]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RepairAction:
    """A critic-suggested repair action."""

    action_type: str
    reason: str
    params: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ToolResult:
    """Normalized output returned by a tool."""

    status: str
    summary: str
    produced_artifact_ids: list[str] = dataclasses.field(default_factory=list)
    produced_record_ids: list[str] = dataclasses.field(default_factory=list)
    raw_output: Any = None
    retryable: bool = True
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CriticResult:
    """Decision returned by the critic."""

    status: str
    reasons: list[str]
    repair_actions: list[RepairAction] = dataclasses.field(default_factory=list)
    score: float | None = None
    accepted_artifact_id: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RuntimeEvent:
    """Execution event stored in task memory."""

    event_id: str
    timestamp: float
    kind: str
    message: str
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> "RuntimeEvent":
        """Create a timestamped event."""
        return cls(
            event_id=_make_id("evt"),
            timestamp=time.time(),
            kind=kind,
            message=message,
            payload=payload or {},
        )


@dataclasses.dataclass
class AgentRuntimeOptions:
    """Configuration for the comprehensive agent runtime."""

    enable_recaption: bool = True
    max_repair_rounds: int = 2
    max_plan_iterations: int = 3
    enable_input_understanding: bool = True
    enable_vlm_critic: bool = True
    save_checkpoints: bool = True
    target_prompt_length: int | None = None


@dataclasses.dataclass
class AgentRunResult:
    """Final output of the comprehensive agent runtime."""

    final_images: list[Image.Image]
    final_prompt: str
    final_status: str
    workspace_snapshot: dict[str, Any]
    execution_trace: list[dict[str, Any]]
    plans: list[dict[str, Any]]
    critic_summary: dict[str, Any]
    group_indices: list[list[int]]
    rois: list[dict[str, Any]]
    debug_image_artifacts: dict[str, Image.Image] = dataclasses.field(default_factory=dict)

    @property
    def images(self) -> list[Image.Image]:
        """Compatibility alias matching the legacy agent result."""
        return self.final_images

    @property
    def prompt(self) -> str:
        """Compatibility alias matching the legacy agent result."""
        return self.final_prompt


__all__ = [
    "AgentRunResult",
    "AgentRuntimeOptions",
    "Artifact",
    "CriticResult",
    "ExecutionPlan",
    "PlanStep",
    "RepairAction",
    "RuntimeEvent",
    "ToolResult",
    "_make_id",
]
