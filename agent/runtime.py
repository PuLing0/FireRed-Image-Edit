"""Comprehensive closed-loop agent runtime for FireRed-Image-Edit."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable

from PIL import Image

from agent.critic import HybridCritic
from agent.memory import TaskMemory
from agent.planner import RuleBasedPlanner
from agent.router import ToolRouter
from agent.runtime_types import (
    AgentRunResult,
    AgentRuntimeOptions,
    CriticResult,
    ExecutionPlan,
    RepairAction,
)
from agent.tools import AgentTool, build_tool_registry
from agent.workspace import Workspace


class AgentRuntime:
    """Unified planning → tool routing → workspace → critic → repair loop."""

    def __init__(
        self,
        *,
        edit_tool: AgentTool | None = None,
        tools: dict[str, AgentTool] | None = None,
        planner: RuleBasedPlanner | None = None,
        critic: HybridCritic | None = None,
        max_output_images: int = 3,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.max_output_images = max_output_images
        registry = tools or build_tool_registry(edit_tool=edit_tool)
        if "image_edit" not in registry:
            raise ValueError("AgentRuntime requires an image_edit tool.")
        self.router = ToolRouter(registry)
        self.planner = planner or RuleBasedPlanner(max_output_images=max_output_images)
        self.critic = critic or HybridCritic(registry, enable_vlm=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[AgentRuntime] {message}")

    def run(
        self,
        images: Iterable[Image.Image],
        user_goal: str,
        options: AgentRuntimeOptions | None = None,
    ) -> AgentRunResult:
        """Execute the comprehensive agent loop."""
        options = options or AgentRuntimeOptions()
        self.critic.enable_vlm = options.enable_vlm_critic

        workspace = Workspace()
        workspace.seed_inputs(list(images), user_goal)
        memory = TaskMemory()

        repair_hints: dict[str, Any] = {}
        final_critic: CriticResult | None = None

        for iteration in range(options.max_plan_iterations):
            plan = self.planner.build_plan(
                goal=user_goal,
                workspace=workspace,
                options=options,
                repair_round=memory.repair_round,
                repair_hints=repair_hints,
            )
            memory.add_plan(plan)
            self._log(
                f"Running plan {plan.plan_id} (repair_round={plan.repair_round}) with "
                f"{len(plan.steps)} step(s)."
            )
            plan_start_ckpt = workspace.create_checkpoint(f"plan_start_{plan.plan_id}")

            step_failure: CriticResult | None = None
            for step in plan.steps:
                memory.add_event(
                    "step_start",
                    f"Starting {step.kind}",
                    {"step_id": step.step_id, "tool_name": step.tool_name},
                )
                tool = self.router.resolve(step, workspace)
                self._log(f"Step {step.kind}: executing {step.tool_name}.")
                result = tool.run(workspace, step, memory)
                memory.add_event(
                    "tool_result",
                    result.summary,
                    {
                        "step_id": step.step_id,
                        "tool_name": step.tool_name,
                        "status": result.status,
                        "artifacts": list(result.produced_artifact_ids),
                        "records": list(result.produced_record_ids),
                        "metadata": dict(result.metadata),
                    },
                )
                step_review = self.critic.evaluate_step(
                    workspace=workspace,
                    step=step,
                    tool_result=result,
                    memory=memory,
                )
                memory.add_event(
                    "step_critic",
                    "; ".join(step_review.reasons),
                    {
                        "step_id": step.step_id,
                        "status": step_review.status,
                        "repair_actions": [
                            dataclasses.asdict(action)
                            for action in step_review.repair_actions
                        ],
                    },
                )
                if step_review.status != "pass":
                    step_failure = step_review
                    self._log(
                        f"Step {step.kind} failed critic checks: "
                        + "; ".join(step_review.reasons)
                    )
                    break

                if options.save_checkpoints:
                    workspace.create_checkpoint(f"after_{step.step_id}")

            if step_failure is None:
                final_critic = self.critic.evaluate_final(
                    workspace=workspace,
                    memory=memory,
                    goal=user_goal,
                )
                memory.add_event(
                    "final_critic",
                    "; ".join(final_critic.reasons),
                    {
                        "status": final_critic.status,
                        "score": final_critic.score,
                        "repair_actions": [
                            dataclasses.asdict(action)
                            for action in final_critic.repair_actions
                        ],
                    },
                )
                if final_critic.status == "pass":
                    self._log("Final candidate accepted by critic.")
                    return self._build_result(
                        workspace=workspace,
                        memory=memory,
                        final_critic=final_critic,
                        final_status="success",
                    )
                step_failure = final_critic
                self._log(
                    "Final critic requested repair: "
                    + "; ".join(final_critic.reasons)
                )

            assert step_failure is not None
            if memory.repair_round >= options.max_repair_rounds:
                final_critic = step_failure
                self._log("Repair budget exhausted; returning last candidate.")
                break

            action = self._select_repair_action(step_failure.repair_actions)
            if action is None:
                final_critic = step_failure
                self._log("No repair action available; returning last candidate.")
                break

            memory.record_failure(action.action_type)
            memory.repair_round += 1
            memory.add_event(
                "repair",
                f"Applying repair action {action.action_type}",
                {"reason": action.reason, "params": dict(action.params)},
            )
            repair_hints = self._merge_repair_hints(repair_hints, action)
            workspace.restore_checkpoint(plan_start_ckpt.checkpoint_id)
            self._log(
                f"Repair action selected: {action.action_type} "
                f"({action.reason or 'no additional reason'})"
            )

        return self._build_result(
            workspace=workspace,
            memory=memory,
            final_critic=final_critic,
            final_status="failed",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_repair_action(
        self,
        actions: list[RepairAction],
    ) -> RepairAction | None:
        return actions[0] if actions else None

    def _merge_repair_hints(
        self,
        hints: dict[str, Any],
        action: RepairAction,
    ) -> dict[str, Any]:
        updated = dict(hints)
        if action.action_type == "disable_roi":
            updated["disable_roi"] = True
        elif action.action_type == "force_prompt_rewrite":
            updated["force_prompt_rewrite"] = True
            guidance = action.params.get("prompt_guidance") or action.reason
            if guidance:
                updated["prompt_guidance"] = guidance
        elif action.action_type == "balanced_stitch":
            updated["balanced_stitch"] = True
        elif action.action_type == "retry_edit_seed":
            # No special hint needed: the edit tool varies the seed by repair round.
            pass
        return updated

    def _build_result(
        self,
        *,
        workspace: Workspace,
        memory: TaskMemory,
        final_critic: CriticResult | None,
        final_status: str,
    ) -> AgentRunResult:
        if workspace.active_candidate_id is not None:
            final_images = [workspace.get_image(workspace.active_candidate_id)]
        else:
            final_images = workspace.current_images()

        return AgentRunResult(
            final_images=final_images,
            final_prompt=workspace.current_prompt(),
            final_status=final_status,
            workspace_snapshot={
                **workspace.snapshot(),
                "memory": memory.snapshot(),
            },
            execution_trace=memory.snapshot()["events"],
            plans=memory.snapshot()["plans"],
            critic_summary={
                "status": final_critic.status if final_critic else "unknown",
                "reasons": final_critic.reasons if final_critic else [],
                "score": final_critic.score if final_critic else None,
                "repair_actions": [
                    dataclasses.asdict(action)
                    for action in (final_critic.repair_actions if final_critic else [])
                ],
                "accepted_artifact_id": (
                    final_critic.accepted_artifact_id if final_critic else None
                ),
                "metadata": final_critic.metadata if final_critic else {},
            },
            group_indices=[list(group) for group in workspace.group_indices],
            rois=[dict(roi) for roi in workspace.rois],
            debug_image_artifacts={
                artifact_id: artifact.data
                for artifact_id, artifact in workspace.artifacts.items()
                if artifact.kind == "image"
            },
        )


__all__ = ["AgentRuntime"]
