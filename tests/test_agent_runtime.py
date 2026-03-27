"""Tests for the comprehensive FireRed agent runtime."""

from __future__ import annotations

import unittest

from PIL import Image

from agent.critic import HybridCritic
from agent.runtime import AgentRuntime
from agent.runtime_types import AgentRuntimeOptions, PlanStep, ToolResult
from agent.tools import CropTool, DetectOrSegmentTool, PromptRewriteTool, StitchTool
from agent.workspace import Workspace


class FakeEditTool:
    """Deterministic edit tool used for runtime tests."""

    name = "image_edit"

    def __init__(self, *, change_on_attempt: int = 1) -> None:
        self.change_on_attempt = change_on_attempt

    def run(self, workspace: Workspace, step: PlanStep, memory) -> ToolResult:
        input_ids = list(workspace.current_image_ids)
        base_image = workspace.get_image(input_ids[0]).copy().convert("RGB")
        attempt_index = int(step.params.get("attempt_index", 0))
        if attempt_index >= self.change_on_attempt:
            base_image = Image.new("RGB", base_image.size, (255, 0, 0))

        artifact_id = workspace.add_image(
            base_image,
            label=f"fake_candidate_{attempt_index}",
            step_id=step.step_id,
            parents=input_ids + [workspace.current_prompt_id],
            metadata={
                "edit_input_ids": list(input_ids),
                "prompt_id": workspace.current_prompt_id,
                "attempt_index": attempt_index,
            },
        )
        workspace.set_active_candidate(artifact_id)
        return ToolResult(
            status="success",
            summary="Fake edit candidate created.",
            produced_artifact_ids=[artifact_id],
        )


def _full_image_rois(images, instruction):
    return [
        {"image_index": idx, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        for idx in range(len(images))
    ]


class AgentRuntimeTest(unittest.TestCase):
    """Core runtime behavior."""

    def _make_runtime(self, edit_tool: FakeEditTool) -> AgentRuntime:
        registry = {
            "detect_or_segment": DetectOrSegmentTool(detect_fn=_full_image_rois),
            "crop": CropTool(),
            "stitch": StitchTool(),
            "prompt_rewrite": PromptRewriteTool(),
            "image_edit": edit_tool,
        }
        critic = HybridCritic(registry, enable_vlm=False)
        return AgentRuntime(
            tools=registry,
            critic=critic,
            verbose=False,
        )

    def test_workspace_checkpoint_restore(self) -> None:
        workspace = Workspace()
        workspace.seed_inputs([Image.new("RGB", (8, 8), "white")], "make it blue")
        original_prompt = workspace.current_prompt()
        checkpoint = workspace.create_checkpoint("initial")

        new_prompt_id = workspace.add_text(
            "new prompt",
            label="current_prompt",
            step_id="test",
            parents=[workspace.current_prompt_id],
        )
        workspace.set_current_prompt(new_prompt_id)
        workspace.restore_checkpoint(checkpoint.checkpoint_id)

        self.assertEqual(workspace.current_prompt(), original_prompt)

    def test_runtime_single_image_uses_short_plan(self) -> None:
        runtime = self._make_runtime(FakeEditTool(change_on_attempt=0))
        result = runtime.run(
            [Image.new("RGB", (16, 16), "white")],
            "turn the square blue",
            options=AgentRuntimeOptions(
                enable_recaption=False,
                enable_input_understanding=False,
                enable_vlm_critic=False,
                max_repair_rounds=0,
                max_plan_iterations=1,
            ),
        )

        self.assertEqual(result.final_status, "success")
        self.assertEqual(len(result.plans), 1)
        self.assertEqual(len(result.plans[0]["steps"]), 1)
        self.assertEqual(result.plans[0]["steps"][0]["tool_name"], "image_edit")

    def test_runtime_repair_loop_retries_and_succeeds(self) -> None:
        images = [
            Image.new("RGB", (32, 32), color)
            for color in ("white", "blue", "green", "yellow")
        ]
        runtime = self._make_runtime(FakeEditTool(change_on_attempt=1))
        result = runtime.run(
            images,
            "place the accessory from image 2 onto image 1",
            options=AgentRuntimeOptions(
                enable_recaption=False,
                enable_input_understanding=False,
                enable_vlm_critic=False,
                max_repair_rounds=2,
                max_plan_iterations=3,
            ),
        )

        self.assertEqual(result.final_status, "success")
        self.assertGreaterEqual(len(result.plans), 2)
        repair_events = [
            event for event in result.execution_trace
            if event["kind"] == "repair"
        ]
        self.assertTrue(repair_events)
        self.assertEqual(repair_events[0]["payload"]["reason"], "The edited result is almost unchanged from the edit input.")

    def test_runtime_fails_when_repair_budget_exhausted(self) -> None:
        runtime = self._make_runtime(FakeEditTool(change_on_attempt=99))
        result = runtime.run(
            [Image.new("RGB", (16, 16), "white")],
            "change the object color",
            options=AgentRuntimeOptions(
                enable_recaption=False,
                enable_input_understanding=False,
                enable_vlm_critic=False,
                max_repair_rounds=0,
                max_plan_iterations=1,
            ),
        )

        self.assertEqual(result.final_status, "failed")
        self.assertEqual(result.critic_summary["status"], "soft_fail")


if __name__ == "__main__":
    unittest.main()
