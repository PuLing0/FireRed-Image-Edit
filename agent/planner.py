"""Planner for the comprehensive FireRed agent runtime."""

from __future__ import annotations

from typing import Any

from agent.runtime_types import AgentRuntimeOptions, ExecutionPlan, PlanStep, _make_id
from agent.workspace import Workspace


_TEXT_HINTS = (
    "text",
    "文字",
    "字体",
    "font",
    "caption",
    "title",
    "logo",
    "招牌",
    "海报",
)


def _looks_like_text_task(goal: str) -> bool:
    """Return whether *goal* likely involves explicit text editing."""
    lowered = goal.lower()
    return any(hint in lowered for hint in _TEXT_HINTS)


class RuleBasedPlanner:
    """Heuristic planner for the first comprehensive-agent iteration."""

    def __init__(self, max_output_images: int = 3) -> None:
        self.max_output_images = max_output_images

    def build_plan(
        self,
        *,
        goal: str,
        workspace: Workspace,
        options: AgentRuntimeOptions,
        repair_round: int = 0,
        repair_hints: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Create an execution plan for the current workspace state."""
        repair_hints = repair_hints or {}
        num_inputs = len(workspace.input_image_ids)
        requires_preprocess = num_inputs > self.max_output_images
        text_task = _looks_like_text_task(goal)

        use_roi = requires_preprocess and not repair_hints.get("disable_roi", False)
        background_first = not repair_hints.get("balanced_stitch", False)
        force_prompt_rewrite = repair_hints.get("force_prompt_rewrite", False)
        enable_prompt_rewrite = (
            requires_preprocess or options.enable_recaption or force_prompt_rewrite
        )
        prompt_guidance = repair_hints.get("prompt_guidance", "")

        steps: list[PlanStep] = []

        if options.enable_input_understanding:
            steps.append(
                PlanStep(
                    step_id=_make_id("step"),
                    kind="input_understanding",
                    tool_name="image_understanding",
                    purpose="Summarize input images for planning and later verification.",
                    params={"source": "inputs", "optional": True},
                    success_checks=["A structured summary or graceful skip is recorded."],
                )
            )

        if text_task:
            steps.append(
                PlanStep(
                    step_id=_make_id("step"),
                    kind="ocr_inputs",
                    tool_name="ocr",
                    purpose="Extract visible source text before editing.",
                    params={"source": "inputs", "optional": True},
                    success_checks=["OCR text is recorded or the step cleanly skips."],
                )
            )

        if requires_preprocess:
            if use_roi:
                steps.append(
                    PlanStep(
                        step_id=_make_id("step"),
                        kind="detect_regions",
                        tool_name="detect_or_segment",
                        purpose="Detect the most relevant ROI in each source image.",
                        success_checks=["A ROI record exists for every input image."],
                    )
                )
                steps.append(
                    PlanStep(
                        step_id=_make_id("step"),
                        kind="crop_inputs",
                        tool_name="crop",
                        purpose="Crop each input image to its selected ROI.",
                        params={"source": "inputs"},
                        success_checks=["Cropped image artifacts are created for all inputs."],
                    )
                )

            steps.append(
                PlanStep(
                    step_id=_make_id("step"),
                    kind="stitch_inputs",
                    tool_name="stitch",
                    purpose="Convert many input images into <=3 composite images.",
                    params={"background_first": background_first},
                    success_checks=["Composite image count is <= runtime max output images."],
                )
            )

        if enable_prompt_rewrite:
            steps.append(
                PlanStep(
                    step_id=_make_id("step"),
                    kind="rewrite_prompt",
                    tool_name="prompt_rewrite",
                    purpose="Rewrite prompt references and improve edit specificity.",
                    params={
                        "llm_enabled": options.enable_recaption,
                        "guidance": prompt_guidance,
                        "target_length": options.target_prompt_length,
                    },
                    success_checks=["A current prompt artifact is updated."],
                )
            )

        steps.append(
            PlanStep(
                step_id=_make_id("step"),
                kind="image_edit",
                tool_name="image_edit",
                purpose="Run FireRed image editing on the current images and prompt.",
                params={"attempt_index": repair_round},
                success_checks=["A new edited candidate image is produced."],
            )
        )

        success_checks = [
            "The final candidate image exists.",
            "The final candidate passes rule-based validation.",
            "If VLM critic is enabled, the final candidate passes critic acceptance.",
        ]
        if requires_preprocess:
            success_checks.append("The composite image count remains within FireRed limits.")
        if text_task:
            success_checks.append("Visible text edits do not obviously regress after editing.")

        return ExecutionPlan(
            plan_id=_make_id("plan"),
            goal=goal,
            repair_round=repair_round,
            steps=steps,
            success_checks=success_checks,
            budget={
                "max_repair_rounds": options.max_repair_rounds,
                "max_plan_iterations": options.max_plan_iterations,
            },
            metadata={
                "requires_preprocess": requires_preprocess,
                "use_roi": use_roi,
                "background_first": background_first,
                "text_task": text_task,
                "prompt_guidance": prompt_guidance,
            },
        )


__all__ = ["RuleBasedPlanner"]
