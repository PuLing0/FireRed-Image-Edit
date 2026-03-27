"""Tool adapters used by the comprehensive FireRed agent runtime."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol

from PIL import Image

from agent.gemini_agent import detect_rois
from agent.image_tools import (
    build_group_mapping,
    crop_image_normalized,
    partition_and_stitch,
)
from agent.memory import TaskMemory
from agent.recaption import _replace_image_refs, build_reference_map, recaption
from agent.runtime_types import PlanStep, ToolResult
from agent.visual_tools import ocr_images, score_edit_result, understand_images
from agent.workspace import Workspace

try:
    import torch
except ImportError:  # pragma: no cover - optional for unit tests without torch
    torch = None  # type: ignore[assignment]


class AgentTool(Protocol):
    """Protocol implemented by all tool adapters."""

    name: str

    def run(
        self,
        workspace: Workspace,
        step: PlanStep,
        memory: TaskMemory,
    ) -> ToolResult:
        """Execute the tool and mutate the workspace in place."""


def _source_image_ids(workspace: Workspace, source: str | None) -> list[str]:
    """Resolve the image pointer requested by a tool step."""
    if source == "inputs":
        return list(workspace.input_image_ids)
    return list(workspace.current_image_ids)


class DetectOrSegmentTool:
    """ROI detection tool backed by Gemini."""

    name = "detect_or_segment"

    def __init__(
        self,
        detect_fn: Callable[[list[Image.Image], str], list[dict[str, Any]] | None] = detect_rois,
    ) -> None:
        self.detect_fn = detect_fn

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        image_ids = _source_image_ids(workspace, step.params.get("source"))
        images = [workspace.get_image(artifact_id) for artifact_id in image_ids]
        rois = self.detect_fn(images, workspace.current_prompt())
        fallback_used = False
        if rois is None:
            fallback_used = True
            rois = [
                {"image_index": idx, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
                for idx in range(len(images))
            ]

        workspace.set_rois(rois)
        record_id = workspace.add_record(
            {"rois": rois, "fallback_used": fallback_used},
            label="rois",
            step_id=step.step_id,
            parents=image_ids + [workspace.current_prompt_id],
        )
        return ToolResult(
            status="success",
            summary="Detected ROI regions." if not fallback_used else "ROI detection unavailable; using full images.",
            produced_record_ids=[record_id],
            raw_output=rois,
            retryable=not fallback_used,
            metadata={"fallback_used": fallback_used},
        )


class CropTool:
    """Crop active images to the current ROI set."""

    name = "crop"

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        image_ids = _source_image_ids(workspace, step.params.get("source"))
        if not workspace.rois:
            return ToolResult(
                status="failed",
                summary="No ROI data available for crop step.",
                retryable=False,
            )

        cropped_ids: list[str] = []
        for roi in workspace.rois:
            idx = int(roi["image_index"])
            source_id = image_ids[idx]
            cropped_image = crop_image_normalized(
                workspace.get_image(source_id),
                (roi["x1"], roi["y1"], roi["x2"], roi["y2"]),
            )
            cropped_ids.append(
                workspace.add_image(
                    cropped_image,
                    label=f"cropped_image_{idx}",
                    step_id=step.step_id,
                    parents=[source_id],
                    metadata={"source_index": idx},
                )
            )

        workspace.set_current_images(cropped_ids)
        return ToolResult(
            status="success",
            summary=f"Cropped {len(cropped_ids)} input image(s) to ROI.",
            produced_artifact_ids=cropped_ids,
        )


class StitchTool:
    """Partition and stitch many inputs into <= 3 composites."""

    name = "stitch"

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        image_ids = _source_image_ids(workspace, step.params.get("source"))
        images = [workspace.get_image(artifact_id) for artifact_id in image_ids]
        background_first = bool(step.params.get("background_first", False))

        group_indices, _ = build_group_mapping(
            images,
            max_groups=3,
            background_first=background_first,
        )
        stitched_images = partition_and_stitch(
            images,
            max_groups=3,
            background_first=background_first,
        )

        stitched_ids: list[str] = []
        for idx, image in enumerate(stitched_images):
            parent_ids = [image_ids[source_idx] for source_idx in group_indices[idx]]
            stitched_ids.append(
                workspace.add_image(
                    image,
                    label=f"composite_image_{idx}",
                    step_id=step.step_id,
                    parents=parent_ids,
                    metadata={"group_indices": list(group_indices[idx])},
                )
            )

        record_id = workspace.add_record(
            {"group_indices": group_indices, "background_first": background_first},
            label="group_indices",
            step_id=step.step_id,
            parents=stitched_ids,
        )
        workspace.set_current_images(stitched_ids)
        workspace.set_group_indices(group_indices)
        return ToolResult(
            status="success",
            summary=f"Stitched images into {len(stitched_ids)} composite(s).",
            produced_artifact_ids=stitched_ids,
            produced_record_ids=[record_id],
            raw_output=group_indices,
        )


class PromptRewriteTool:
    """Rewrite image references and expand the prompt when requested."""

    name = "prompt_rewrite"

    def __init__(
        self,
        recaption_fn: Callable[..., str] = recaption,
    ) -> None:
        self.recaption_fn = recaption_fn

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        current_prompt = workspace.current_prompt()
        guidance = str(step.params.get("guidance", "") or "").strip()
        llm_enabled = bool(step.params.get("llm_enabled", True))
        target_length = step.params.get("target_length")

        if llm_enabled:
            rewritten = self.recaption_fn(
                current_prompt,
                workspace.group_indices,
                target_length=target_length or 512,
                additional_guidance=guidance or None,
            )
        else:
            ref_map = build_reference_map(workspace.group_indices)
            rewritten = _replace_image_refs(current_prompt, ref_map)
            if guidance:
                rewritten = f"{rewritten}\n\nAdditional repair guidance: {guidance}"

        prompt_id = workspace.add_text(
            rewritten,
            label="current_prompt",
            step_id=step.step_id,
            parents=[workspace.current_prompt_id],
            metadata={"guidance": guidance, "llm_enabled": llm_enabled},
        )
        workspace.set_current_prompt(prompt_id)
        return ToolResult(
            status="success",
            summary="Prompt rewritten for current workspace state.",
            produced_artifact_ids=[prompt_id],
            raw_output=rewritten,
        )


class ImageUnderstandingTool:
    """Optional Gemini-based visual summarizer."""

    name = "image_understanding"

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        image_ids = _source_image_ids(workspace, step.params.get("source"))
        result = understand_images(
            [workspace.get_image(artifact_id) for artifact_id in image_ids],
            workspace.current_prompt(),
        )
        if result is None:
            return ToolResult(
                status="skipped" if step.params.get("optional", False) else "failed",
                summary="Image understanding unavailable.",
                retryable=False,
            )

        record_id = workspace.add_record(
            result,
            label="image_understanding",
            step_id=step.step_id,
            parents=image_ids + [workspace.current_prompt_id],
        )
        return ToolResult(
            status="success",
            summary="Captured visual understanding summary.",
            produced_record_ids=[record_id],
            raw_output=result,
            retryable=False,
        )


class OCRTool:
    """Optional OCR tool backed by Gemini."""

    name = "ocr"

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        image_ids = _source_image_ids(workspace, step.params.get("source"))
        result = ocr_images([workspace.get_image(artifact_id) for artifact_id in image_ids])
        if result is None:
            return ToolResult(
                status="skipped" if step.params.get("optional", False) else "failed",
                summary="OCR unavailable.",
                retryable=False,
            )

        record_id = workspace.add_record(
            result,
            label="ocr",
            step_id=step.step_id,
            parents=image_ids,
        )
        return ToolResult(
            status="success",
            summary="Extracted visible text from current images.",
            produced_record_ids=[record_id],
            raw_output=result,
            retryable=False,
        )


class ImageScoringTool:
    """Optional Gemini-based image scoring tool."""

    name = "image_scoring"

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        candidate_id = workspace.active_candidate_id
        if candidate_id is None:
            return ToolResult(
                status="failed",
                summary="No active candidate image available for scoring.",
                retryable=False,
            )

        candidate_artifact = workspace.get_artifact(candidate_id)
        before_image_ids = list(candidate_artifact.metadata.get("edit_input_ids", []))
        if not before_image_ids:
            before_image_ids = list(workspace.current_image_ids)
        result = score_edit_result(
            before_images=[workspace.get_image(artifact_id) for artifact_id in before_image_ids],
            after_image=workspace.get_image(candidate_id),
            goal=workspace.current_prompt(),
        )
        if result is None:
            return ToolResult(
                status="skipped" if step.params.get("optional", True) else "failed",
                summary="Image scoring unavailable.",
                retryable=False,
            )

        record_id = workspace.add_record(
            result,
            label="image_score",
            step_id=step.step_id,
            parents=before_image_ids + [candidate_id, workspace.current_prompt_id],
        )
        return ToolResult(
            status="success",
            summary="Scored the current edit candidate.",
            produced_record_ids=[record_id],
            raw_output=result,
            retryable=False,
        )


@dataclasses.dataclass
class FireRedEditConfig:
    """Configuration for the FireRed image-edit tool."""

    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: str = " "
    seed: int = 49
    generator_device: str = "auto"


class FireRedEditTool:
    """Adapter that uses the FireRed diffusion pipeline as a tool."""

    name = "image_edit"

    def __init__(
        self,
        pipeline: Any,
        config: FireRedEditConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or FireRedEditConfig()

    def run(self, workspace: Workspace, step: PlanStep, memory: TaskMemory) -> ToolResult:
        if torch is None:
            return ToolResult(
                status="failed",
                summary="torch is required for the FireRed edit tool.",
                retryable=False,
            )
        image_ids = list(workspace.current_image_ids)
        prompt = workspace.current_prompt()
        attempt_index = int(step.params.get("attempt_index", 0))

        generator_device = self.config.generator_device
        if generator_device == "auto":
            generator_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        seed = self.config.seed + attempt_index
        with torch.inference_mode():
            result = self.pipeline(
                image=[workspace.get_image(artifact_id) for artifact_id in image_ids],
                prompt=prompt,
                generator=torch.Generator(device=generator_device).manual_seed(seed),
                true_cfg_scale=self.config.true_cfg_scale,
                guidance_scale=self.config.guidance_scale,
                negative_prompt=self.config.negative_prompt,
                num_inference_steps=self.config.num_inference_steps,
                num_images_per_prompt=1,
            )

        image = result.images[0]
        candidate_id = workspace.add_image(
            image,
            label=f"edit_candidate_{attempt_index}",
            step_id=step.step_id,
            parents=image_ids + [workspace.current_prompt_id],
            metadata={
                "edit_input_ids": list(image_ids),
                "prompt_id": workspace.current_prompt_id,
                "attempt_index": attempt_index,
                "seed": seed,
            },
        )
        workspace.set_active_candidate(candidate_id)
        return ToolResult(
            status="success",
            summary="FireRed produced a new edited candidate image.",
            produced_artifact_ids=[candidate_id],
            raw_output={"candidate_id": candidate_id, "seed": seed},
        )


def build_tool_registry(edit_tool: AgentTool | None = None) -> dict[str, AgentTool]:
    """Return the default tool registry for the comprehensive runtime."""
    registry: dict[str, AgentTool] = {
        "detect_or_segment": DetectOrSegmentTool(),
        "crop": CropTool(),
        "stitch": StitchTool(),
        "prompt_rewrite": PromptRewriteTool(),
        "image_understanding": ImageUnderstandingTool(),
        "ocr": OCRTool(),
        "image_scoring": ImageScoringTool(),
    }
    if edit_tool is not None:
        registry["image_edit"] = edit_tool
    return registry


__all__ = [
    "AgentTool",
    "FireRedEditConfig",
    "FireRedEditTool",
    "build_tool_registry",
]
