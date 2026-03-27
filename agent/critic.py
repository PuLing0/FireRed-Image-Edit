"""Hybrid critic for the comprehensive FireRed agent runtime."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageChops, ImageStat

from agent.memory import TaskMemory
from agent.runtime_types import CriticResult, PlanStep, RepairAction, ToolResult, _make_id
from agent.tools import AgentTool
from agent.workspace import Workspace


def _mean_change_ratio(before: Image.Image, after: Image.Image) -> float:
    """Return the mean absolute pixel difference ratio in [0, 1]."""
    if before.mode != "RGB":
        before = before.convert("RGB")
    if after.mode != "RGB":
        after = after.convert("RGB")
    if before.size != after.size:
        after = after.resize(before.size)

    diff = ImageChops.difference(before, after)
    stat = ImageStat.Stat(diff)
    mean_channel_diff = sum(stat.mean) / max(len(stat.mean), 1)
    return float(mean_channel_diff) / 255.0


class HybridCritic:
    """First-pass critic combining rules with optional Gemini scoring."""

    def __init__(
        self,
        tools: dict[str, AgentTool],
        *,
        enable_vlm: bool = True,
    ) -> None:
        self.tools = tools
        self.enable_vlm = enable_vlm

    # ------------------------------------------------------------------
    # Step checks
    # ------------------------------------------------------------------

    def evaluate_step(
        self,
        *,
        workspace: Workspace,
        step: PlanStep,
        tool_result: ToolResult,
        memory: TaskMemory,
    ) -> CriticResult:
        """Validate one executed step."""
        if tool_result.status == "failed":
            reason = f"{step.tool_name} failed: {tool_result.summary}"
            return CriticResult(
                status="hard_fail",
                reasons=[reason],
                repair_actions=self._repair_actions_for_step(
                    step=step,
                    reasons=[reason],
                    workspace=workspace,
                    memory=memory,
                ),
            )

        if step.kind == "stitch_inputs" and len(workspace.current_image_ids) > 3:
            reason = "Stitch step still produced more than 3 composite images."
            return CriticResult(
                status="hard_fail",
                reasons=[reason],
                repair_actions=self._repair_actions_for_step(
                    step=step,
                    reasons=[reason],
                    workspace=workspace,
                    memory=memory,
                ),
            )

        if step.kind == "rewrite_prompt" and not workspace.current_prompt().strip():
            reason = "Prompt rewrite produced an empty prompt."
            return CriticResult(
                status="hard_fail",
                reasons=[reason],
                repair_actions=self._repair_actions_for_step(
                    step=step,
                    reasons=[reason],
                    workspace=workspace,
                    memory=memory,
                ),
            )

        if step.kind == "image_edit" and workspace.active_candidate_id is None:
            reason = "Image edit step did not register a candidate image."
            return CriticResult(
                status="hard_fail",
                reasons=[reason],
                repair_actions=self._repair_actions_for_step(
                    step=step,
                    reasons=[reason],
                    workspace=workspace,
                    memory=memory,
                ),
            )

        return CriticResult(status="pass", reasons=[f"{step.tool_name} passed structural checks."])

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------

    def evaluate_final(
        self,
        *,
        workspace: Workspace,
        memory: TaskMemory,
        goal: str,
    ) -> CriticResult:
        """Evaluate the final edited candidate."""
        candidate_id = workspace.active_candidate_id
        if candidate_id is None:
            return CriticResult(
                status="hard_fail",
                reasons=["No final candidate image was produced."],
                repair_actions=[RepairAction("retry_edit_seed", "Missing final candidate.")],
            )

        candidate = workspace.get_artifact(candidate_id)
        if candidate.kind != "image":
            return CriticResult(
                status="hard_fail",
                reasons=["Final candidate artifact is not an image."],
                repair_actions=[RepairAction("retry_edit_seed", "Candidate artifact type mismatch.")],
            )

        reasons: list[str] = []
        repair_actions: list[RepairAction] = []
        score: float | None = None

        before_ids = list(candidate.metadata.get("edit_input_ids", []))
        if before_ids:
            before_image = workspace.get_image(before_ids[0])
            change_ratio = _mean_change_ratio(before_image, workspace.get_image(candidate_id))
            if change_ratio < 0.01:
                reasons.append("The edited result is almost unchanged from the edit input.")
                repair_actions.extend(
                    self._repair_actions_for_failure(
                        reasons=reasons,
                        workspace=workspace,
                        memory=memory,
                    )
                )

        if self.enable_vlm and "image_scoring" in self.tools:
            scoring_step = PlanStep(
                step_id=_make_id("step"),
                kind="score_output",
                tool_name="image_scoring",
                purpose="Score the current final image.",
                params={"optional": True},
            )
            score_result = self.tools["image_scoring"].run(workspace, scoring_step, memory)
            if score_result.status == "success" and score_result.produced_record_ids:
                record = workspace.get_record(score_result.produced_record_ids[0])
                parsed_score = record.get("score")
                if isinstance(parsed_score, (int, float)):
                    score = float(parsed_score)
                verdict = str(record.get("verdict", "")).lower()
                verdict_reasons = list(record.get("reasons", []))
                if verdict == "hard_fail" or (score is not None and score <= 1.5):
                    reasons.extend(verdict_reasons)
                    repair_actions.extend(
                        self._repair_actions_for_failure(
                            reasons=reasons or ["VLM critic marked the result as hard fail."],
                            workspace=workspace,
                            memory=memory,
                        )
                    )
                    return CriticResult(
                        status="hard_fail",
                        reasons=reasons or ["VLM critic rejected the result."],
                        repair_actions=repair_actions,
                        score=score,
                        metadata={"score_result": score_result.raw_output},
                    )
                if verdict == "soft_fail" or (score is not None and score < 3.0):
                    reasons.extend(verdict_reasons)
                    repair_actions.extend(
                        self._repair_actions_for_failure(
                            reasons=reasons or ["VLM critic marked the result as soft fail."],
                            workspace=workspace,
                            memory=memory,
                        )
                    )

        if reasons:
            return CriticResult(
                status="soft_fail",
                reasons=reasons,
                repair_actions=repair_actions or self._repair_actions_for_failure(
                    reasons=reasons,
                    workspace=workspace,
                    memory=memory,
                ),
                score=score,
            )

        return CriticResult(
            status="pass",
            reasons=["Final candidate passed rule-based and optional VLM verification."],
            score=score,
            accepted_artifact_id=candidate_id,
        )

    # ------------------------------------------------------------------
    # Repair strategy
    # ------------------------------------------------------------------

    def _repair_actions_for_step(
        self,
        *,
        step: PlanStep,
        reasons: list[str],
        workspace: Workspace,
        memory: TaskMemory,
    ) -> list[RepairAction]:
        if step.kind in {"detect_regions", "crop_inputs"}:
            return [RepairAction("disable_roi", "; ".join(reasons))]
        if step.kind == "stitch_inputs":
            return [RepairAction("balanced_stitch", "; ".join(reasons))]
        if step.kind == "rewrite_prompt":
            return [RepairAction("force_prompt_rewrite", "; ".join(reasons))]
        return self._repair_actions_for_failure(
            reasons=reasons,
            workspace=workspace,
            memory=memory,
        )

    def _repair_actions_for_failure(
        self,
        *,
        reasons: list[str],
        workspace: Workspace,
        memory: TaskMemory,
    ) -> list[RepairAction]:
        joined_reasons = "; ".join(dict.fromkeys(reasons))

        if (
            len(workspace.input_image_ids) > 3
            and memory.failure_counts.get("disable_roi", 0) == 0
        ):
            return [RepairAction("disable_roi", joined_reasons)]

        if memory.failure_counts.get("force_prompt_rewrite", 0) == 0:
            return [
                RepairAction(
                    "force_prompt_rewrite",
                    joined_reasons,
                    params={"prompt_guidance": joined_reasons},
                )
            ]

        if memory.failure_counts.get("balanced_stitch", 0) == 0:
            return [RepairAction("balanced_stitch", joined_reasons)]

        return [RepairAction("retry_edit_seed", joined_reasons)]


__all__ = ["HybridCritic"]
