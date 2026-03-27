"""Workspace for the comprehensive FireRed agent runtime."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable

from PIL import Image

from agent.runtime_types import Artifact, _make_id


@dataclasses.dataclass
class WorkspaceCheckpoint:
    """Named pointer snapshot used for repair and rollback."""

    checkpoint_id: str
    label: str
    current_image_ids: list[str]
    current_prompt_id: str
    active_candidate_id: str | None
    group_indices: list[list[int]]
    rois: list[dict[str, Any]]


class Workspace:
    """In-memory state store for one agent run."""

    def __init__(self) -> None:
        self.artifacts: dict[str, Artifact] = {}
        self.checkpoints: list[WorkspaceCheckpoint] = []
        self.input_image_ids: list[str] = []
        self.current_image_ids: list[str] = []
        self.current_prompt_id: str = ""
        self.group_indices: list[list[int]] = []
        self.rois: list[dict[str, Any]] = []
        self.active_candidate_id: str | None = None

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------

    def add_image(
        self,
        image: Image.Image,
        *,
        label: str,
        step_id: str,
        parents: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register an image artifact and return its identifier."""
        artifact_id = _make_id("img")
        self.artifacts[artifact_id] = Artifact(
            artifact_id=artifact_id,
            kind="image",
            data=image,
            label=label,
            step_id=step_id,
            parents=list(parents or []),
            metadata=metadata or {},
        )
        return artifact_id

    def add_text(
        self,
        text: str,
        *,
        label: str,
        step_id: str,
        parents: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a text artifact and return its identifier."""
        artifact_id = _make_id("txt")
        self.artifacts[artifact_id] = Artifact(
            artifact_id=artifact_id,
            kind="text",
            data=text,
            label=label,
            step_id=step_id,
            parents=list(parents or []),
            metadata=metadata or {},
        )
        return artifact_id

    def add_record(
        self,
        record: dict[str, Any],
        *,
        label: str,
        step_id: str,
        parents: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a structured record and return its identifier."""
        artifact_id = _make_id("rec")
        self.artifacts[artifact_id] = Artifact(
            artifact_id=artifact_id,
            kind="record",
            data=record,
            label=label,
            step_id=step_id,
            parents=list(parents or []),
            metadata=metadata or {},
        )
        return artifact_id

    # ------------------------------------------------------------------
    # Typed getters
    # ------------------------------------------------------------------

    def get_artifact(self, artifact_id: str) -> Artifact:
        """Return an artifact by id."""
        return self.artifacts[artifact_id]

    def get_image(self, artifact_id: str) -> Image.Image:
        """Return an image artifact."""
        artifact = self.get_artifact(artifact_id)
        if artifact.kind != "image":
            raise TypeError(f"Artifact {artifact_id} is not an image.")
        return artifact.data

    def get_text(self, artifact_id: str) -> str:
        """Return a text artifact."""
        artifact = self.get_artifact(artifact_id)
        if artifact.kind != "text":
            raise TypeError(f"Artifact {artifact_id} is not text.")
        return artifact.data

    def get_record(self, artifact_id: str) -> dict[str, Any]:
        """Return a record artifact."""
        artifact = self.get_artifact(artifact_id)
        if artifact.kind != "record":
            raise TypeError(f"Artifact {artifact_id} is not a record.")
        return artifact.data

    def latest_by_label(
        self,
        label: str,
        *,
        kind: str | None = None,
    ) -> Artifact | None:
        """Return the latest artifact matching a label."""
        matches = [
            artifact
            for artifact in self.artifacts.values()
            if artifact.label == label and (kind is None or artifact.kind == kind)
        ]
        return matches[-1] if matches else None

    # ------------------------------------------------------------------
    # State initialisation / mutation
    # ------------------------------------------------------------------

    def seed_inputs(self, images: list[Image.Image], prompt: str) -> None:
        """Seed the workspace with original images and prompt."""
        self.input_image_ids = [
            self.add_image(
                image,
                label=f"input_image_{idx}",
                step_id="ingest",
                metadata={"input_index": idx, "role": "input"},
            )
            for idx, image in enumerate(images)
        ]
        self.current_image_ids = list(self.input_image_ids)
        self.current_prompt_id = self.add_text(
            prompt,
            label="current_prompt",
            step_id="ingest",
            metadata={"source": "user"},
        )
        self.group_indices = [[idx] for idx in range(len(images))]
        self.rois = [
            {"image_index": idx, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
            for idx in range(len(images))
        ]
        self.active_candidate_id = None

    def set_current_images(self, artifact_ids: list[str]) -> None:
        """Update the currently active image list."""
        self.current_image_ids = list(artifact_ids)

    def set_current_prompt(self, artifact_id: str) -> None:
        """Update the active prompt artifact pointer."""
        self.current_prompt_id = artifact_id

    def set_group_indices(self, group_indices: list[list[int]]) -> None:
        """Update the active image grouping."""
        self.group_indices = [list(group) for group in group_indices]

    def set_rois(self, rois: list[dict[str, Any]]) -> None:
        """Update the active ROI list."""
        self.rois = [dict(roi) for roi in rois]

    def set_active_candidate(self, artifact_id: str | None) -> None:
        """Update the current accepted edit candidate."""
        self.active_candidate_id = artifact_id

    def current_images(self) -> list[Image.Image]:
        """Return the active image list."""
        return [self.get_image(artifact_id) for artifact_id in self.current_image_ids]

    def current_prompt(self) -> str:
        """Return the active prompt text."""
        return self.get_text(self.current_prompt_id)

    # ------------------------------------------------------------------
    # Repair / rollback
    # ------------------------------------------------------------------

    def create_checkpoint(self, label: str) -> WorkspaceCheckpoint:
        """Capture a pointer snapshot."""
        checkpoint = WorkspaceCheckpoint(
            checkpoint_id=_make_id("ckpt"),
            label=label,
            current_image_ids=list(self.current_image_ids),
            current_prompt_id=self.current_prompt_id,
            active_candidate_id=self.active_candidate_id,
            group_indices=[list(group) for group in self.group_indices],
            rois=[dict(roi) for roi in self.rois],
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> WorkspaceCheckpoint:
        """Restore a previous pointer snapshot."""
        checkpoint = next(
            item for item in reversed(self.checkpoints)
            if item.checkpoint_id == checkpoint_id
        )
        self.current_image_ids = list(checkpoint.current_image_ids)
        self.current_prompt_id = checkpoint.current_prompt_id
        self.active_candidate_id = checkpoint.active_candidate_id
        self.group_indices = [list(group) for group in checkpoint.group_indices]
        self.rois = [dict(roi) for roi in checkpoint.rois]
        return checkpoint

    # ------------------------------------------------------------------
    # Debug export
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable summary of the workspace."""
        return {
            "input_image_ids": list(self.input_image_ids),
            "current_image_ids": list(self.current_image_ids),
            "current_prompt_id": self.current_prompt_id,
            "active_candidate_id": self.active_candidate_id,
            "group_indices": [list(group) for group in self.group_indices],
            "rois": [dict(roi) for roi in self.rois],
            "artifacts": {
                artifact_id: artifact.summary()
                for artifact_id, artifact in self.artifacts.items()
            },
            "checkpoints": [
                dataclasses.asdict(checkpoint) for checkpoint in self.checkpoints
            ],
        }


__all__ = ["Workspace", "WorkspaceCheckpoint"]
