"""Task-local memory for the comprehensive FireRed agent runtime."""

from __future__ import annotations

import dataclasses
from typing import Any

from agent.runtime_types import ExecutionPlan, RuntimeEvent


@dataclasses.dataclass
class TaskMemory:
    """Execution trace and failure history for one runtime invocation."""

    plans: list[ExecutionPlan] = dataclasses.field(default_factory=list)
    events: list[RuntimeEvent] = dataclasses.field(default_factory=list)
    failure_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    repair_round: int = 0

    def add_plan(self, plan: ExecutionPlan) -> None:
        """Record a generated plan."""
        self.plans.append(plan)
        self.add_event(
            "plan",
            f"Registered plan {plan.plan_id} with {len(plan.steps)} step(s).",
            {"plan_id": plan.plan_id, "repair_round": plan.repair_round},
        )

    def add_event(
        self,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        """Append an execution event."""
        event = RuntimeEvent.create(kind=kind, message=message, payload=payload)
        self.events.append(event)
        return event

    def record_failure(self, failure_key: str) -> None:
        """Increment a named failure bucket."""
        self.failure_counts[failure_key] = self.failure_counts.get(failure_key, 0) + 1

    def latest_event(self, kind: str | None = None) -> RuntimeEvent | None:
        """Return the latest event, optionally filtered by kind."""
        events = self.events if kind is None else [
            event for event in self.events if event.kind == kind
        ]
        return events[-1] if events else None

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable memory summary."""
        return {
            "repair_round": self.repair_round,
            "failure_counts": dict(self.failure_counts),
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "goal": plan.goal,
                    "repair_round": plan.repair_round,
                    "steps": [dataclasses.asdict(step) for step in plan.steps],
                    "success_checks": list(plan.success_checks),
                    "budget": dict(plan.budget),
                    "metadata": dict(plan.metadata),
                }
                for plan in self.plans
            ],
            "events": [
                dataclasses.asdict(event)
                for event in self.events
            ],
        }


__all__ = ["TaskMemory"]
