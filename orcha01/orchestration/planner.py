from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np


# ============================================================
# Contracts
# ============================================================

class PlannerInput(BaseModel):
    task_id: str
    query: str

    # budgets
    max_cost: float = 1.0
    max_latency: float = 30.0
    max_iterations: int = 3

    # signals from previous loops
    iteration: int = 0
    last_quality: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(BaseModel):
    parallel_width: int
    quality_threshold: float
    stop: bool = False
    escalation_level: int = 0
    budget_remaining: float = 1.0
    reason: str = ""


# ============================================================
# Industrial Planner
# ============================================================

class Planner:
    """
    Global control tower of orchestration.

    Responsible for:
    - how hard we try
    - how wide we search
    - when to stop
    - how much compute remains
    """

    def __init__(self):
        pass

    # --------------------------------------------------------

    def plan(self, data: PlannerInput) -> PlannerOutput:
        logger.info(
            f"[Planner] iteration {data.iteration}/{data.max_iterations} for {data.task_id}"
        )

        # ----------------------------------------------------
        # Stop conditions
        # ----------------------------------------------------
        if data.iteration >= data.max_iterations:
            logger.info("[Planner] stopping → max iterations")
            return PlannerOutput(
                parallel_width=0,
                quality_threshold=0.0,
                stop=True,
                escalation_level=data.iteration,
                budget_remaining=0.0,
                reason="max_iterations",
            )

        if data.last_quality is not None and data.last_quality >= 0.9:
            logger.info("[Planner] stopping → high confidence")
            return PlannerOutput(
                parallel_width=0,
                quality_threshold=0.0,
                stop=True,
                escalation_level=data.iteration,
                budget_remaining=data.max_cost,
                reason="quality_good",
            )

        # ----------------------------------------------------
        # Compute how aggressive we should be
        # ----------------------------------------------------

        progress = data.iteration / max(1, data.max_iterations)

        # increase effort as retries grow
        escalation_level = int(progress * 10)

        # more retries → more experts
        parallel_width = int(np.clip(3 + escalation_level, 3, 20))

        # dynamic threshold
        quality_threshold = float(np.clip(0.7 + progress * 0.2, 0.7, 0.9))

        # budget decay
        budget_remaining = float(
            max(0.0, data.max_cost * (1.0 - progress))
        )

        logger.info(
            f"[Planner] width={parallel_width}, threshold={quality_threshold:.2f}, "
            f"budget={budget_remaining:.2f}"
        )

        return PlannerOutput(
            parallel_width=parallel_width,
            quality_threshold=quality_threshold,
            stop=False,
            escalation_level=escalation_level,
            budget_remaining=budget_remaining,
            reason="continue",
        )


# ============================================================
# Factory
# ============================================================

_planner_instance: Optional[Planner] = None


def get_planner() -> Planner:
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = Planner()
    return _planner_instance
