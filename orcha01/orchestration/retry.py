from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np


# ============================================================
# Contracts
# ============================================================

class RetryDecision(BaseModel):
    retry: bool
    escalate: bool = False
    boost_diversity: bool = False
    selected_experts: List[str] = Field(default_factory=list)
    reason: str = ""


class RetryInput(BaseModel):
    task_id: str
    iteration: int
    max_iterations: int
    previous_experts: List[str]
    scores: List[float]
    threshold: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Industrial Retry Engine
# ============================================================

class RetryEngine:
    """
    High-level intelligence that determines whether
    the system should attempt another pass.

    Think:
    - quality gate
    - diversity gate
    - budget protection
    """

    def __init__(self):
        pass

    # --------------------------------------------------------

    def decide(self, data: RetryInput) -> RetryDecision:
        logger.info(
            f"[Retry] iteration {data.iteration}/{data.max_iterations} for {data.task_id}"
        )

        if data.iteration >= data.max_iterations:
            logger.info("[Retry] max iterations reached")
            return RetryDecision(
                retry=False,
                reason="max_iterations_reached",
            )

        if not data.scores:
            logger.info("[Retry] no scores → retry")
            return RetryDecision(
                retry=True,
                escalate=True,
                boost_diversity=True,
                selected_experts=data.previous_experts,
                reason="no_scores",
            )

        best = max(data.scores)
        mean = float(np.mean(data.scores))

        logger.info(f"[Retry] best={best:.3f}, mean={mean:.3f}")

        # ----------------------------------------------------
        # If quality is acceptable → stop
        # ----------------------------------------------------
        if best >= data.threshold:
            return RetryDecision(
                retry=False,
                reason="quality_sufficient",
            )

        # ----------------------------------------------------
        # If very low → escalate pool
        # ----------------------------------------------------
        if best < data.threshold * 0.5:
            logger.warning("[Retry] severe failure → escalate experts")

            return RetryDecision(
                retry=True,
                escalate=True,
                boost_diversity=True,
                selected_experts=data.previous_experts,
                reason="severe_failure",
            )

        # ----------------------------------------------------
        # Medium → keep good ones, drop worst
        # ----------------------------------------------------
        idx = np.argsort(data.scores)[::-1]

        keep_n = max(1, len(idx) // 2)
        selected = [data.previous_experts[i] for i in idx[:keep_n]]

        logger.info(
            f"[Retry] refining with {len(selected)} experts (top half)"
        )

        return RetryDecision(
            retry=True,
            escalate=False,
            boost_diversity=False,
            selected_experts=selected,
            reason="refinement_retry",
        )


# ============================================================
# Factory
# ============================================================

_retry_instance: Optional[RetryEngine] = None


def get_retry_engine() -> RetryEngine:
    global _retry_instance
    if _retry_instance is None:
        _retry_instance = RetryEngine()
    return _retry_instance

# ============================================================
# Factory
# ============================================================

from typing import Optional


def get_retry_controller(config: Optional[dict] = None):
    """
    Default retry strategy.

    Future upgrades:
    - adaptive retry based on failure type
    - cost-aware retries
    - confidence-triggered re-execution
    - expert replacement
    """
    return RetryController(config=config)
