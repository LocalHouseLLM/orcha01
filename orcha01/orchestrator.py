from __future__ import annotations

import asyncio
import uuid
from typing import Dict, Any, List

from loguru import logger
from pydantic import BaseModel, Field

from orchestration.decomposer import get_decomposer
from orchestration.selector import get_selector
from orchestration.executor import Executor, ExecutionRequest
from orchestration.aggregator import get_aggregator
from orchestration.planner import get_planner, PlannerInput
from orchestration.evaluator import get_evaluator
from orchestration.retry import get_retry_controller


# ============================================================
# Contracts
# ============================================================

class OrchestratorRequest(BaseModel):
    query: str
    max_cost: float = 1.0
    max_latency: float = 60.0
    max_iterations: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorResponse(BaseModel):
    answer: str
    confidence: float
    iterations: int
    explainability: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Orchestrator
# ============================================================

class Orchestrator:
    """
    Main coordination runtime.

    Flow:
    Decompose → Plan → Select → Execute → Aggregate → Evaluate → Retry
    """

    def __init__(self):
        self.decomposer = get_decomposer()
        self.selector = get_selector()
        self.executor = Executor()
        self.aggregator = get_aggregator()
        self.planner = get_planner()
        self.evaluator = get_evaluator()
        self.retry_controller = get_retry_controller()

    # --------------------------------------------------------

    async def run(self, request: OrchestratorRequest) -> OrchestratorResponse:
        task_id = str(uuid.uuid4())

        logger.info(f"[Orchestrator] new task {task_id}")

        explain = {
            "task_id": task_id,
            "steps": [],
        }

        # ----------------------------------------------------
        # Step 1 — Decompose
        # ----------------------------------------------------
        subtasks = self.decomposer.decompose(request.query)

        explain["steps"].append({
            "stage": "decomposition",
            "subtasks": subtasks,
        })

        last_quality = None
        final_answer = ""
        confidence = 0.0

        # ----------------------------------------------------
        # Iterative orchestration loop
        # ----------------------------------------------------
        for iteration in range(request.max_iterations):

            logger.info(f"[Orchestrator] iteration {iteration}")

            # --------------------------------------------
            # Planning
            # --------------------------------------------
            plan = self.planner.plan(
                PlannerInput(
                    task_id=task_id,
                    query=request.query,
                    max_cost=request.max_cost,
                    max_latency=request.max_latency,
                    max_iterations=request.max_iterations,
                    iteration=iteration,
                    last_quality=last_quality,
                )
            )

            explain["steps"].append({
                "stage": "planning",
                "plan": plan.dict(),
            })

            if plan.stop:
                logger.info("[Orchestrator] planner requested stop")
                break

            # --------------------------------------------
            # Selection
            # --------------------------------------------
            selected = self.selector.select(
                query=request.query,
                width=plan.parallel_width,
                metadata=request.metadata,
            )

            explain["steps"].append({
                "stage": "selection",
                "experts": [e.name for e in selected],
            })

            # --------------------------------------------
            # Execution
            # --------------------------------------------
            exec_requests: List[ExecutionRequest] = []

            for expert in selected:
                exec_requests.append(
                    ExecutionRequest(
                        expert_name=expert.name,
                        coro=lambda e=expert: e.run(request.query),
                        max_retries=1,
                        metadata={"task_id": task_id},
                    )
                )

            results = await self.executor.run_batch(exec_requests)

            explain["steps"].append({
                "stage": "execution",
                "results": [r.dict() for r in results],
            })

            # --------------------------------------------
            # Aggregation
            # --------------------------------------------
            aggregated = self.aggregator.aggregate(results)

            final_answer = aggregated.answer

            explain["steps"].append({
                "stage": "aggregation",
                "details": aggregated.dict(),
            })

            # --------------------------------------------
            # Evaluation
            # --------------------------------------------
            evaluation = self.evaluator.evaluate(
                query=request.query,
                answer=final_answer,
                metadata=request.metadata,
            )

            confidence = evaluation.confidence
            last_quality = confidence

            explain["steps"].append({
                "stage": "evaluation",
                "details": evaluation.dict(),
            })

            # --------------------------------------------
            # Retry decision
            # --------------------------------------------
            retry = self.retry_controller.should_retry(
                confidence=confidence,
                threshold=plan.quality_threshold,
                iteration=iteration,
            )

            explain["steps"].append({
                "stage": "retry_decision",
                "retry": retry,
            })

            if not retry:
                break

        logger.info(f"[Orchestrator] completed with confidence {confidence:.2f}")

        return OrchestratorResponse(
            answer=final_answer,
            confidence=confidence,
            iterations=iteration + 1,
            explainability=explain,
        )


# ============================================================
# Factory
# ============================================================

_orchestrator_instance: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
