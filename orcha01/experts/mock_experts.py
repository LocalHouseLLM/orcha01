from __future__ import annotations

import random
import asyncio
import time
from typing import Dict

from .base import BaseExpert, ExpertOutput


# ============================================================
# Mock Experts
# ============================================================

class FastButNoisyFinanceExpert(BaseExpert):
    name = "finance_fast_noisy"
    domain = "finance"
    version = "1.0"
    cost_per_1k_tokens = 0.01

    async def execute(self, query: str) -> ExpertOutput:
        start = time.perf_counter()

        await asyncio.sleep(0.5)

        answer = f"Quick estimate for: {query}"
        tokens = max(1, len(answer) // 4)
        latency = time.perf_counter() - start

        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.4, 0.7),
            tokens_used=tokens,
            latency=latency,
            cost=self.estimate_cost(tokens),
        )


# ------------------------------------------------------------

class SlowAccurateFinanceExpert(BaseExpert):
    name = "finance_slow_accurate"
    domain = "finance"
    version = "1.0"
    cost_per_1k_tokens = 0.03

    async def execute(self, query: str) -> ExpertOutput:
        start = time.perf_counter()

        await asyncio.sleep(2)

        answer = f"Detailed financial reasoning for: {query}"
        tokens = max(1, len(answer) // 4)
        latency = time.perf_counter() - start

        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.75, 0.95),
            tokens_used=tokens,
            latency=latency,
            cost=self.estimate_cost(tokens),
        )


# ------------------------------------------------------------

class GeneralReasoningExpert(BaseExpert):
    name = "general_reasoner"
    domain = "general"
    version = "1.0"
    cost_per_1k_tokens = 0.02

    async def execute(self, query: str) -> ExpertOutput:
        start = time.perf_counter()

        await asyncio.sleep(1)

        answer = f"Logical breakdown of: {query}"
        tokens = max(1, len(answer) // 4)
        latency = time.perf_counter() - start

        return ExpertOutput(
            answer=answer,
            confidence=random.uniform(0.6, 0.85),
            tokens_used=tokens,
            latency=latency,
            cost=self.estimate_cost(tokens),
        )


# ============================================================
# Registry Helper
# ============================================================

def load_mock_experts() -> Dict[str, BaseExpert]:
    experts = [
        FastButNoisyFinanceExpert(),
        SlowAccurateFinanceExpert(),
        GeneralReasoningExpert(),
    ]
    return {e.name: e for e in experts}
