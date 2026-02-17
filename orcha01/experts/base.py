from __future__ import annotations

import abc
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================
# Expert Output Contract
# ============================================================

class ExpertOutput(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_used: int = 0
    latency: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Base Expert
# ============================================================

class BaseExpert(abc.ABC):
    """
    Base interface every expert must implement.

    Guarantees:
    - async execution
    - confidence reporting
    - telemetry support
    """

    name: str = "unknown"
    domain: str = "general"
    version: str = "0.0"
    cost_per_1k_tokens: float = 0.0

    # --------------------------------------------------------

    async def run(self, query: str) -> str:
        """
        Compatibility wrapper for executor.
        """
        result = await self.execute(query)
        return result.answer

    # --------------------------------------------------------

    @abc.abstractmethod
    async def execute(self, query: str) -> ExpertOutput:
        """
        Implement model logic here.
        """
        pass

    # --------------------------------------------------------

    def estimate_cost(self, tokens: int) -> float:
        return (tokens / 1000.0) * self.cost_per_1k_tokens

    # --------------------------------------------------------

    async def healthcheck(self) -> bool:
        """
        Override if expert supports advanced health probing.
        """
        return True
