from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Awaitable, Optional, List
import logging


logger = logging.getLogger("anvira.executor")


# ============================================================
# Contracts
# ============================================================

@dataclass
class ExecutionRequest:
    expert_name: str
    coro: Callable[[], Awaitable[str]]
    timeout: float = 60.0
    max_retries: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    expert_name: str
    output: str
    latency: float
    tokens: int
    success: bool
    error: Optional[str] = None
    retries_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Executor
# ============================================================

class Executor:
    """
    High-reliability async runtime for model experts.

    Guarantees:
    - Isolation between experts
    - Timeout enforcement
    - Retry policy
    - Structured errors
    """

    def __init__(self, global_timeout: Optional[float] = None):
        self.global_timeout = global_timeout

    # --------------------------------------------------------

    async def run_batch(
        self,
        requests: List[ExecutionRequest],
    ) -> List[ExecutionResult]:

        logger.info(f"Launching batch with {len(requests)} experts")

        tasks = [self._run_with_guard(r) for r in requests]

        if self.global_timeout:
            try:
                return await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=self.global_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("GLOBAL EXECUTION TIMEOUT")
                # cancel everything
                for t in tasks:
                    if not t.done():
                        t.cancel()
                raise
        else:
            return await asyncio.gather(*tasks)

    # --------------------------------------------------------

    async def _run_with_guard(self, request: ExecutionRequest) -> ExecutionResult:
        retries = 0

        while True:
            try:
                start = time.perf_counter()

                output = await asyncio.wait_for(
                    request.coro(),
                    timeout=request.timeout,
                )

                latency = time.perf_counter() - start
                tokens = self._estimate_tokens(output)

                return ExecutionResult(
                    expert_name=request.expert_name,
                    output=output,
                    latency=latency,
                    tokens=tokens,
                    success=True,
                    retries_used=retries,
                    metadata=request.metadata,
                )

            except asyncio.TimeoutError:
                err = f"timeout after {request.timeout}s"
                logger.warning(f"{request.expert_name}: {err}")

            except asyncio.CancelledError:
                err = "cancelled"
                logger.warning(f"{request.expert_name}: cancelled")
                return ExecutionResult(
                    expert_name=request.expert_name,
                    output="",
                    latency=0,
                    tokens=0,
                    success=False,
                    error=err,
                    retries_used=retries,
                    metadata=request.metadata,
                )

            except Exception:
                err = traceback.format_exc()
                logger.error(f"{request.expert_name} crashed:\n{err}")

            # Retry?
            if retries >= request.max_retries:
                return ExecutionResult(
                    expert_name=request.expert_name,
                    output="",
                    latency=0,
                    tokens=0,
                    success=False,
                    error=err,
                    retries_used=retries,
                    metadata=request.metadata,
                )

            retries += 1
            logger.info(f"{request.expert_name}: retry {retries}")

    # --------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """
        Cheap heuristic until tokenizer accounting is plugged in.
        Replace later with real tokenizer.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
