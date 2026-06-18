"""
orcha.orchestration.retry
==========================
Decides whether the orchestration loop should run another iteration,
and prepares instructions for the next one if it does.

Decision logic
--------------
STOP if any of:
  - evaluation passed (quality_score >= threshold)
  - budget is exhausted (cost, latency, or iteration cap hit)

RETRY otherwise, and prepare a richer context for the next iteration:
  - Mark failed experts for exclusion so the selector tries fresh models
  - Recommend a wider parallel_width for the next iteration
  - Record why the retry was triggered for observability

Retry enrichment
----------------
When retrying, the controller writes into the packet payload:
  - excluded_experts: names of experts that failed or produced low-quality
    output this iteration (selector will avoid them next time)
  - retry_reason: why we're retrying (for trace/debugging)
  - retry_count: how many times we've retried so far

This information is carried forward in packet.fork() automatically since
fork() merges the full payload, so the planner and selector see it on the
next iteration without any explicit coordination.

Quality-based expert exclusion
--------------------------------
Experts are excluded on retry if:
  a) They explicitly failed (success=False), OR
  b) Their individual confidence score was below EXCLUSION_CONF_THRESHOLD
     AND at least one other expert scored higher.
  c) The synthesizer is never excluded (it runs separately from scoring).
"""
from __future__ import annotations

import time
from typing import List, Optional, Set

from ..core.packets import OrchaPacket, PacketKind, ExpertResult, RetryReason


EXCLUSION_CONF_THRESHOLD = 0.45   # exclude experts below this on retry
MIN_GOOD_CONF            = 0.60   # at least one expert must beat this
                                  # before we start excluding low-scorers


class RetryController:
    """
    Post-evaluation decision gate. Prepares context for the next iteration
    when retrying.
    """

    def __init__(self, synthesizer_expert: Optional[str] = None):
        self.synthesizer_expert = synthesizer_expert

    def decide(self, packet: OrchaPacket) -> OrchaPacket:
        t0         = time.perf_counter()
        passed     = bool(packet.payload.get("passed", False))
        budget     = packet.budget
        iteration  = budget.iterations
        results    = packet.get_results()

        # ── Stop conditions ───────────────────────────────────────────
        if passed:
            return self._stop(packet, RetryReason.PASSED, iteration, t0)

        if budget.exhausted:
            return self._stop(packet, RetryReason.BUDGET_EXHAUSTED, iteration, t0)

        # ── Retry ─────────────────────────────────────────────────────
        reason = self._diagnose(packet, results)
        excluded = self._pick_exclusions(results)
        retry_count = int(packet.payload.get("retry_count", 0)) + 1

        duration_ms = (time.perf_counter() - t0) * 1000
        return packet.fork(
            PacketKind.RETRY,
            retry=True,
            retry_reason=reason.value,
            retry_count=retry_count,
            excluded_experts=list(excluded),
        ).stamp(
            "retry_decision", duration_ms,
            retry=True,
            reason=reason.value,
            retry_count=retry_count,
            excluded=list(excluded),
            iteration=iteration,
            budget_remaining=round(budget.remaining_cost, 4),
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _stop(
        self, packet: OrchaPacket, reason: RetryReason, iteration: int, t0: float
    ) -> OrchaPacket:
        duration_ms = (time.perf_counter() - t0) * 1000
        return packet.fork(
            PacketKind.RETRY,
            retry=False,
            retry_reason=reason.value,
        ).stamp(
            "retry_decision", duration_ms,
            retry=False,
            reason=reason.value,
            iteration=iteration,
        )

    def _diagnose(
        self, packet: OrchaPacket, results: List[ExpertResult]
    ) -> RetryReason:
        """Identify the primary reason quality was insufficient."""
        successful = [r for r in results if r.success]
        if not successful:
            return RetryReason.ALL_EXPERTS_FAILED
        answer = packet.payload.get("answer", "")
        if not answer or not answer.strip():
            return RetryReason.EMPTY_ANSWER
        return RetryReason.LOW_CONFIDENCE

    def _pick_exclusions(self, results: List[ExpertResult]) -> Set[str]:
        """
        Choose which experts to exclude from the next iteration.
        Never excludes the synthesizer (it runs outside the main pool).
        """
        excluded: Set[str] = set()
        successful = [r for r in results if r.success and r.output.strip()]

        # Always exclude explicit failures
        for r in results:
            if not r.success:
                excluded.add(r.name)

        # Exclude low-confidence experts if at least one did well
        if successful:
            best_conf = max(r.confidence for r in successful)
            if best_conf >= MIN_GOOD_CONF:
                for r in successful:
                    if r.confidence < EXCLUSION_CONF_THRESHOLD:
                        excluded.add(r.name)

        # Never exclude the synthesizer
        if self.synthesizer_expert:
            excluded.discard(self.synthesizer_expert)

        return excluded


def get_retry_controller(
    synthesizer_expert: Optional[str] = None,
) -> RetryController:
    return RetryController(synthesizer_expert=synthesizer_expert)
