"""
orcha.core.packets
==================
The foundational data model for Orcha. Every component in the pipeline
receives an OrchaPacket, reads from it, writes results back, stamps a
trace entry, and returns it. This makes the entire orchestration loop a
typed, inspectable, serialisable chain of transformations.

Design goals
------------
- One type flows everywhere: no ad-hoc dicts passed between stages.
- Full observability: every decision is recorded in the trace.
- Budget enforcement: cost, latency, and iteration limits are tracked
  centrally and checked before each stage.
- Forkable: child packets inherit parent state so branches never share
  mutable references.
- Serialisable: the whole packet round-trips through JSON for logging,
  caching, and replay.
"""
from __future__ import annotations

import copy
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────

class PacketKind(str, Enum):
    """Lifecycle state of an OrchaPacket as it moves through the pipeline."""
    QUERY       = "query"       # initial user request
    SUBTASKS    = "subtasks"    # after decomposition
    PLAN        = "plan"        # after planning
    SELECTION   = "selection"   # after expert selection
    EXECUTION   = "execution"   # after parallel model execution
    AGGREGATION = "aggregation" # after answer combination / synthesis
    EVALUATION  = "evaluation"  # after quality assessment
    RETRY       = "retry"       # after retry decision
    RESPONSE    = "response"    # final output
    ERROR       = "error"       # unrecoverable failure


class Domain(str, Enum):
    """Known expert domains used for routing."""
    GENERAL   = "general"
    REASONING = "reasoning"
    FINANCE   = "finance"
    CODE      = "code"
    SCIENCE   = "science"
    CREATIVE  = "creative"
    MATH      = "math"
    MEDICAL   = "medical"
    LEGAL     = "legal"
    UNKNOWN   = "unknown"


class AggregationMode(str, Enum):
    """How the aggregator combined expert outputs."""
    SYNTHESIS         = "synthesis"          # synthesizer model rewrote everything
    CONFIDENCE_WEIGHT = "confidence_weighted" # highest confidence answer wins
    VOTE              = "vote"               # majority or plurality answer
    SINGLE            = "single"             # only one expert answered
    EMPTY             = "empty"             # no expert answered


class RetryReason(str, Enum):
    """Why the retry controller decided to loop (or stop)."""
    PASSED            = "passed"
    BUDGET_EXHAUSTED  = "budget_exhausted"
    RETRYING          = "retrying"
    LOW_CONFIDENCE    = "low_confidence"
    EMPTY_ANSWER      = "empty_answer"
    ALL_EXPERTS_FAILED = "all_experts_failed"


# ── Sub-models ────────────────────────────────────────────────────────────────

class TraceStep(BaseModel):
    """One audit entry stamped by a pipeline stage."""
    stage:       str
    ts:          float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    data:        Dict[str, Any] = {}

    def summary(self) -> str:
        extras = "  ".join(f"{k}={v}" for k, v in self.data.items())
        return f"[{self.stage:>16}] {self.duration_ms:7.2f}ms  {extras}"


class BudgetState(BaseModel):
    """
    Tracks and enforces resource limits across the whole orchestration run.

    Any of the three limits being hit marks the budget as exhausted, which
    causes the planner to stop the loop regardless of answer quality.
    """
    # ── Limits ────────────────────────────────────────────────────────
    max_cost:       float = 1.0    # USD, summed across all expert calls
    max_latency_s:  float = 120.0  # wall-clock seconds (max-latency per iter)
    max_iterations: int   = 3      # hard cap on pipeline loop count

    # ── Consumed ──────────────────────────────────────────────────────
    cost_used:      float = 0.0
    latency_used_s: float = 0.0
    iterations:     int   = 0

    # ── Derived ───────────────────────────────────────────────────────
    @property
    def exhausted(self) -> bool:
        return (
            self.cost_used      >= self.max_cost
            or self.latency_used_s >= self.max_latency_s
            or self.iterations     >= self.max_iterations
        )

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.max_cost - self.cost_used)

    @property
    def remaining_latency_s(self) -> float:
        return max(0.0, self.max_latency_s - self.latency_used_s)

    @property
    def remaining_iterations(self) -> int:
        return max(0, self.max_iterations - self.iterations)

    def utilisation(self) -> Dict[str, float]:
        """Fraction of each budget dimension consumed (0–1)."""
        return {
            "cost":       self.cost_used / self.max_cost if self.max_cost else 0,
            "latency":    self.latency_used_s / self.max_latency_s if self.max_latency_s else 0,
            "iterations": self.iterations / self.max_iterations if self.max_iterations else 0,
        }

    def most_constrained_dimension(self) -> str:
        u = self.utilisation()
        return max(u, key=u.__getitem__)


class SubTask(BaseModel):
    """A structured sub-goal produced by the decomposer."""
    id:          str
    description: str
    domain:      str   = Domain.GENERAL
    priority:    float = Field(default=0.5, ge=0.0, le=1.0)
    metadata:    Dict[str, Any] = {}


class ExpertSlot(BaseModel):
    """A selected expert with routing metadata set by the selector."""
    name:        str
    domain:      str
    description: str
    score:       float = Field(ge=0.0, le=1.0)
    excluded:    bool  = False   # True if circuit-breaker tripped this expert
    metadata:    Dict[str, Any] = {}


class ExpertResult(BaseModel):
    """
    Output from a single expert execution, fully self-describing.
    A failed result carries an error string and zero confidence.
    """
    name:           str
    output:         str
    confidence:     float = Field(default=0.0, ge=0.0, le=1.0)
    tokens:         int   = 0
    latency_s:      float = 0.0
    cost:           float = 0.0
    success:        bool  = True
    error:          Optional[str] = None
    finish_reason:  Optional[str] = None   # "stop", "length", "timeout", …
    model_name:     Optional[str] = None   # underlying model identifier
    metadata:       Dict[str, Any] = {}

    @property
    def failed(self) -> bool:
        return not self.success or not self.output.strip()

    def quality_score(self) -> float:
        """
        Simple composite quality signal used by the aggregator for ranking.
        Combines confidence with a small bonus for longer, complete answers.
        """
        if self.failed:
            return 0.0
        length_bonus = min(0.05, len(self.output.split()) / 2000)
        finish_bonus = 0.05 if self.finish_reason in ("stop", None) else 0.0
        return min(1.0, self.confidence + length_bonus + finish_bonus)


# ── The packet ────────────────────────────────────────────────────────────────

class OrchaPacket(BaseModel):
    """
    The universal typed message that flows through every Orcha stage.

    Contract
    --------
    - Every stage has the signature:
        def stage(self, packet: OrchaPacket) -> OrchaPacket
      (async for executor and aggregator).

    - A stage MUST NOT mutate the incoming packet's payload directly —
      it calls fork() to produce a child, then stamps the child.

    - The payload dict accumulates data as the packet traverses the pipeline.
      Later stages can read data written by earlier ones (e.g. the aggregator
      reads `results` written by the executor).

    Payload keys by stage
    ---------------------
    decompose  → subtasks: list[SubTask], domains: list[str], complexity: float
    plan       → stop: bool, parallel_width: int, quality_threshold: float,
                 force_all_experts: bool, excluded_experts: list[str]
    select     → selected_experts: list[ExpertSlot]
    execute    → results: list[ExpertResult]
    aggregate  → answer: str, confidence: float, contributors: list[str],
                 primary: str, synthesized: bool, agg_mode: str
    evaluate   → passed: bool, confidence: float, quality_dimensions: dict
    retry      → retry: bool, reason: str
    """
    id:         str  = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id:  Optional[str] = None          # set when forked from another packet
    created_at: float = Field(default_factory=time.time)
    kind:       PacketKind
    query:      str
    payload:    Dict[str, Any]  = {}
    budget:     BudgetState     = Field(default_factory=BudgetState)
    trace:      List[TraceStep] = []
    metadata:   Dict[str, Any]  = {}
    tags:       List[str]       = []          # free-form labels for filtering

    model_config = {"arbitrary_types_allowed": True}

    # ── Mutation helpers ──────────────────────────────────────────────

    def stamp(self, stage: str, duration_ms: float = 0.0, **data: Any) -> "OrchaPacket":
        """Append a trace entry. Mutates in-place; returns self for chaining."""
        self.trace.append(TraceStep(stage=stage, duration_ms=duration_ms, data=data))
        return self

    def fork(self, kind: PacketKind, **payload_updates: Any) -> "OrchaPacket":
        """
        Produce a child packet that inherits this packet's query, budget,
        trace, metadata, and tags, then merges payload_updates on top of
        the existing payload.

        The budget AND the payload are deep-copied so the child and parent
        never share mutable state — list/dict payload values written by an
        earlier stage cannot be mutated in place by a later one. This makes
        the "stages MUST NOT mutate the incoming packet's payload" contract
        enforced rather than hoped.
        """
        new_payload = copy.deepcopy(self.payload)
        new_payload.update(payload_updates)
        return OrchaPacket(
            parent_id=self.id,
            kind=kind,
            query=self.query,
            payload=new_payload,
            budget=self.budget.model_copy(deep=True),
            trace=list(self.trace),
            metadata=dict(self.metadata),
            tags=list(self.tags),
        )

    def error_packet(self, message: str, **extra: Any) -> "OrchaPacket":
        """Convenience: produce an ERROR-kind packet with an error message."""
        return self.fork(PacketKind.ERROR, error=message, **extra)

    def tag(self, *labels: str) -> "OrchaPacket":
        """Add free-form labels. Mutates in-place; returns self."""
        self.tags.extend(labels)
        return self

    # ── Read helpers ──────────────────────────────────────────────────

    def get_results(self) -> List[ExpertResult]:
        """Parse the results list from payload into typed ExpertResult objects."""
        raw = self.payload.get("results", [])
        return [ExpertResult(**r) if isinstance(r, dict) else r for r in raw]

    def get_subtasks(self) -> List[SubTask]:
        raw = self.payload.get("subtasks", [])
        return [SubTask(**t) if isinstance(t, dict) else t for t in raw]

    def get_selected_experts(self) -> List[ExpertSlot]:
        raw = self.payload.get("selected_experts", [])
        return [ExpertSlot(**e) if isinstance(e, dict) else e for e in raw]

    def successful_results(self) -> List[ExpertResult]:
        return [r for r in self.get_results() if not r.failed]

    # ── Serialisation ─────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "OrchaPacket":
        return cls.model_validate_json(text)

    # ── Introspection ─────────────────────────────────────────────────

    def age_s(self) -> float:
        return time.time() - self.created_at

    def stage_duration(self, stage: str) -> Optional[float]:
        """Return the duration_ms for the first trace entry with the given stage."""
        for step in self.trace:
            if step.stage == stage:
                return step.duration_ms
        return None

    def explain(self) -> str:
        """Human-readable pipeline trace."""
        lines = [f"Packet {self.id[:8]}  kind={self.kind}  age={self.age_s():.2f}s"]
        lines.append(f"Query: {self.query[:120]}")
        lines.append("")
        for step in self.trace:
            lines.append(step.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OrchaPacket(id={self.id[:8]}, kind={self.kind}, "
            f"query={self.query[:40]!r})"
        )
