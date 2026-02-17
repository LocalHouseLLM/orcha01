from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import random
import math

from loguru import logger


# ============================================================
# Contracts
# ============================================================

class SelectionMode(str, Enum):
    BEST = "best"              # pick highest score
    TOP_K = "top_k"            # deterministic top k
    STOCHASTIC = "stochastic"  # weighted random
    ALL = "all"                # everything matching


@dataclass
class ExpertCapability:
    """
    Describes what an expert can do.
    """
    domains: List[str]
    skills: List[str] = field(default_factory=list)
    priority: float = 1.0  # static weight


@dataclass
class ExpertStats:
    """
    Runtime statistics updated by evaluator.
    """
    success_rate: float = 1.0
    avg_latency: float = 1.0
    avg_tokens: float = 1.0
    health: float = 1.0  # failures reduce this


@dataclass
class ExpertInfo:
    name: str
    capability: ExpertCapability
    stats: ExpertStats = field(default_factory=ExpertStats)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionRequest:
    """
    Comes from planner / decomposer.
    """
    domain: str
    required_skills: List[str] = field(default_factory=list)
    max_experts: int = 3
    budget_tokens: Optional[int] = None
    budget_latency: Optional[float] = None
    mode: SelectionMode = SelectionMode.TOP_K


@dataclass
class SelectedExpert:
    name: str
    score: float
    info: ExpertInfo


# ============================================================
# Selector Engine
# ============================================================

class ExpertSelector:
    """
    Responsible for choosing WHICH experts should run.

    Combines:
    - domain match
    - skill match
    - historical performance
    - health
    - cost
    """

    def __init__(self):
        self.registry: Dict[str, ExpertInfo] = {}

    # --------------------------------------------------------

    def register(self, expert: ExpertInfo):
        logger.info(f"Register expert: {expert.name}")
        self.registry[expert.name] = expert

    # --------------------------------------------------------

    def list_experts(self) -> List[str]:
        return list(self.registry.keys())

    # --------------------------------------------------------

    def select(self, req: SelectionRequest) -> List[SelectedExpert]:
        logger.info(
            f"Selecting experts | domain={req.domain} "
            f"skills={req.required_skills} "
            f"mode={req.mode}"
        )

        candidates = self._filter_candidates(req)

        if not candidates:
            logger.warning("No experts matched request")
            return []

        scored = [(self._score(e, req), e) for e in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        if req.mode == SelectionMode.ALL:
            return [SelectedExpert(e.name, s, e) for s, e in scored]

        if req.mode == SelectionMode.BEST:
            s, e = scored[0]
            return [SelectedExpert(e.name, s, e)]

        if req.mode == SelectionMode.TOP_K:
            return [
                SelectedExpert(e.name, s, e)
                for s, e in scored[: req.max_experts]
            ]

        if req.mode == SelectionMode.STOCHASTIC:
            return self._stochastic_pick(scored, req.max_experts)

        return []

    # --------------------------------------------------------

    def _filter_candidates(self, req: SelectionRequest) -> List[ExpertInfo]:
        out = []

        for expert in self.registry.values():
            if req.domain not in expert.capability.domains:
                continue

            if req.required_skills:
                if not set(req.required_skills).intersection(
                    expert.capability.skills
                ):
                    continue

            out.append(expert)

        return out

    # --------------------------------------------------------

    def _score(self, expert: ExpertInfo, req: SelectionRequest) -> float:
        """
        Weighted score.

        Later you can plug PR, evaluator feedback, etc.
        """

        stats = expert.stats
        cap = expert.capability

        # performance
        perf = stats.success_rate * stats.health

        # speed penalty
        latency_penalty = 1 / (1 + stats.avg_latency)

        # token efficiency
        token_penalty = 1 / (1 + stats.avg_tokens)

        # static priority
        priority = cap.priority

        score = perf * latency_penalty * token_penalty * priority

        return float(score)

    # --------------------------------------------------------

    def _stochastic_pick(
        self,
        scored: List[Tuple[float, ExpertInfo]],
        k: int,
    ) -> List[SelectedExpert]:

        scores = [max(1e-6, s) for s, _ in scored]
        total = sum(scores)

        probs = [s / total for s in scores]

        chosen = set()
        result = []

        while len(result) < min(k, len(scored)):
            idx = random.choices(range(len(scored)), weights=probs, k=1)[0]

            if idx in chosen:
                continue

            chosen.add(idx)
            s, e = scored[idx]
            result.append(SelectedExpert(e.name, s, e))

        return result

    # --------------------------------------------------------

    def update_stats(
        self,
        expert_name: str,
        success: bool,
        latency: float,
        tokens: int,
    ):
        """
        Called by evaluator after execution.
        """

        if expert_name not in self.registry:
            return

        stats = self.registry[expert_name].stats

        alpha = 0.2  # smoothing factor

        stats.avg_latency = (
            alpha * latency + (1 - alpha) * stats.avg_latency
        )
        stats.avg_tokens = (
            alpha * tokens + (1 - alpha) * stats.avg_tokens
        )

        if success:
            stats.success_rate = min(1.0, stats.success_rate + 0.05)
        else:
            stats.success_rate = max(0.0, stats.success_rate - 0.1)
            stats.health = max(0.0, stats.health - 0.05)

        logger.debug(
            f"Updated stats for {expert_name}: "
            f"success={stats.success_rate:.2f}, "
            f"health={stats.health:.2f}"
        )
        
# ============================================================
# Factory
# ============================================================

from typing import Dict
from experts.base import BaseExpert


def get_selector(experts: Dict[str, BaseExpert]):
    """
    Default selector factory.

    Later you can replace this with:
    - RL selector
    - cost optimizer
    - latency aware selector
    - confidence predictor
    """
    return Selector(experts)
