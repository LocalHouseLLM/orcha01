from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import statistics
import hashlib
import logging


# ============================================================
# Optional ML capabilities
# ============================================================

try:
    from sentence_transformers import SentenceTransformer, util

    _EMBED_AVAILABLE = True
except Exception:
    _EMBED_AVAILABLE = False


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("anvira.aggregator")


# ============================================================
# Data Contracts
# ============================================================

@dataclass
class ExpertResult:
    name: str
    output: str
    confidence: float
    tokens: int
    latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    abstained: bool = False


@dataclass
class AggregationConfig:
    confidence_weight: float = 0.5
    latency_weight: float = 0.15
    cost_weight: float = 0.15
    agreement_weight: float = 0.2

    minimum_confidence: float = 0.05
    tie_margin: float = 0.02

    enable_merging: bool = True

    # NEW
    use_semantic_clustering: bool = False
    semantic_model_name: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.82


@dataclass
class AggregationReport:
    final_answer: str
    strategy_used: str
    chosen_experts: List[str]
    scores: Dict[str, float]
    clusters: Dict[str, List[str]]
    trace: Dict[str, Any]


# ============================================================
# Utilities
# ============================================================

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


# ============================================================
# Aggregator
# ============================================================

class Aggregator:
    """
    Production ensemble decision engine.

    Deterministic.
    Pluggable.
    Extensible.
    """

    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
        self._validate()

        self.embedder = None
        if self.config.use_semantic_clustering and _EMBED_AVAILABLE:
            logger.info("Loading semantic model for aggregation...")
            self.embedder = SentenceTransformer(self.config.semantic_model_name)
        elif self.config.use_semantic_clustering:
            logger.warning("Semantic clustering requested but dependency missing.")

    # --------------------------------------------------------

    def _validate(self):
        total = (
            self.config.confidence_weight
            + self.config.latency_weight
            + self.config.cost_weight
            + self.config.agreement_weight
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("Aggregator weights must sum to 1.0")

    # --------------------------------------------------------

    def aggregate(self, results: List[ExpertResult]) -> AggregationReport:
        if not results:
            raise ValueError("No expert results supplied")

        active = [r for r in results if not r.abstained]

        if not active:
            return AggregationReport(
                final_answer="No expert produced an answer.",
                strategy_used="all_abstained",
                chosen_experts=[],
                scores={},
                clusters={},
                trace={"reason": "abstain"},
            )

        valid = [r for r in active if r.confidence >= self.config.minimum_confidence]
        if not valid:
            valid = active

        # ====================================================
        # CLUSTERING
        # ====================================================
        if self.embedder:
            clusters = self._semantic_cluster(valid)
            clustering_type = "semantic"
        else:
            clusters = self._hash_cluster(valid)
            clustering_type = "hash"

        # ====================================================
        # AGREEMENT
        # ====================================================
        agreement_scores = self._agreement_scores(clusters)

        # ====================================================
        # NORMALIZATION
        # ====================================================
        latency_scores = self._normalize_inverse([r.latency for r in valid])
        cost_scores = self._normalize_inverse([r.tokens for r in valid])

        # ====================================================
        # SCORING
        # ====================================================
        scores: Dict[str, float] = {}
        for i, r in enumerate(valid):
            score = (
                self.config.confidence_weight * r.confidence
                + self.config.latency_weight * latency_scores[i]
                + self.config.cost_weight * cost_scores[i]
                + self.config.agreement_weight * agreement_scores[r.name]
            )
            scores[r.name] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]

        winners = [
            name for name, s in ranked if abs(s - top_score) <= self.config.tie_margin
        ]

        # ====================================================
        # FINAL DECISION
        # ====================================================
        if len(winners) == 1 or not self.config.enable_merging:
            winner = next(r for r in valid if r.name == winners[0])
            final_answer = winner.output
            strategy = "single_winner"
        else:
            final_answer = self._merge([r for r in valid if r.name in winners])
            strategy = "merge"

        # ====================================================
        # TRACE
        # ====================================================
        trace = {
            "cluster_type": clustering_type,
            "num_total": len(results),
            "num_active": len(active),
            "num_valid": len(valid),
            "avg_confidence": statistics.mean([r.confidence for r in valid]),
            "winners": winners,
            "weights": vars(self.config),
        }

        return AggregationReport(
            final_answer=final_answer,
            strategy_used=strategy,
            chosen_experts=winners,
            scores=scores,
            clusters=clusters,
            trace=trace,
        )

    # ========================================================
    # Clustering
    # ========================================================

    def _hash_cluster(self, results: List[ExpertResult]) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for r in results:
            key = stable_hash(r.output)
            buckets.setdefault(key, []).append(r.name)
        return buckets

    # --------------------------------------------------------

    def _semantic_cluster(self, results: List[ExpertResult]) -> Dict[str, List[str]]:
        texts = [r.output for r in results]
        embeddings = self.embedder.encode(texts, convert_to_tensor=True)

        clusters: Dict[str, List[str]] = {}
        used = set()

        for i, r in enumerate(results):
            if i in used:
                continue

            cluster_key = f"group_{i}"
            clusters[cluster_key] = [r.name]
            used.add(i)

            for j in range(i + 1, len(results)):
                if j in used:
                    continue
                score = util.cos_sim(embeddings[i], embeddings[j]).item()
                if score >= self.config.semantic_threshold:
                    clusters[cluster_key].append(results[j].name)
                    used.add(j)

        return clusters

    # ========================================================
    # Agreement
    # ========================================================

    def _agreement_scores(self, clusters: Dict[str, List[str]]) -> Dict[str, float]:
        total = sum(len(v) for v in clusters.values())
        scores = {}
        for names in clusters.values():
            agreement = len(names) / total
            for n in names:
                scores[n] = agreement
        return scores

    # ========================================================
    # Merge
    # ========================================================

    def _merge(self, winners: List[ExpertResult]) -> str:
        outputs = sorted([w.output.strip() for w in winners])
        return "\n\n".join(outputs)

    # ========================================================
    # Normalize
    # ========================================================

    def _normalize_inverse(self, values: List[float]) -> List[float]:
        if len(values) == 1:
            return [1.0]

        max_v = max(values)
        min_v = min(values)

        if math.isclose(max_v, min_v):
            return [1.0 for _ in values]

        return [(max_v - v) / (max_v - min_v) for v in values]
