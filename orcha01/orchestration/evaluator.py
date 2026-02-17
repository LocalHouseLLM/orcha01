from __future__ import annotations

from typing import List, Dict, Any, Optional
import time

import numpy as np
from scipy.spatial.distance import cosine

from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from loguru import logger


# ============================================================
# Contracts
# ============================================================

class EvaluationInput(BaseModel):
    query: str
    task_id: str
    candidates: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    text: str
    relevance: float
    agreement: float
    final_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    ranked: List[CandidateScore]
    best: Optional[CandidateScore]
    needs_retry: bool
    evaluation_time: float


# ============================================================
# Industrial Evaluator
# ============================================================

class OutputEvaluator:
    """
    Scores outputs using:
        - relevance to query
        - cross-candidate agreement

    Designed to later plug into LLM judges,
    but embedding evaluation is MUCH faster.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        retry_threshold: float = 0.35,
    ):
        logger.info("Loading evaluator model…")

        self.model = SentenceTransformer(model_name, device=device)
        self.retry_threshold = retry_threshold

    # --------------------------------------------------------

    def _relevance_scores(
        self,
        query_emb: np.ndarray,
        outputs_emb: np.ndarray,
    ) -> List[float]:
        """
        Cosine similarity to original query.
        """
        sims = []

        for emb in outputs_emb:
            sims.append(1 - cosine(query_emb, emb))

        return sims

    # --------------------------------------------------------

    def _agreement_scores(self, outputs_emb: np.ndarray) -> List[float]:
        """
        Measures consensus between experts.
        """

        n = len(outputs_emb)
        if n <= 1:
            return [1.0] * n

        scores = []

        for i in range(n):
            others = [
                1 - cosine(outputs_emb[i], outputs_emb[j])
                for j in range(n)
                if j != i
            ]
            scores.append(float(np.mean(others)))

        return scores

    # --------------------------------------------------------

    def evaluate(self, data: EvaluationInput) -> EvaluationResult:
        start = time.perf_counter()

        if not data.candidates:
            return EvaluationResult(
                ranked=[],
                best=None,
                needs_retry=True,
                evaluation_time=0,
            )

        logger.info(
            f"Evaluating {len(data.candidates)} candidates for task {data.task_id}"
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------
        texts = [data.query] + data.candidates
        emb = self.model.encode(texts, normalize_embeddings=True)
        emb = np.array(emb)

        query_emb = emb[0]
        outputs_emb = emb[1:]

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------
        relevance = self._relevance_scores(query_emb, outputs_emb)
        agreement = self._agreement_scores(outputs_emb)

        ranked: List[CandidateScore] = []

        for i, text in enumerate(data.candidates):
            final = (0.7 * relevance[i]) + (0.3 * agreement[i])

            ranked.append(
                CandidateScore(
                    text=text,
                    relevance=float(relevance[i]),
                    agreement=float(agreement[i]),
                    final_score=float(final),
                    metadata={},
                )
            )

        ranked.sort(key=lambda x: x.final_score, reverse=True)

        best = ranked[0] if ranked else None
        needs_retry = best.final_score < self.retry_threshold if best else True

        t = time.perf_counter() - start

        logger.info(
            f"Best score: {best.final_score if best else 'none'} | retry={needs_retry}"
        )

        return EvaluationResult(
            ranked=ranked,
            best=best,
            needs_retry=needs_retry,
            evaluation_time=t,
        )


# ============================================================
# Factory
# ============================================================

_evaluator_instance: Optional[OutputEvaluator] = None


def get_evaluator() -> OutputEvaluator:
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = OutputEvaluator()
    return _evaluator_instance
