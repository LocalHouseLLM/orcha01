from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from loguru import logger


# ============================================================
# Contracts
# ============================================================

class DecomposeRequest(BaseModel):
    query: str
    max_tasks: int = 8


class SubTask(BaseModel):
    id: str
    description: str
    domain: Optional[str] = None
    priority: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DecomposeResponse(BaseModel):
    tasks: List[SubTask]
    embedding_time: float
    clustering_time: float


# ============================================================
# Helpers
# ============================================================

@dataclass
class TaskTemplate:
    """
    Represents known decomposition archetypes.
    """
    name: str
    description: str
    domain: str
    priority: float = 0.5


# ============================================================
# Industrial Decomposer
# ============================================================

class IntelligentDecomposer:
    """
    Semantic task decomposer.

    Converts:
        "build trading strategy"

    into structured subtasks like:
        market analysis
        risk evaluation
        entry logic
        exit logic
        backtest
        explanation

    using embeddings + similarity search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        logger.info("Loading decomposer model…")

        self.model = SentenceTransformer(model_name, device=device)
        self.templates: List[TaskTemplate] = []
        self.index = None
        self.template_matrix = None

        self._bootstrap_templates()

    # --------------------------------------------------------

    def _bootstrap_templates(self):
        """
        Initial expert reasoning building blocks.
        You will expand this over time.
        """

        logger.info("Bootstrapping decomposition templates")

        base = [
            # finance
            TaskTemplate("market_analysis", "analyze market conditions", "finance", 0.9),
            TaskTemplate("risk_model", "evaluate risk exposure", "finance", 0.9),
            TaskTemplate("entry_logic", "define entry signals", "finance", 0.8),
            TaskTemplate("exit_logic", "define exit signals", "finance", 0.8),
            TaskTemplate("position_sizing", "calculate capital allocation", "finance", 0.7),
            TaskTemplate("backtest", "run historical simulation", "finance", 0.95),

            # reasoning
            TaskTemplate("fact_check", "verify factual correctness", "general", 0.9),
            TaskTemplate("critique", "identify logical flaws", "general", 0.7),
            TaskTemplate("summarize", "produce concise summary", "general", 0.6),
            TaskTemplate("explain", "generate explanation", "general", 0.8),
        ]

        self.templates = base

        self._build_index()

    # --------------------------------------------------------

    def _build_index(self):
        logger.info("Building FAISS index for templates")

        texts = [t.description for t in self.templates]
        emb = self.model.encode(texts, normalize_embeddings=True)
        emb = np.array(emb).astype("float32")

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)
        self.template_matrix = emb

        logger.info(f"Indexed {len(self.templates)} decomposition templates")

    # --------------------------------------------------------

    def decompose(self, request: DecomposeRequest) -> DecomposeResponse:
        import time

        if not request.query.strip():
            return DecomposeResponse(tasks=[], embedding_time=0, clustering_time=0)

        logger.info(f"Decomposing query: {request.query}")

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------
        t0 = time.perf_counter()
        q_emb = self.model.encode(
            [request.query],
            normalize_embeddings=True,
        ).astype("float32")
        embedding_time = time.perf_counter() - t0

        # ----------------------------------------------------
        # Similarity search
        # ----------------------------------------------------
        t1 = time.perf_counter()
        k = min(request.max_tasks, len(self.templates))
        scores, ids = self.index.search(q_emb, k)
        clustering_time = time.perf_counter() - t1

        tasks: List[SubTask] = []

        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue

            template = self.templates[idx]

            # ignore weak matches
            if float(score) < 0.25:
                continue

            tasks.append(
                SubTask(
                    id=template.name,
                    description=template.description,
                    domain=template.domain,
                    priority=template.priority * float(score),
                    metadata={"similarity": float(score)},
                )
            )

        logger.info(f"Generated {len(tasks)} subtasks")

        return DecomposeResponse(
            tasks=tasks,
            embedding_time=embedding_time,
            clustering_time=clustering_time,
        )


# ============================================================
# Factory
# ============================================================

_decomposer_instance: Optional[IntelligentDecomposer] = None


def get_decomposer() -> IntelligentDecomposer:
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = IntelligentDecomposer()
    return _decomposer_instance
