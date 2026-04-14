"""Cross-encoder re-ranker for filtering retrieved documents."""

from __future__ import annotations

import logging
import warnings

from flashrank import Ranker, RerankRequest

import config
from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Minimum reranker score for a query to be considered in-domain.
# FlashRank scores are bimodal (near-0 for OOD, >0.5 for in-domain).
# A low threshold catches truly irrelevant queries (weather, politics)
# while letting vague but legitimate banking queries through.
MIN_RELEVANCE_SCORE: float = 0.05


class Reranker:
    """Re-ranks retrieved documents using a lightweight cross-encoder.

    Uses FlashRank for fast, accurate re-ranking that filters the
    most relevant context for the small 3B LLM.
    """

    def __init__(self, top_k: int | None = None) -> None:
        self._top_k = top_k or config.RERANK_TOP_K
        # Suppress flashrank progress and download messages
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            flashrank_logger = logging.getLogger("flashrank.Ranker")
            old_level = flashrank_logger.level
            flashrank_logger.setLevel(logging.ERROR)
            self._ranker = Ranker()
            flashrank_logger.setLevel(old_level)
        logger.info("Reranker initialized (top_k=%d)", self._top_k)

    def rerank(
        self, query: str, documents: list[Document]
    ) -> tuple[list[Document], list[float]]:
        """Re-rank documents by relevance to the query.

        Args:
            query: The user's query.
            documents: List of candidate documents from retrieval.

        Returns:
            Tuple of (top-k re-ranked documents, corresponding scores).
        """
        if not documents:
            return [], []

        passages = [
            {"id": i, "text": doc.content} for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        reranked: list[Document] = []
        scores: list[float] = []
        for result in results[: self._top_k]:
            idx = result["id"]
            reranked.append(documents[idx])
            scores.append(float(result.get("score", 0.0)))

        logger.debug(
            "Reranked %d → %d documents (top score=%.3f) for query: %s",
            len(documents),
            len(reranked),
            scores[0] if scores else 0.0,
            query[:50],
        )
        return reranked, scores
