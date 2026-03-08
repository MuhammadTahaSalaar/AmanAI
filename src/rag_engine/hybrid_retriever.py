"""Hybrid retriever combining BM25 sparse and ChromaDB dense retrieval."""

from __future__ import annotations

import logging
from collections import defaultdict

import config
from src.data_processing.base_processor import Document
from src.rag_engine.bm25_retriever import BM25Retriever
from src.rag_engine.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever:
    """Combines BM25 keyword retrieval with dense vector retrieval.

    Uses a weighted ensemble: Final_Score = BM25_WEIGHT * BM25 + VECTOR_WEIGHT * Vector.
    This ensures exact keyword matches (product names, identifiers) are
    captured alongside semantic similarity.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_retriever = bm25_retriever
        self._bm25_weight = bm25_weight or config.BM25_WEIGHT
        self._vector_weight = vector_weight or config.VECTOR_WEIGHT

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve documents using hybrid BM25 + Vector search.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            Merged and re-scored list of Document objects.
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        # Retrieve from both sources (fetch more than top_k for better merging)
        fetch_k = top_k * 2
        vector_results = self._vector_store.query(query, top_k=fetch_k)
        bm25_results = self._bm25_retriever.retrieve(query, top_k=fetch_k)

        # Reciprocal Rank Fusion for combining scores
        content_scores: defaultdict[str, float] = defaultdict(float)
        content_to_doc: dict[str, Document] = {}

        for rank, doc in enumerate(vector_results):
            rrf = 1.0 / (rank + 60)  # RRF constant k=60
            content_scores[doc.content] += self._vector_weight * rrf
            content_to_doc[doc.content] = doc

        for rank, doc in enumerate(bm25_results):
            rrf = 1.0 / (rank + 60)
            content_scores[doc.content] += self._bm25_weight * rrf
            content_to_doc[doc.content] = doc

        # Sort by combined score
        sorted_contents = sorted(
            content_scores.keys(),
            key=lambda c: content_scores[c],
            reverse=True,
        )

        results = [content_to_doc[c] for c in sorted_contents[:top_k]]
        logger.debug(
            "Hybrid retrieval: %d vector + %d bm25 → %d merged results",
            len(vector_results),
            len(bm25_results),
            len(results),
        )
        return results
