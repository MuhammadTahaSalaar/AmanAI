"""BM25 sparse retriever for keyword-based document retrieval."""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BM25Retriever:
    """Sparse keyword retriever using the BM25 (Okapi) algorithm.

    Indexes document texts using term-frequency scoring for exact
    keyword matching (e.g., product names, account identifiers).
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    @property
    def is_indexed(self) -> bool:
        """Whether the BM25 index has been built."""
        return self._bm25 is not None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words, stripping punctuation."""
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, documents: list[Document]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            documents: List of Document objects to index.
        """
        self._documents = documents
        tokenized = [self._tokenize(doc.content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(documents))

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Retrieve documents by BM25 keyword relevance.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of Document objects scored by BM25.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built; returning empty results")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [self._documents[i] for i in ranked_indices if scores[i] > 0]
