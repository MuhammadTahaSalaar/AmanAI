"""Embedding model wrapper for generating vector embeddings."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Embedder:
    """Wraps a SentenceTransformer model for generating document and query embeddings.

    Uses BAAI/bge-small-en-v1.5 by default with instruction prefixes
    for optimal retrieval performance.
    """

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._device = device or config.EMBEDDING_DEVICE
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info(
            "Embedder initialized: model=%s, device=%s",
            self._model_name,
            self._device,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts.

        Args:
            texts: Document texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query with instruction prefix.

        Args:
            query: The search query.

        Returns:
            Embedding vector for the query.
        """
        prefixed = self.QUERY_PREFIX + query
        embedding = self._model.encode(prefixed, show_progress_bar=False)
        return embedding.tolist()
