"""ChromaDB vector store wrapper for persistent document storage and retrieval."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import chromadb

import config
from src.data_processing.base_processor import Document
from src.rag_engine.embedder import Embedder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """Manages a persistent ChromaDB collection for document embeddings.

    Provides methods for adding documents, querying by embedding, and
    managing the collection lifecycle.
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or config.CHROMA_COLLECTION_NAME

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized: collection='%s', docs=%d",
            self._collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to embed and store.
        """
        if not documents:
            return

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # Use UUID to avoid ID collisions when add_documents is called multiple times
        ids = [f"doc_{uuid.uuid4().hex[:16]}" for _ in documents]

        # Embed in batches to avoid memory issues
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_metadatas = metadatas[start:end]
            batch_ids = ids[start:end]

            embeddings = self._embedder.embed_documents(batch_texts)
            self._collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

        logger.info("Added %d documents to vector store", len(documents))

    def query(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Query the vector store for relevant documents.

        Args:
            query: The search query string.
            top_k: Number of results to return.
            where: Optional metadata filter dict.

        Returns:
            List of Document objects ranked by relevance.
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        query_embedding = self._embedder.embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        documents: list[Document] = []
        if results["documents"] and results["documents"][0]:
            for i, text in enumerate(results["documents"][0]):
                metadata = (
                    results["metadatas"][0][i] if results["metadatas"] else {}
                )
                documents.append(Document(content=text, metadata=metadata))

        return documents

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset: collection '%s'", self._collection_name)
