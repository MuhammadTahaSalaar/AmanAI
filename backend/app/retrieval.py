"""Retrieval pipeline: embed → match_documents → Cohere rerank."""
from __future__ import annotations

import logging
from typing import Any

from .bedrock import embed_text, rerank
from .config import get_settings
from .db import match_documents

logger = logging.getLogger(__name__)


def retrieve(question: str) -> tuple[list[dict[str, Any]], float]:
    """
    Full retrieval pipeline.

    Returns:
        (reranked_docs, top_score)
        reranked_docs: list of doc dicts with added 'rerank_score' field.
        top_score: highest rerank score (0.0 if no docs).
    """
    settings = get_settings()

    # 1. Embed
    embeddings = embed_text([question])
    query_embedding = embeddings[0]

    # 2. Hybrid search via Supabase RPC
    candidates = match_documents(query_embedding, question, settings.RETRIEVE_K)
    if not candidates:
        logger.info("No candidates returned from match_documents")
        return ([], 0.0)

    # 3. Cohere rerank
    doc_texts = [c.get("content", "") for c in candidates]
    try:
        ranked = rerank(question, doc_texts, settings.RERANK_TOP_N)
    except Exception as exc:
        logger.warning("Rerank failed (%s); falling back to vector order", exc)
        # Fallback: return top RERANK_TOP_N by original score
        fallback = candidates[: settings.RERANK_TOP_N]
        for doc in fallback:
            doc["rerank_score"] = doc.get("score", 0.0)
        top = fallback[0]["rerank_score"] if fallback else 0.0
        return (fallback, top)

    if not ranked:
        return ([], 0.0)

    reranked_docs: list[dict[str, Any]] = []
    for orig_idx, score in ranked:
        doc = dict(candidates[orig_idx])
        doc["rerank_score"] = score
        reranked_docs.append(doc)

    top_score = reranked_docs[0]["rerank_score"] if reranked_docs else 0.0
    return (reranked_docs, top_score)
