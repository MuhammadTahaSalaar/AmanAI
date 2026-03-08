"""Full RAG chain: retrieve → rerank → generate."""

from __future__ import annotations

import logging
import re

import config
from src.data_processing.base_processor import Document
from src.llm.model_loader import ModelLoader
from src.llm.prompt_templates import PromptTemplates
from src.rag_engine.hybrid_retriever import HybridRetriever
from src.rag_engine.reranker import Reranker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Banking-related keywords.  If a query contains at least one of these
# we assume it is broadly on-topic and skip the score-based OOD gate.
_BANKING_KEYWORDS: frozenset[str] = frozenset({
    "account", "accounts", "deposit", "deposits", "savings", "saving",
    "term", "rate", "rates", "profit", "interest", "loan", "finance",
    "mortgage", "transfer", "payment", "card", "debit", "credit",
    "cheque", "branch", "balance", "withdrawal", "withdraw", "atm",
    "pos", "paypak", "remittance", "asaan", "waqaar", "sahar",
    "maximiser", "bachat", "pensioner", "roshan", "car", "auto",
    "home", "personal", "eligibility", "eligible", "open", "opening",
    "limit", "fee", "charges", "schedule", "maturity", "tenor",
    "monthly", "annually", "quarterly", "pkr", "usd", "cnic", "iban",
    "fund", "funds", "sms", "alert", "inet", "mobile", "banking",
    "bank", "nust", "aman", "amanai", "champs", "little",
})


def _has_banking_intent(query: str) -> bool:
    """Return True if the query contains any banking-related keyword."""
    tokens = set(re.findall(r"\b\w+\b", query.lower()))
    return bool(tokens & _BANKING_KEYWORDS)


class RAGChain:
    """Orchestrates the full RAG pipeline: retrieve, rerank, and generate.

    Wires together the hybrid retriever, re-ranker, and LLM to produce
    grounded, context-aware responses.
    """

    _OOD_RESPONSE = (
        "I can only assist with NUST Bank banking queries. "
        "For other questions, please use an appropriate service. "
        "For banking help, call +92 (51) 111 000 494."
    )

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        model_loader: ModelLoader,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._model_loader = model_loader

    def query(
        self,
        user_query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[Document]]:
        """Run the full RAG pipeline for a user query.

        Args:
            user_query: The sanitized user question.
            chat_history: Optional list of previous messages
                          [{"role": "user"|"assistant", "content": "..."}].

        Returns:
            Tuple of (LLM response, list of retrieved Document objects used).
            If the query is out-of-domain (low relevance score), returns a
            canned redirect response without calling the LLM.
        """
        from src.rag_engine.reranker import MIN_RELEVANCE_SCORE

        has_intent = _has_banking_intent(user_query)

        # 0. Query augmentation — prefix "NUST Bank" for retrieval when the
        #    user doesn't mention it explicitly, so BM25 + vectors find
        #    NUST-specific documents even for short generic queries.
        retrieval_query = user_query
        if "nust" not in user_query.lower():
            retrieval_query = f"NUST Bank {user_query}"

        # 1. Retrieve candidates using augmented query
        candidates = self._retriever.retrieve(retrieval_query)
        logger.debug("Retrieved %d candidates for '%s'", len(candidates), retrieval_query[:60])

        # 2. OOD gate — two tiers:
        #    a) If the query has banking keywords we trust it is on-topic;
        #       rerank with AUGMENTED query for best document ordering.
        #    b) If NO banking keywords, rerank with the ORIGINAL query
        #       so the cross-encoder judges true semantic relevance.
        #       If best score is too low, reject as OOD immediately.
        if has_intent:
            # Tier A: banking-related query — augmented reranking
            reranked, scores = self._reranker.rerank(retrieval_query, candidates)
            logger.debug(
                "Banking intent detected — augmented rerank (top=%.4f)",
                scores[0] if scores else 0.0,
            )
        else:
            # Tier B: no banking keywords — original query reranking for OOD check
            reranked, scores = self._reranker.rerank(user_query, candidates)
            best_score = scores[0] if scores else 0.0
            logger.debug(
                "No banking intent — orig rerank (top=%.4f, threshold=%.4f)",
                best_score, MIN_RELEVANCE_SCORE,
            )
            if best_score < MIN_RELEVANCE_SCORE:
                logger.info("OOD rejected (score=%.4f): %s", best_score, user_query[:80])
                return self._OOD_RESPONSE, []
            # Passed OOD — re-rank with augmented query for better context
            reranked, scores = self._reranker.rerank(retrieval_query, candidates)

        logger.debug("Reranked to %d documents", len(reranked))

        # 3. Build context
        context = self._build_context(reranked)

        # 4. Build prompt
        prompt = PromptTemplates.build_rag_prompt(
            user_query=user_query,
            context=context,
            chat_history=chat_history,
        )

        # 5. Generate
        response = self._model_loader.generate(prompt)
        logger.info("Generated response for query: %s", user_query[:80])

        return response, reranked

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        """Concatenate document contents into a single context string."""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("product", doc.metadata.get("source_sheet", ""))
            parts.append(f"[Source {i}: {source}]\n{doc.content}")
        return "\n\n".join(parts)
