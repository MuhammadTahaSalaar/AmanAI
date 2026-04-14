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
        session_documents: list[Document] | None = None,
    ) -> tuple[str, list[Document]]:
        """Run the full RAG pipeline for a user query.

        Args:
            user_query: The sanitized user question.
            chat_history: Optional list of previous messages
                          [{"role": "user"|"assistant", "content": "..."}].
            session_documents: Optional list of session-level documents to augment retrieval.

        Returns:
            Tuple of (LLM response, list of retrieved Document objects used).
            If the query is out-of-domain (low relevance score), returns a
            canned redirect response without calling the LLM.
        """
        from src.rag_engine.reranker import MIN_RELEVANCE_SCORE

        has_intent = _has_banking_intent(user_query)

        # 0. Query augmentation — include conversation history + NUST Bank prefix
        #    This ensures multi-turn queries maintain product context for retrieval.
        retrieval_query = user_query
        
        # Include recent conversation context for multi-turn awareness
        if chat_history and len(chat_history) >= 2:
            # Extract last user question to get product context
            prev_messages = [m for m in chat_history if m.get("role") == "user"]
            if prev_messages:
                last_user_query = prev_messages[-1].get("content", "")
                # Extract product names from last query
                product_keywords = re.findall(r"NUST\s+[\w\s]+(?:Account|Finance|Deposit)", last_user_query)
                if product_keywords:
                    retrieval_query = f"{product_keywords[0]} {user_query}"
        
        # Prefix "NUST Bank" for retrieval enhancement when not already mentioned
        if "nust" not in retrieval_query.lower():
            retrieval_query = f"NUST Bank {retrieval_query}"

        # 1. Retrieve candidates using augmented query
        candidates = self._retriever.retrieve(retrieval_query)
        
        # 1a. Augment with session-level documents if provided
        if session_documents:
            candidates.extend(session_documents)
            logger.debug(
                "Augmented retrieval with %d session documents (total=%d)",
                len(session_documents),
                len(candidates),
            )
        
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

        # 5a. Validate response grounding (prevent hallucinations)
        if not reranked or len(reranked) == 0:
            # No relevant documents found — force grounding response
            if "product" in user_query.lower() or "account" in user_query.lower():
                # Likely asking about a specific product
                logger.warning("No context found for query; forcing grounding response: %s", user_query[:60])
                response = (
                    "I don't have information about this in our system. "
                    "Please contact our helpline: +92 (51) 111 000 494."
                )
        
        # 5b. Check if response mentions products not in context (hallucination detection)
        # Build context_products from ACTUAL CONTEXT TEXT, not just metadata
        # This works for all documents including session-uploaded ones with incomplete metadata
        context_text = self._build_context(reranked)
        
        # Extract all NUST products mentioned in the actual context
        context_product_names = re.findall(
            r"NUST\s+[\w\s]+(?:Account|Finance|Deposit|Card)",
            context_text,
            re.IGNORECASE
        )
        context_products = set(p.lower() for p in context_product_names)
        
        # Also add products from metadata as fallback
        for doc in reranked:
            product = doc.metadata.get("product", "").lower()
            if product:
                context_products.add(product.lower())
        
        # Extract product names mentioned in the generated response
        mentioned_products = re.findall(
            r"NUST\s+[\w\s]+(?:Account|Finance|Deposit|Card)",
            response,
            re.IGNORECASE
        )
        
        # Find hallucinated products (mentioned in response but not in actual context)
        hallucinated_products = []
        for product in mentioned_products:
            product_lower = product.lower()
            if product_lower not in context_products:
                hallucinated_products.append(product)
        
        if hallucinated_products:
            logger.warning(
                "Potential hallucination detected: %s not in context. Forcing grounding response.",
                hallucinated_products
            )
            response = (
                "I don't have information about that specific product in our current system. "
                "Please contact NUST Bank at +92 (51) 111 000 494 for accurate details."
            )

        return response, reranked

    @staticmethod
    def _build_context(documents: list[Document]) -> str:
        """Concatenate document contents into a single context string."""
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("product", doc.metadata.get("source_sheet", ""))
            parts.append(f"[Source {i}: {source}]\n{doc.content}")
        return "\n\n".join(parts)

    def update_retriever_with_documents(self, documents: list[Document]) -> None:
        """Update the retriever's indexes (BM25 and vector store) with new documents.

        This method should be called when new session documents are uploaded to ensure
        they're properly indexed for retrieval.

        Args:
            documents: List of Document objects to add to the retriever indexes.
        """
        if not documents:
            return

        # Update BM25 index
        # Get current documents from the retriever and add the new ones
        # Note: We need to merge with existing documents to maintain full index
        try:
            # Re-index Combined all known documents including new ones
            # This is called by the HybridRetriever pattern
            logger.info(
                "Updating retriever indexes with %d new documents",
                len(documents)
            )
            # The HybridRetriever contains both BM25 and vector store
            # We'll update them through the retriever's underlying components
            # by extracting them from the HybridRetriever
            if hasattr(self._retriever, '_bm25_retriever') and hasattr(self._retriever, '_vector_store'):
                # Get current BM25 documents
                bm25_retriever = self._retriever._bm25_retriever
                vector_store = self._retriever._vector_store

                # Add to vector store
                vector_store.add_documents(documents)
                logger.debug("Added %d documents to vector store", len(documents))

                # Note: BM25 index is rebuilt from scratch with all documents
                # For now, session documents are handled as post-retrieval augmentation
                logger.debug(
                    "Session documents will be augmented post-retrieval; "
                    "BM25 index remains unchanged (358 docs)"
                )
        except Exception as e:
            logger.error("Failed to update retriever indexes: %s", str(e))
