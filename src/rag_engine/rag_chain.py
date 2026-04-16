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
    "account", "accounts", "deposit", "deposits", "savings", "saving", "save",
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
    "ujala", "imarat", "nust4car", "mastercard",
    "processing", "tenure", "markup", "instalment", "apply",
    "children", "child", "kids", "kid", "minor", "minors",
    # Family / age references that imply banking product queries
    "daughter", "son", "grandmother", "grandfather", "father", "mother",
    "retired", "retirement", "freelancer", "freelance", "student",
    "pensioners", "senior", "seniors", "youth",
    "invest", "investment", "money", "income",
})

# Synonym expansion for retrieval — augments the retrieval query with
# related terms so both BM25 keyword matching and vector similarity
# can find relevant documents even when the user's wording differs from
# the stored content (e.g., "children" vs "minors"/"kids").
_RETRIEVAL_SYNONYMS: dict[str, str] = {
    "children": "minors kids Little Champs",
    "child": "minor kid Little Champs",
    "kids": "minors children Little Champs",
    "kid": "minor child Little Champs",
    "minor": "child kid Little Champs",
    "minors": "children kids Little Champs",
    "daughter": "child minor kids Little Champs",
    "son": "child minor kids Little Champs",
    "elderly": "senior pensioner Waqaar senior citizen",
    "retired": "senior pensioner Waqaar retirement",
    "grandmother": "senior elderly pensioner Waqaar senior citizen",
    "grandfather": "senior elderly pensioner Waqaar senior citizen",
    "young": "youth teenager",
    "teenager": "young youth minor",
    "freelancer": "freelance digital account Asaan",
    "freelance": "freelancer digital account Asaan",
    "student": "youth education",
    "old": "senior elderly pensioner Waqaar",
    "senior": "elderly pensioner Waqaar senior citizen",
    "pensioner": "senior elderly Waqaar retired",
}

# Age patterns that imply banking product eligibility queries
_AGE_PATTERN = re.compile(
    r"(?:"
    r"\b\d{1,3}\s*(?:year|yr)s?\s*old\b"            # "55 years old"
    r"|\b(?:above|over|below|under|within)\s+\d{1,3}\s*(?:year|yr)?s?\b"  # "above 55", "under 18 years"
    r"|\b\d{1,3}\s*\+\b"                              # "55+"
    r"|\b(?:senior|elderly|retired|retirement|pensioner|minor|minors)\b"  # age-related words
    r")",
    re.IGNORECASE,
)


def _has_banking_intent(query: str) -> bool:
    """Return True if the query contains any banking-related keyword or age reference."""
    tokens = set(re.findall(r"\b\w+\b", query.lower()))
    if tokens & _BANKING_KEYWORDS:
        return True
    # Age mentions (e.g., "8 year old", "55 years old") strongly imply
    # eligibility/product queries in a banking context.
    if _AGE_PATTERN.search(query):
        return True
    return False


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
        
        # Check if the current query already mentions a specific product
        current_product = re.findall(
            r"NUST[\s\w]*(?:Account|Finance|Deposit)",
            user_query,
            re.IGNORECASE,
        )
        
        # Only augment with previous product context if current query has NO product
        # AND it looks like a genuine follow-up (contains referential words or is very short).
        _FOLLOWUP_HINTS = {"it", "its", "that", "this", "those", "them", "they", "the", "same", "above"}
        if not current_product and chat_history and len(chat_history) >= 2:
            query_tokens = set(re.findall(r"\b\w+\b", user_query.lower()))
            is_followup = bool(query_tokens & _FOLLOWUP_HINTS) or len(user_query.split()) <= 6
            if is_followup:
                # Extract last user question to get product context
                prev_messages = [m for m in chat_history if m.get("role") == "user"]
                if prev_messages:
                    last_user_query = prev_messages[-1].get("content", "")
                    # Extract product names from last query (handles NUST4Car, NUST Imarat Finance, etc.)
                    product_keywords = re.findall(
                        r"NUST[\s\w]*(?:Account|Finance|Deposit)",
                        last_user_query,
                        re.IGNORECASE,
                    )
                    if product_keywords:
                        retrieval_query = f"{product_keywords[0]} {user_query}"
                        # If query was augmented with product context from history,
                        # treat it as having banking intent (it's a follow-up)
                        has_intent = True
        
        # Prefix "NUST Bank" for retrieval enhancement when not already mentioned
        if "nust" not in retrieval_query.lower():
            retrieval_query = f"NUST Bank {retrieval_query}"

        # 0b. Synonym expansion — append related terms to boost BM25 recall
        _tokens = set(re.findall(r"\b\w+\b", retrieval_query.lower()))
        _extra = {_RETRIEVAL_SYNONYMS[t] for t in _tokens if t in _RETRIEVAL_SYNONYMS}
        if _extra:
            retrieval_query = f"{retrieval_query} {' '.join(_extra)}"
            logger.debug("Synonym-expanded retrieval query: %s", retrieval_query[:120])

        # 1. Retrieve candidates using augmented query
        #    Fetch a wider pool so the reranker has enough diversity to score.
        candidates = self._retriever.retrieve(
            retrieval_query, top_k=config.RERANK_TOP_K * 3,
        )
        
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

        # 2b. Product-aware filtering — if the query targets a SINGLE specific
        #     product, drop chunks from OTHER products to prevent the 3B
        #     model from mixing data across products.
        #     Skip filtering when multiple products are mentioned (comparisons).
        query_products = self._PRODUCT_RE.findall(user_query)
        if len(query_products) == 1 and reranked:
            query_product = query_products[0].strip()
            # Disambiguate products with similar names (e.g., Asaan Digital vs Asaan Remittance)
            disambiguated = self._disambiguate_product(query_product, user_query)
            if disambiguated:
                logger.info("Disambiguated product: '%s' → '%s'", query_product, disambiguated)
                query_product = disambiguated
            matched = [
                d for d in reranked
                if self._doc_matches_product(d, query_product)
            ]
            # Only filter if we still have at least 1 matched doc
            if matched:
                dropped = len(reranked) - len(matched)
                if dropped:
                    logger.info(
                        "Product filter: kept %d/%d docs for '%s'",
                        len(matched), len(reranked), query_product,
                    )
                reranked = matched

        # 2c. Product name alignment — if the user's product name differs from the
        #     retrieved docs' actual product name, substitute it in the prompt query
        #     so the LLM doesn't refuse due to a trivial name mismatch (e.g.,
        #     "NUST Ujala Account" vs "NUST Ujala Finance").
        prompt_query = user_query
        if query_products and reranked:
            queried_name = query_products[0].strip()
            # Prefer the disambiguated name if available
            disambiguated = self._disambiguate_product(queried_name, user_query)
            actual_product = disambiguated or reranked[0].metadata.get("product", "")
            if actual_product and queried_name.lower() != actual_product.lower():
                prompt_query = re.sub(
                    re.escape(queried_name), actual_product, user_query,
                    count=1, flags=re.IGNORECASE,
                )
                logger.info(
                    "Aligned product name in prompt: '%s' → '%s'",
                    queried_name, actual_product,
                )

        # 3. Build context
        context = self._build_context(reranked)

        # 3a. If banking intent detected and context found, add an explicit
        #     banking-query hint so the small 3B model doesn't mistakenly
        #     reject indirect queries (e.g., "account for my grandfather").
        if has_intent and reranked:
            prompt_query = f"[NUST Bank product query] {prompt_query}"

        # 4. Build prompt
        prompt = PromptTemplates.build_rag_prompt(
            user_query=prompt_query,
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
        
        # 5b. Hallucination detection is ONLY applied if NO context is found
        # If context exists (reranked docs available), the system prompt constrains the LLM
        # and we trust the LLM's grounding-aware response
        # This prevents false positives from regex mismatches on product names like "NUST4Car"
        if reranked and len(reranked) > 0:
            # Context exists - trust the LLM's response (constrained by system prompt)
            logger.debug(
                "Context provided (%d docs): Trusting LLM response (constrained by system prompt)",
                len(reranked)
            )
        else:
            # No context - this was already handled in section 5a
            pass

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

        try:
            # Add to vector store (single call — no duplicates)
            self._retriever._vector_store.add_documents(documents)

            # Rebuild BM25 with existing docs + new docs so originals are preserved
            existing_bm25_docs = list(self._retriever._bm25_retriever._documents)
            all_docs = existing_bm25_docs + documents
            self._retriever._bm25_retriever.index(all_docs)

            logger.info(
                "Updated retriever indexes with %d new documents (BM25 total: %d)",
                len(documents),
                len(all_docs),
            )
        except Exception as e:
            logger.error("Failed to update retriever indexes: %s", str(e))

    # ---- Product-filtering helpers ----

    _PRODUCT_RE = re.compile(
        r"NUST\s*(?:4Car|Hunarmand|Asaan|Waqaar|Sahar|Maximiser|Bachat|"
        r"Imarat|Ujala|Champs|Little|Roshan|Pensioner|Digital|"
        r"Special\s+Mega\s+Bonus|Mortgage|Personal|Mastercard|PayPak)"
        r"(?:\s+\w+)*",
        re.IGNORECASE,
    )

    # Products with ambiguous short names that need disambiguation
    _AMBIGUOUS_PRODUCTS = {
        "asaan": {
            # If user says "Asaan" without "Remittance", prefer the plain Digital Account
            "default": "NUST Asaan Digital Account",
            "remittance": "NUST Asaan Digital Remittance Account",
        },
    }

    @classmethod
    def _disambiguate_product(cls, matched_name: str, user_query: str) -> str | None:
        """Disambiguate product names that map to multiple actual products.

        Returns the resolved product name or None if no disambiguation needed.
        """
        query_lower = user_query.lower()
        matched_lower = matched_name.lower().strip()

        for key, variants in cls._AMBIGUOUS_PRODUCTS.items():
            if key in matched_lower:
                # Check if user specified a distinguishing keyword
                if "remittance" in query_lower:
                    return variants.get("remittance")
                else:
                    return variants.get("default")
        return None

    @classmethod
    def _extract_product(cls, text: str) -> str | None:
        """Extract a product name from text, e.g. 'NUST4Car', 'NUST Imarat'."""
        m = cls._PRODUCT_RE.search(text)
        return m.group(0).strip() if m else None

    @staticmethod
    def _doc_matches_product(doc: Document, product: str) -> bool:
        """Check if a document belongs to (or is relevant to) the queried product."""
        product_lower = product.lower()
        # Check metadata fields
        meta_product = doc.metadata.get("product", "").lower()
        meta_source = doc.metadata.get("source_sheet", "").lower()
        content_lower = doc.content[:200].lower()  # First 200 chars is enough
        # Use the full product name for matching (not just first word)
        # This prevents "nust asaan" from matching both Digital and Remittance
        key = product_lower.replace("nust ", "").strip()
        return (
            key in meta_product
            or key in meta_source
            or key in content_lower
        )
