"""End-to-end prompt tests for AmanAI.

Tests the full RAG pipeline (retrieval → rerank → generation) against
20+ real-world prompts to ensure correct product recommendations, OOD
rejection, and factual accuracy.

These tests require the full pipeline (embedder, vector store, BM25,
reranker, LLM) to be available. Skip with: pytest -m "not e2e"
"""

from __future__ import annotations

import os
import sys
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test Data: Expected prompt → product/keyword mappings ─────────────────

E2E_CASES = [
    # --- Age-based product recommendations ---
    {
        "id": "age_child_8",
        "prompt": "Which account is suitable for my 8 year old child?",
        "must_contain": ["Little Champs"],
        "must_not_contain": ["Waqaar", "Freelancer"],
    },
    {
        "id": "age_brother_14",
        "prompt": "Can I open an account for my little brother who is 14 years old?",
        "must_contain": ["Little Champs"],
        "must_not_contain": ["Waqaar", "Freelancer", "Asaan Digital"],
    },
    {
        "id": "age_12_year_old",
        "prompt": "which account should I open for someone who is 12 years old?",
        "must_contain": ["Little Champs"],
        "must_not_contain": ["Waqaar"],
    },
    {
        "id": "age_grandmother_55",
        "prompt": "My grandmother wishes to open an account, which one would be best? She is 60.",
        "must_contain": ["Waqaar"],
        "must_not_contain": ["Little Champs", "Freelancer"],
    },
    {
        "id": "age_retired_58",
        "prompt": "Best account for a retired person aged 58?",
        "must_contain": ["Waqaar"],
        "must_not_contain": ["Little Champs"],
    },
    # --- Demographic-based ---
    {
        "id": "freelancer",
        "prompt": "What account is best for freelancers?",
        "must_contain": ["Freelancer"],
        "must_not_contain": ["Waqaar", "Little Champs"],
    },
    {
        "id": "new_to_banking",
        "prompt": "What account for a person who is new to banking?",
        "must_contain": ["Asaan"],
        "must_not_contain": ["Waqaar", "Little Champs"],
    },
    {
        "id": "woman_entrepreneur",
        "prompt": "Which account is designed for women?",
        "must_contain": ["Sahar"],
        "must_not_contain": [],
    },
    # --- Product-specific queries ---
    {
        "id": "car_finance",
        "prompt": "I want to buy a car, do you have financing?",
        "must_contain": ["NUST4Car", "Auto", "car", "finance"],
        "must_not_contain": [],
        "any_contain": True,  # any of must_contain is sufficient
    },
    {
        "id": "solar_finance",
        "prompt": "Tell me about solar financing options",
        "must_contain": ["Ujala"],
        "must_not_contain": [],
    },
    {
        "id": "home_finance",
        "prompt": "Tell me about home financing or mortgage options",
        "must_contain": ["Mortgage", "Imarat", "home", "property"],
        "must_not_contain": [],
        "any_contain": True,
    },
    # --- Rate queries ---
    {
        "id": "savings_rate",
        "prompt": "What is the profit rate on savings?",
        "must_contain": ["%"],
        "must_not_contain": [],
    },
    {
        "id": "term_deposit_rates",
        "prompt": "What are the term deposit rates?",
        "must_contain": ["%"],
        "must_not_contain": [],
    },
    # --- General banking ---
    {
        "id": "fund_transfer",
        "prompt": "How do I transfer funds to another account?",
        "must_contain": ["transfer", "fund", "IBFT", "RAAST"],
        "must_not_contain": [],
        "any_contain": True,
    },
    {
        "id": "delete_account",
        "prompt": "How do I delete my mobile banking account?",
        "must_contain": ["+92", "111 000 494", "helpline", "call"],
        "must_not_contain": [],
        "any_contain": True,
    },
    {
        "id": "little_champs_docs",
        "prompt": "What documents do I need for Little Champs account?",
        "must_contain": ["Little Champs"],
        "must_not_contain": [],
    },
    {
        "id": "features",
        "prompt": "What are your features, what can you tell me about?",
        "must_contain": ["NUST Bank", "banking", "account", "product", "service"],
        "must_not_contain": [],
        "any_contain": True,
    },
    # --- OOD rejection ---
    {
        "id": "ood_bitcoin",
        "prompt": "I want to invest in Bitcoin, can you help?",
        "must_contain": ["only", "NUST Bank", "banking", "assist", "help"],
        "must_not_contain": [],
        "any_contain": True,
        "is_ood": True,
    },
    # --- Non-resident ---
    {
        "id": "overseas_pakistani",
        "prompt": "I live in Dubai and want to open a Pakistani bank account",
        "must_contain": ["Roshan Digital"],
        "must_not_contain": [],
    },
    {
        "id": "business_account",
        "prompt": "I need a business account for my startup",
        "must_contain": ["Value Plus Business", "Business", "business"],
        "must_not_contain": [],
        "any_contain": True,
    },
]


def _check_case(response: str, case: dict) -> list[str]:
    """Check response against a test case. Returns list of failure reasons."""
    failures = []
    resp_lower = response.lower()

    # Check must_contain
    if case.get("any_contain"):
        if not any(kw.lower() in resp_lower for kw in case["must_contain"]):
            failures.append(
                f"Response must contain at least one of {case['must_contain']}"
            )
    else:
        for kw in case["must_contain"]:
            if kw.lower() not in resp_lower:
                failures.append(f"Response missing required keyword: '{kw}'")

    # Check must_not_contain
    for kw in case.get("must_not_contain", []):
        if kw.lower() in resp_lower:
            failures.append(f"Response contains forbidden keyword: '{kw}'")

    return failures


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rag_pipeline():
    """Build the full RAG pipeline once for all tests in this module."""
    import config
    from src.data_processing.etl_pipeline import ETLPipeline
    from src.rag_engine.embedder import Embedder
    from src.rag_engine.vector_store import VectorStore
    from src.rag_engine.bm25_retriever import BM25Retriever
    from src.rag_engine.hybrid_retriever import HybridRetriever
    from src.rag_engine.reranker import Reranker
    from src.rag_engine.rag_chain import RAGChain
    from src.llm.model_loader import ModelLoader

    # ETL
    pipeline = ETLPipeline()
    documents = pipeline.run()

    # Components
    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)
    if vector_store.count == 0:
        vector_store.add_documents(documents)

    bm25 = BM25Retriever()
    bm25.index(documents)
    hybrid = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
    reranker = Reranker()

    model_loader = ModelLoader()
    model_loader.load()

    rag_chain = RAGChain(retriever=hybrid, reranker=reranker, model_loader=model_loader)
    return rag_chain


@pytest.mark.e2e
class TestE2EPrompts:
    """End-to-end tests for real-world prompts."""

    @pytest.mark.parametrize(
        "case",
        E2E_CASES,
        ids=[c["id"] for c in E2E_CASES],
    )
    def test_prompt(self, rag_pipeline, case):
        """Test a single prompt against expected keywords."""
        response, docs = rag_pipeline.query(case["prompt"])
        failures = _check_case(response, case)
        if failures:
            msg = (
                f"\nPrompt: {case['prompt']}\n"
                f"Response: {response[:500]}\n"
                f"Failures:\n  - " + "\n  - ".join(failures)
            )
            pytest.fail(msg)


@pytest.mark.e2e
class TestMultiTurn:
    """Tests for multi-turn conversation context."""

    def test_followup_maintains_product_context(self, rag_pipeline):
        """After asking about Waqaar, a follow-up about rates should stay on Waqaar."""
        # Turn 1
        resp1, _ = rag_pipeline.query("Tell me about the NUST Waqaar Account")
        assert "waqaar" in resp1.lower(), f"Turn 1 should mention Waqaar: {resp1[:300]}"

        # Turn 2 — follow-up
        history = [
            {"role": "user", "content": "Tell me about the NUST Waqaar Account"},
            {"role": "assistant", "content": resp1},
        ]
        resp2, _ = rag_pipeline.query("What are the rates?", chat_history=history)
        # Should reference Waqaar or at least provide rate info
        assert "%" in resp2 or "waqaar" in resp2.lower(), (
            f"Follow-up should provide Waqaar rates: {resp2[:300]}"
        )

    def test_child_followup(self, rag_pipeline):
        """After recommending Little Champs, follow-up about docs should stay on Little Champs."""
        resp1, _ = rag_pipeline.query("Which account for my 14 year old brother?")
        assert "little champs" in resp1.lower(), f"Turn 1 should mention Little Champs: {resp1[:300]}"

        history = [
            {"role": "user", "content": "Which account for my 14 year old brother?"},
            {"role": "assistant", "content": resp1},
        ]
        resp2, _ = rag_pipeline.query("What documents are needed?", chat_history=history)
        assert "little champs" in resp2.lower() or "document" in resp2.lower(), (
            f"Follow-up should be about Little Champs docs: {resp2[:300]}"
        )


# ── Retrieval-only tests (no LLM needed) ─────────────────────────────────────

@pytest.fixture(scope="module")
def retrieval_pipeline():
    """Build retrieval pipeline (no LLM) for testing doc retrieval quality."""
    import config
    from src.data_processing.etl_pipeline import ETLPipeline
    from src.rag_engine.embedder import Embedder
    from src.rag_engine.vector_store import VectorStore
    from src.rag_engine.bm25_retriever import BM25Retriever
    from src.rag_engine.hybrid_retriever import HybridRetriever
    from src.rag_engine.reranker import Reranker

    pipeline = ETLPipeline()
    documents = pipeline.run()

    embedder = Embedder()
    vector_store = VectorStore(embedder=embedder)
    if vector_store.count == 0:
        vector_store.add_documents(documents)

    bm25 = BM25Retriever()
    bm25.index(documents)
    hybrid = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
    reranker = Reranker()
    return hybrid, reranker


class TestRetrieval:
    """Test that the right documents are retrieved for key queries."""

    @pytest.mark.parametrize("query,expected_product", [
        ("account for my 12 year old son", "Little Champs"),
        ("account for senior citizen grandmother", "Waqaar"),
        ("freelancer account to receive USD payments", "Freelancer"),
        ("I want to finance a car", "NUST4Car"),
        ("solar panel financing", "Ujala"),
        ("account for new to banking person", "Asaan"),
    ])
    def test_retrieval_finds_correct_product(self, retrieval_pipeline, query, expected_product):
        """Verify the correct product appears in top retrieved documents."""
        hybrid, reranker = retrieval_pipeline
        candidates = hybrid.retrieve(query, top_k=15)
        reranked, scores = reranker.rerank(query, candidates)

        found = any(
            expected_product.lower() in doc.metadata.get("product", "").lower()
            or expected_product.lower() in doc.content[:200].lower()
            for doc in reranked[:5]
        )
        assert found, (
            f"Expected '{expected_product}' in top-5 for '{query}'. "
            f"Got: {[d.metadata.get('product', '?') for d in reranked[:5]]}"
        )
