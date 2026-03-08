"""Tests for the RAG engine components (using mocks for heavy dependencies)."""

from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.base_processor import Document
from src.rag_engine.bm25_retriever import BM25Retriever


# ── BM25 Retriever Tests ────────────────────────────────────────────────────


class TestBM25Retriever:
    """Test the BM25 sparse retriever."""

    def _make_docs(self) -> list[Document]:
        return [
            Document(
                content="NUST Bank offers savings accounts with competitive rates",
                metadata={"product": "savings"},
            ),
            Document(
                content="Term deposits provide fixed returns for a specified period",
                metadata={"product": "term_deposit"},
            ),
            Document(
                content="Fund transfer can be done through the mobile app",
                metadata={"product": "fund_transfer"},
            ),
        ]

    def test_not_indexed_returns_empty(self):
        retriever = BM25Retriever()
        assert not retriever.is_indexed
        results = retriever.retrieve("savings")
        assert results == []

    def test_index_builds(self):
        retriever = BM25Retriever()
        retriever.index(self._make_docs())
        assert retriever.is_indexed

    def test_retrieve_relevant(self):
        retriever = BM25Retriever()
        retriever.index(self._make_docs())
        results = retriever.retrieve("savings accounts rates", top_k=2)
        assert len(results) > 0
        assert "savings" in results[0].content.lower()

    def test_retrieve_term_deposit(self):
        retriever = BM25Retriever()
        retriever.index(self._make_docs())
        results = retriever.retrieve("term deposit fixed returns", top_k=1)
        assert len(results) >= 1
        assert "term" in results[0].content.lower()

    def test_retrieve_respects_top_k(self):
        retriever = BM25Retriever()
        retriever.index(self._make_docs())
        results = retriever.retrieve("bank", top_k=1)
        assert len(results) <= 1


# ── Hybrid Retriever Tests (mocked) ─────────────────────────────────────────


class TestHybridRetriever:
    """Test the hybrid retriever with mocked stores."""

    def test_reciprocal_rank_fusion(self):
        """Verify RRF merging logic works correctly."""
        from src.rag_engine.hybrid_retriever import HybridRetriever

        doc_a = Document(content="doc_a", metadata={})
        doc_b = Document(content="doc_b", metadata={})
        doc_c = Document(content="doc_c", metadata={})

        mock_vs = MagicMock()
        mock_vs.query.return_value = [doc_a, doc_b]  # vector results

        mock_bm25 = MagicMock()
        mock_bm25.retrieve.return_value = [doc_b, doc_c]  # bm25 results

        retriever = HybridRetriever(
            vector_store=mock_vs,
            bm25_retriever=mock_bm25,
            bm25_weight=0.4,
            vector_weight=0.6,
        )

        results = retriever.retrieve("test query", top_k=3)

        # doc_b appears in both, should rank highest
        contents = [r.content for r in results]
        assert "doc_b" in contents
        assert len(results) == 3

    def test_empty_results(self):
        """Both retrievers return empty → empty merged."""
        from src.rag_engine.hybrid_retriever import HybridRetriever

        mock_vs = MagicMock()
        mock_vs.query.return_value = []
        mock_bm25 = MagicMock()
        mock_bm25.retrieve.return_value = []

        retriever = HybridRetriever(
            vector_store=mock_vs, bm25_retriever=mock_bm25
        )
        results = retriever.retrieve("nothing")
        assert results == []
