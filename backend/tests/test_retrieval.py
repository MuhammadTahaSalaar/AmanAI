"""Unit tests for retrieval pipeline and refusal gate."""
from __future__ import annotations

import pytest


def test_retrieve_returns_docs_and_score(monkeypatch):
    monkeypatch.setattr("backend.app.retrieval.embed_text", lambda texts: [[0.1] * 1024])
    monkeypatch.setattr(
        "backend.app.retrieval.match_documents",
        lambda emb, text, k: [{"id": 1, "content": "Rate is 12%", "metadata": {}, "score": 0.8}],
    )
    monkeypatch.setattr("backend.app.retrieval.rerank", lambda q, docs, top_n: [(0, 0.75)])

    from backend.app.retrieval import retrieve
    docs, top_score = retrieve("What is the rate?")
    assert len(docs) == 1
    assert top_score == pytest.approx(0.75)
    assert docs[0]["rerank_score"] == pytest.approx(0.75)


def test_retrieve_empty_candidates(monkeypatch):
    monkeypatch.setattr("backend.app.retrieval.embed_text", lambda texts: [[0.0] * 1024])
    monkeypatch.setattr("backend.app.retrieval.match_documents", lambda emb, text, k: [])

    from backend.app.retrieval import retrieve
    docs, top_score = retrieve("obscure query")
    assert docs == []
    assert top_score == 0.0


def test_retrieve_rerank_fallback(monkeypatch):
    """When rerank raises, fall back to original vector order."""
    monkeypatch.setattr("backend.app.retrieval.embed_text", lambda texts: [[0.0] * 1024])
    monkeypatch.setattr(
        "backend.app.retrieval.match_documents",
        lambda emb, text, k: [
            {"id": 1, "content": "Doc A", "metadata": {}, "score": 0.9},
            {"id": 2, "content": "Doc B", "metadata": {}, "score": 0.7},
        ],
    )
    monkeypatch.setattr("backend.app.retrieval.rerank", lambda q, docs, top_n: (_ for _ in ()).throw(Exception("throttled")))

    from backend.app import config
    config.get_settings.cache_clear()

    from backend.app.retrieval import retrieve
    docs, top_score = retrieve("query")
    assert len(docs) > 0
    assert top_score > 0.0


def test_refusal_gate_triggered(monkeypatch):
    """Top score below MIN_RERANK_SCORE → refused=True, no LLM call."""
    monkeypatch.setattr("backend.app.retrieval.embed_text", lambda texts: [[0.0] * 1024])
    monkeypatch.setattr(
        "backend.app.retrieval.match_documents",
        lambda emb, text, k: [{"id": 1, "content": "Irrelevant doc", "metadata": {}, "score": 0.1}],
    )
    monkeypatch.setattr("backend.app.retrieval.rerank", lambda q, docs, top_n: [(0, 0.05)])
    monkeypatch.setattr("backend.app.chat.apply_guardrail", lambda text, source: ("NONE", text))

    generate_called = []
    monkeypatch.setattr("backend.app.chat.generate", lambda msgs: generate_called.append(1) or "answer")

    from backend.app.chat import handle_chat
    from backend.app.schemas import ChatRequest

    resp = handle_chat(ChatRequest(message="What is quantum banking?", history=[]))
    assert resp.refused is True
    assert generate_called == []


def test_refusal_gate_passes(monkeypatch):
    """Top score above MIN_RERANK_SCORE → generate is called."""
    monkeypatch.setattr("backend.app.retrieval.embed_text", lambda texts: [[0.0] * 1024])
    monkeypatch.setattr(
        "backend.app.retrieval.match_documents",
        lambda emb, text, k: [{"id": 1, "content": "Savings rate is 12%", "metadata": {"product": "Savings"}, "score": 0.9}],
    )
    monkeypatch.setattr("backend.app.retrieval.rerank", lambda q, docs, top_n: [(0, 0.85)])
    monkeypatch.setattr("backend.app.chat.apply_guardrail", lambda text, source: ("NONE", text))
    monkeypatch.setattr("backend.app.chat.generate", lambda msgs: "The savings rate is 12%.")

    from backend.app.chat import handle_chat
    from backend.app.schemas import ChatRequest

    resp = handle_chat(ChatRequest(message="What is the savings rate?", history=[]))
    assert resp.refused is False
    assert "12%" in resp.answer
