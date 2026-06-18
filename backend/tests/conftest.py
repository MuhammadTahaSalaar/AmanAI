"""Shared pytest fixtures — all external calls mocked (no network)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Settings override so no real env vars are needed
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    """Override settings with safe test defaults before each test."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_TESTPOOL")
    monkeypatch.setenv("COGNITO_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "")
    # Clear lru_cache so settings are re-read
    from backend.app import config
    config.get_settings.cache_clear()


@pytest.fixture
def fake_embedding() -> list[float]:
    return [0.0] * 1024


@pytest.fixture
def fake_docs(fake_embedding) -> list[dict]:
    return [
        {
            "id": 1,
            "content": "The savings account rate is 12% per annum.",
            "metadata": {"product": "Savings Account", "source_sheet": "rates"},
            "score": 0.9,
            "rerank_score": 0.85,
        }
    ]


@pytest.fixture
def test_client(monkeypatch):
    """TestClient with all external dependencies mocked."""
    _mock_bedrock(monkeypatch)
    _mock_db(monkeypatch)
    _mock_auth(monkeypatch)

    from backend.app.main import app
    return TestClient(app, raise_server_exceptions=False)


def _mock_bedrock(monkeypatch):
    monkeypatch.setattr("backend.app.bedrock.embed_text", lambda texts: [[0.0] * 1024] * len(texts))
    monkeypatch.setattr("backend.app.bedrock.generate", lambda messages: "The savings account rate is 12%.")
    monkeypatch.setattr("backend.app.bedrock.rerank", lambda q, docs, top_n: [(0, 0.9)] * min(top_n, len(docs)))
    monkeypatch.setattr("backend.app.bedrock.apply_guardrail", lambda text, source: ("NONE", text))


def _mock_db(monkeypatch):
    monkeypatch.setattr(
        "backend.app.db.match_documents",
        lambda emb, text, k: [
            {
                "id": 1,
                "content": "The savings account rate is 12% per annum.",
                "metadata": {"product": "Savings Account", "source_sheet": "rates"},
                "score": 0.9,
            }
        ],
    )
    monkeypatch.setattr("backend.app.db.upsert_documents", lambda rows: {"added": len(rows), "skipped": 0})
    monkeypatch.setattr("backend.app.db.count_documents", lambda: 358)


def _mock_auth(monkeypatch):
    """Bypass JWT verification — inject a fake user claims dict."""
    fake_user_claims = {"sub": "test-user", "cognito:groups": ["users"]}
    fake_admin_claims = {"sub": "admin-user", "cognito:groups": ["admin"]}

    from backend.app import auth
    monkeypatch.setattr(auth, "require_user", lambda: fake_user_claims)
    monkeypatch.setattr(auth, "require_admin", lambda: fake_admin_claims)
