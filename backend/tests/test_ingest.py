"""Unit tests for ingestion pipeline: chunking, dedup, parsing."""
from __future__ import annotations

import json

import pytest

from backend.app.ingest import (
    CHUNK_CHARS,
    MAX_PDF_PAGES,
    chunk_items,
    content_hash,
    ingest_bytes,
    ingest_items,
    parse_bytes,
)


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

def test_content_hash_deterministic():
    h1 = content_hash("hello world")
    h2 = content_hash("hello world")
    assert h1 == h2


def test_content_hash_strips_whitespace():
    assert content_hash("hello") == content_hash("  hello  ")


def test_content_hash_different_content():
    assert content_hash("abc") != content_hash("xyz")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_atomic_item_kept_whole():
    item = {"content": "Rate: 12% per annum", "metadata": {"rate": "12%", "product": "FD"}}
    chunks = chunk_items([item])
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Rate: 12% per annum"


def test_short_free_text_kept_whole():
    item = {"content": "Short text.", "metadata": {}}
    chunks = chunk_items([item])
    assert len(chunks) == 1


def test_long_free_text_chunked():
    long_text = "word " * 1000  # ~5000 chars
    item = {"content": long_text, "metadata": {}}
    chunks = chunk_items([item])
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["content"]) <= CHUNK_CHARS + 10  # allow small overage


def test_empty_content_skipped():
    items = [{"content": "", "metadata": {}}, {"content": "   ", "metadata": {}}]
    chunks = chunk_items(items)
    assert chunks == []


def test_chunk_preserves_metadata():
    long_text = "a" * (CHUNK_CHARS * 2 + 100)
    item = {"content": long_text, "metadata": {"product": "Test"}}
    chunks = chunk_items([item])
    for chunk in chunks:
        assert chunk["metadata"]["product"] == "Test"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_json_list():
    data = json.dumps([{"content": "hello", "metadata": {"k": "v"}}]).encode()
    items = parse_bytes("docs.json", data)
    assert len(items) == 1
    assert items[0]["content"] == "hello"


def test_parse_json_with_documents_key():
    data = json.dumps({"documents": [{"content": "hi", "metadata": {}}]}).encode()
    items = parse_bytes("docs.json", data)
    assert len(items) == 1


def test_parse_txt():
    data = b"This is a plain text document."
    items = parse_bytes("file.txt", data)
    assert len(items) == 1
    assert "plain text" in items[0]["content"]


def test_parse_csv():
    csv_data = b"content,category\nFixed deposit info,FD\nSavings info,SA"
    items = parse_bytes("data.csv", csv_data)
    assert len(items) == 2
    assert items[0]["content"] == "Fixed deposit info"
    assert items[0]["metadata"]["category"] == "FD"


def test_parse_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_bytes("file.xlsx", b"data")


# ---------------------------------------------------------------------------
# ingest_items with mocked bedrock + db
# ---------------------------------------------------------------------------

def test_ingest_items_calls_embed_and_upsert(monkeypatch):
    embed_calls = []
    upsert_calls = []

    import backend.app.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "embed_text", lambda texts: (embed_calls.append(texts) or [[0.0] * 1024] * len(texts)))
    monkeypatch.setattr(ingest_mod, "upsert_documents", lambda rows: (upsert_calls.append(rows) or {"added": len(rows), "skipped": 0}))

    items = [{"content": "Savings rate is 12%.", "metadata": {"product": "SA", "rate": "12%"}}]
    result = ingest_items(items, source_kind="seed")

    assert len(embed_calls) == 1
    assert len(upsert_calls) == 1
    assert result["added"] == 1
    assert result["skipped"] == 0


def test_ingest_items_dedup_hash_present(monkeypatch):
    captured_rows = []

    import backend.app.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "embed_text", lambda texts: [[0.0] * 1024] * len(texts))
    monkeypatch.setattr(ingest_mod, "upsert_documents", lambda rows: (captured_rows.extend(rows) or {"added": len(rows), "skipped": 0}))

    items = [{"content": "Some content", "metadata": {}}]
    ingest_items(items)

    assert "content_hash" in captured_rows[0]
    assert len(captured_rows[0]["content_hash"]) == 64  # sha256 hex


def test_ingest_bytes_oversize_raises():
    """ingest_bytes must raise ValueError when file exceeds MAX_UPLOAD_MB."""
    from backend.app.config import get_settings
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    oversized = b"x" * (max_bytes + 1)
    with pytest.raises(ValueError, match="exceeds maximum size"):
        ingest_bytes("file.txt", oversized)


def test_pdf_page_cap_raises():
    """PDFs exceeding MAX_PDF_PAGES should raise ValueError."""
    from unittest.mock import MagicMock, patch

    from backend.app.ingest import _parse_pdf, MAX_PDF_PAGES

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "some text"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page] * (MAX_PDF_PAGES + 1)

    with patch("pypdf.PdfReader", return_value=fake_reader):
        with pytest.raises(ValueError, match="exceeds the"):
            _parse_pdf(b"%PDF-1.4 fake")


def test_pdf_at_page_limit_passes():
    """PDFs at exactly MAX_PDF_PAGES should be accepted."""
    from unittest.mock import MagicMock, patch

    from backend.app.ingest import _parse_pdf, MAX_PDF_PAGES

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "page text"
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page] * MAX_PDF_PAGES

    with patch("pypdf.PdfReader", return_value=fake_reader):
        pages = _parse_pdf(b"%PDF-1.4 fake")
    assert len(pages) == MAX_PDF_PAGES


def test_ingest_content_sanitized(monkeypatch):
    """Control characters in document content are stripped before storage."""
    captured_rows = []

    import backend.app.ingest as ingest_mod
    monkeypatch.setattr(ingest_mod, "embed_text", lambda texts: [[0.0] * 1024] * len(texts))
    monkeypatch.setattr(ingest_mod, "upsert_documents", lambda rows: (captured_rows.extend(rows) or {"added": len(rows), "skipped": 0}))

    items = [{"content": "hello\x00world\x01test", "metadata": {}}]
    ingest_items(items)

    assert "\x00" not in captured_rows[0]["content"]
    assert "\x01" not in captured_rows[0]["content"]
    assert "helloworld" in captured_rows[0]["content"]
