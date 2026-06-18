"""Shared ingestion pipeline: parse → chunk → embed → upsert."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from typing import Any

from .bedrock import embed_text
from .config import get_settings
from .db import upsert_documents
from .guardrails import _strip_dangerous_chars

logger = logging.getLogger(__name__)

CHUNK_CHARS = 2400   # ~600 tokens at ~4 chars/token
MAX_PDF_PAGES = 200
CHUNK_OVERLAP = 400  # ~100 tokens overlap
EMBED_BATCH = 20     # texts per Titan batch call


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_json(data: bytes) -> list[dict[str, Any]]:
    parsed = json.loads(data)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "documents" in parsed:
        return parsed["documents"]
    raise ValueError("JSON must be a list of {content, metadata} or {documents: [...]}")


def _parse_txt(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    return [{"content": text, "metadata": {}}]


def _parse_csv(data: bytes) -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        content = row.pop("content", None) or " ".join(row.values())
        rows.append({"content": content, "metadata": dict(row)})
    return rows


def _parse_pdf(data: bytes) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF ingestion") from exc

    reader = PdfReader(io.BytesIO(data))
    num_pages = len(reader.pages)
    if num_pages > MAX_PDF_PAGES:
        raise ValueError(
            f"PDF has {num_pages} pages which exceeds the {MAX_PDF_PAGES}-page limit."
        )
    pages: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"content": text, "metadata": {"page": i + 1}})
    return pages


def parse_bytes(filename: str, data: bytes) -> list[dict[str, Any]]:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "json":
        return _parse_json(data)
    if ext == "txt":
        return _parse_txt(data)
    if ext == "csv":
        return _parse_csv(data)
    if ext == "pdf":
        return _parse_pdf(data)
    raise ValueError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _is_atomic(item: dict[str, Any]) -> bool:
    """Q&A pairs and rate rows are kept whole."""
    meta = item.get("metadata", {})
    return bool(meta.get("rate") or meta.get("category") or meta.get("product"))


def _char_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def chunk_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        content = item.get("content", "").strip()
        if not content:
            continue
        meta = item.get("metadata", {})
        if _is_atomic(item) or len(content) <= CHUNK_CHARS:
            result.append({"content": content, "metadata": meta})
        else:
            for chunk in _char_chunks(content, CHUNK_CHARS, CHUNK_OVERLAP):
                if chunk.strip():
                    result.append({"content": chunk.strip(), "metadata": meta})
    return result


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _sanitize_content(text: str) -> str:
    """Strip dangerous/invisible characters from document content before storing."""
    return _strip_dangerous_chars(text)


def ingest_items(
    items: list[dict[str, Any]],
    source_kind: str = "upload",
) -> dict[str, int]:
    """Chunk → embed in batches → upsert. Returns {"added": n, "skipped": n}."""
    # Sanitize content of each item before chunking
    sanitized_items = [
        {**item, "content": _sanitize_content(item.get("content", ""))}
        for item in items
    ]
    chunks = chunk_items(sanitized_items)
    if not chunks:
        return {"added": 0, "skipped": 0}

    total_added = 0
    total_skipped = 0

    # Embed in batches
    for batch_start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[batch_start: batch_start + EMBED_BATCH]
        texts = [c["content"] for c in batch]
        embeddings = embed_text(texts)

        rows = []
        for chunk, embedding in zip(batch, embeddings):
            rows.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding,
                "content_hash": content_hash(chunk["content"]),
                "source_kind": source_kind,
            })

        result = upsert_documents(rows)
        total_added += result["added"]
        total_skipped += result["skipped"]

    return {"added": total_added, "skipped": total_skipped}


def ingest_bytes(
    filename: str,
    data: bytes,
    source_kind: str = "upload",
) -> dict[str, int]:
    """Full pipeline from raw file bytes."""
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise ValueError(f"File exceeds maximum size of {settings.MAX_UPLOAD_MB} MB")
    items = parse_bytes(filename, data)
    return ingest_items(items, source_kind=source_kind)
