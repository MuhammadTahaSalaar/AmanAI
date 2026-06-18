"""Supabase client wrapper for document operations."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from supabase import create_client, Client

from .config import get_settings
from .secrets import resolve_supabase_key

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> Client:
    settings = get_settings()

    if not settings.SUPABASE_URL:
        raise RuntimeError(
            "SUPABASE_URL is not configured. Set it in environment or template.yaml."
        )

    key = resolve_supabase_key(
        service_key=settings.SUPABASE_SERVICE_KEY,
        secret_arn=settings.SUPABASE_SECRET_ARN,
        region=settings.AWS_REGION,
    )

    return create_client(settings.SUPABASE_URL, key)


def match_documents(
    query_embedding: list[float],
    query_text: str,
    k: int,
) -> list[dict[str, Any]]:
    """Call the match_documents RPC and return rows."""
    try:
        resp = (
            _client()
            .rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "query_text": query_text,
                    "match_count": k,
                },
            )
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error("match_documents RPC failed: %s", exc)
        raise


def upsert_documents(rows: list[dict[str, Any]]) -> dict[str, int]:
    """
    Upsert rows into documents table using content_hash for dedup.
    Returns {"added": n, "skipped": n}.
    """
    if not rows:
        return {"added": 0, "skipped": 0}

    added = 0
    skipped = 0
    for row in rows:
        try:
            resp = (
                _client()
                .table("documents")
                .upsert(row, on_conflict="content_hash", ignore_duplicates=True)
                .execute()
            )
            if resp.data:
                added += len(resp.data)
            else:
                skipped += 1
        except Exception as exc:
            logger.error("upsert_documents error on row hash=%s: %s", row.get("content_hash"), exc)
            raise

    return {"added": added, "skipped": skipped}


def count_documents() -> int:
    """Return total number of rows in the documents table."""
    try:
        resp = _client().table("documents").select("id", count="exact").execute()
        return resp.count or 0
    except Exception as exc:
        logger.error("count_documents failed: %s", exc)
        return 0
