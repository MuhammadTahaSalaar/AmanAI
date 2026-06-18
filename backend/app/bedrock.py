"""Thin boto3 clients for Bedrock: embed, generate, rerank, apply_guardrail."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from .config import get_settings

logger = logging.getLogger(__name__)

_THROTTLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}


@lru_cache(maxsize=1)
def _session() -> boto3.Session:
    settings = get_settings()
    return boto3.Session(region_name=settings.AWS_REGION)


def _bedrock_runtime():
    return _session().client("bedrock-runtime")


def _bedrock_agent_runtime():
    return _session().client("bedrock-agent-runtime")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Titan v2. Returns a list of 1024-dim vectors."""
    settings = get_settings()
    client = _bedrock_runtime()
    embeddings: list[list[float]] = []
    for text in texts:
        body = json.dumps({
            "inputText": text,
            "dimensions": settings.EMBED_DIM,
            "normalize": True,
        })
        try:
            resp = client.invoke_model(
                modelId=settings.BEDROCK_EMBED_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            data = json.loads(resp["body"].read())
            embeddings.append(data["embedding"])
        except ClientError as exc:
            logger.error("embed_text failed: %s", exc)
            raise
    return embeddings


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(messages: list[dict]) -> str:
    """
    Call Bedrock converse with the primary model; fall back to the cheaper
    model on throttling / availability errors.
    Attaches guardrail if BEDROCK_GUARDRAIL_ID is set.
    """
    settings = get_settings()
    client = _bedrock_runtime()

    kwargs: dict = {"messages": messages}
    if settings.BEDROCK_GUARDRAIL_ID:
        kwargs["guardrailConfig"] = {
            "guardrailIdentifier": settings.BEDROCK_GUARDRAIL_ID,
            "guardrailVersion": settings.BEDROCK_GUARDRAIL_VERSION,
            "trace": "enabled",
        }

    def _call(model_id: str) -> str:
        resp = client.converse(modelId=model_id, **kwargs)
        return resp["output"]["message"]["content"][0]["text"]

    try:
        return _call(settings.BEDROCK_GEN_MODEL_ID)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in _THROTTLE_CODES:
            logger.warning("Primary model throttled (%s), falling back", code)
            return _call(settings.BEDROCK_GEN_MODEL_ID_FALLBACK)
        raise


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------

def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """
    Cohere rerank via bedrock-agent-runtime.
    Returns list of (original_index, score) sorted by score desc.
    """
    settings = get_settings()
    client = _bedrock_agent_runtime()

    text_sources = [
        {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}}}
        for doc in documents
    ]

    try:
        resp = client.rerank(
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {
                        "modelArn": f"arn:aws:bedrock:{settings.AWS_REGION}::foundation-model/{settings.BEDROCK_RERANK_MODEL_ID}"
                    },
                    "numberOfResults": top_n,
                },
            },
            sources=text_sources,
            query={"type": "TEXT", "textQuery": {"text": query}},
        )
    except ClientError as exc:
        logger.error("rerank failed: %s", exc)
        raise

    results = resp.get("rerankingResults", [])
    return [(r["index"], r["relevanceScore"]) for r in results]


# ---------------------------------------------------------------------------
# Guardrail passthrough
# ---------------------------------------------------------------------------

def apply_guardrail(text: str, source: str) -> tuple[str, str]:
    """
    Apply Bedrock guardrail if configured.
    source: 'INPUT' | 'OUTPUT'
    Returns (action, output_text).
    action is 'NONE' (pass) or 'GUARDRAIL_INTERVENED'.
    """
    settings = get_settings()
    if not settings.BEDROCK_GUARDRAIL_ID:
        return ("NONE", text)

    client = _bedrock_runtime()
    try:
        resp = client.apply_guardrail(
            guardrailIdentifier=settings.BEDROCK_GUARDRAIL_ID,
            guardrailVersion=settings.BEDROCK_GUARDRAIL_VERSION,
            source=source,
            content=[{"text": {"text": text}}],
        )
        action = resp.get("action", "NONE")
        outputs = resp.get("outputs", [])
        output_text = outputs[0]["text"] if outputs else text
        return (action, output_text)
    except ClientError as exc:
        logger.error("apply_guardrail failed: %s", exc)
        raise
