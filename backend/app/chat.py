"""Chat orchestration: guardrail → retrieve → refusal gate → generate → guardrail out."""
from __future__ import annotations

import logging
from typing import Any

from .bedrock import apply_guardrail, generate
from .config import get_settings
from .guardrails import validate_input, _strip_dangerous_chars, _has_jailbreak
from .prompts import REFUSAL_MESSAGE, build_messages
from .retrieval import retrieve
from .schemas import ChatRequest, ChatResponse, Citation, HistoryTurn

logger = logging.getLogger(__name__)

_SAFE_ROLES = {"user", "assistant"}


def _sanitize_history(history: list[HistoryTurn]) -> list[HistoryTurn]:
    """Strip dangerous chars from each history turn; drop turns that fail jailbreak check."""
    clean: list[HistoryTurn] = []
    for turn in history:
        if turn.role not in _SAFE_ROLES:
            logger.warning("Dropping history turn with invalid role=%r", turn.role)
            continue
        sanitized = _strip_dangerous_chars(turn.content)
        if _has_jailbreak(sanitized):
            logger.warning("Dropping history turn with jailbreak content (role=%s)", turn.role)
            continue
        clean.append(HistoryTurn(role=turn.role, content=sanitized))
    return clean


def handle_chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()

    # 1. Local guardrail — sanitize input
    ok, cleaned, reason = validate_input(request.message)
    if not ok:
        logger.warning("Input rejected by local guardrail: %s", reason)
        return ChatResponse(
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
        )

    # 2. Bedrock guardrail on input (no-op if not configured)
    action, guarded_input = apply_guardrail(cleaned, "INPUT")
    if action != "NONE":
        logger.warning("Bedrock guardrail intervened on input (action=%s)", action)
        return ChatResponse(
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
        )

    # 3. Retrieve
    reranked_docs, top_score = retrieve(guarded_input)

    # 4. Refusal gate
    if top_score < settings.MIN_RERANK_SCORE:
        logger.info("Refusal gate triggered: top_score=%.3f < %.3f", top_score, settings.MIN_RERANK_SCORE)
        return ChatResponse(
            answer=REFUSAL_MESSAGE,
            citations=[],
            refused=True,
        )

    # 5. Sanitize history, then build prompt and generate
    safe_history = _sanitize_history(list(request.history))
    messages = build_messages(guarded_input, reranked_docs, safe_history)
    raw_answer = generate(messages)

    # 6. Bedrock guardrail on output
    out_action, final_answer = apply_guardrail(raw_answer, "OUTPUT")
    if out_action != "NONE":
        logger.warning("Bedrock guardrail intervened on output (action=%s)", out_action)
        final_answer = REFUSAL_MESSAGE

    # 7. Build citations from reranked docs
    citations = _build_citations(reranked_docs)

    return ChatResponse(
        answer=final_answer,
        citations=citations,
        refused=False,
    )


def _build_citations(docs: list[dict[str, Any]]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for doc in docs:
        content = doc.get("content", "")
        key = content[:120]
        if key in seen:
            continue
        seen.add(key)
        meta = doc.get("metadata", {})
        citations.append(Citation(
            content=content,
            product=meta.get("product") or None,
            source=meta.get("source") or meta.get("source_sheet") or None,
        ))
    return citations
