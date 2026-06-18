"""Pydantic request/response models matching the API contract."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    content: str
    product: Optional[str] = None
    source: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    docs: Optional[int] = None


class DocumentItem(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class DocumentsJsonRequest(BaseModel):
    documents: list[DocumentItem]


class IngestResponse(BaseModel):
    added: int
    skipped: int
    message: str
