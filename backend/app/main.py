"""FastAPI application entry point."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum

from .auth import require_admin, require_user
from .chat import handle_chat
from .config import get_settings
from .db import count_documents
from .ingest import ingest_bytes, ingest_items
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentsJsonRequest,
    HealthResponse,
    IngestResponse,
)

settings = get_settings()

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="AmanAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    doc_count: int | None
    try:
        doc_count = count_documents()
    except Exception:
        doc_count = None
    return HealthResponse(
        status="ok",
        model=settings.BEDROCK_GEN_MODEL_ID,
        docs=doc_count,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _claims: dict = Depends(require_user),
) -> ChatResponse:
    if len(request.message) > settings.MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Message exceeds {settings.MAX_INPUT_CHARS} characters",
        )
    try:
        return handle_chat(request)
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later.",
        ) from exc


@app.post("/documents", response_model=IngestResponse)
async def documents(
    request: Request,
    _claims: dict = Depends(require_admin),
    file: UploadFile | None = File(default=None),
) -> IngestResponse:
    content_type = request.headers.get("content-type", "")

    try:
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

        if file is not None:
            # Multipart file upload — enforce size limit before full read
            allowed_exts = {"json", "txt", "csv", "pdf"}
            ext = (file.filename or "").rsplit(".", 1)[-1].lower()
            if ext not in allowed_exts:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unsupported file type: .{ext}",
                )
            # Read with cap: abort as soon as we exceed the limit
            chunks: list[bytes] = []
            total = 0
            async for chunk in file:
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_MB} MB",
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
            result = ingest_bytes(file.filename or f"upload.{ext}", data)
        elif "application/json" in content_type:
            # JSON body — enforce content-length check upfront
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request body exceeds maximum size of {settings.MAX_UPLOAD_MB} MB",
                )
            body = await request.json()
            doc_request = DocumentsJsonRequest(**body)
            items = [
                {"content": d.content, "metadata": d.metadata}
                for d in doc_request.documents
            ]
            result = ingest_items(items)
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provide a multipart file or JSON body with 'documents'",
            )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Ingest validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid document format. Please check your input.",
        ) from exc
    except Exception as exc:
        logger.error("Ingest error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingestion failed. Please try again later.",
        ) from exc

    return IngestResponse(
        added=result["added"],
        skipped=result["skipped"],
        message=f"Ingested {result['added']} document(s); {result['skipped']} duplicate(s) skipped.",
    )


# Lambda handler
handler = Mangum(app)
