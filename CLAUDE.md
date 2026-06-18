# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AmanAI is a RAG banking assistant for **NUST Bank** (a fictional Pakistani bank),
rebuilt as a **cheap, serverless, scale-to-zero** system. The legacy 3B/Streamlit
prototype (Llama-3.2-3B 4-bit + ChromaDB/BM25, GPU-bound, Streamlit UI, QLoRA
fine-tuning) was **removed** — it is recoverable only from git history. Do **not**
reintroduce `src/`, `app.py`, `config.py`, `data/lora_adapter`, or any Streamlit/
ChromaDB/fine-tuning code.

**Per-query flow:** user → local sanitize + Bedrock guardrail → Titan embed query →
Supabase `match_documents` hybrid search (RETRIEVE_K=8) → Cohere rerank to top
RERANK_TOP_N=4 → if top score < MIN_RERANK_SCORE (0.30) refuse with a helpline
redirect (no LLM call) → else grounded Llama generate → output guardrail → answer +
citations.

**Stack:** FastAPI on AWS Lambda (Mangum) behind API Gateway HTTP API + Cognito JWT
authorizer; Amazon Bedrock (Llama 3.3 70B default / 3.1 8B fallback, Titan Text
Embeddings v2, Cohere Rerank 3.5, Bedrock Guardrails); Supabase pgvector (hybrid
vector + FTS via RRF in SQL); Next.js + Tailwind on Vercel; AWS Cognito auth with an
`admin` group. **Region: us-east-1.** Fine-tuning is dropped.

> **Source of truth:** [`docs/CONTRACTS.md`](docs/CONTRACTS.md) is the binding contract
> for model IDs, env-var names, the DB schema, and API shapes. Read it before changing
> any interface. To deploy, follow [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Cost model
> is in [`docs/COST.md`](docs/COST.md).

## Commands

All commands run **from the repo root** unless noted. The backend venv lives at
`backend/.venv`.

```bash
# One-time backend env
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements-dev.txt   # dev (tests); runtime: requirements.txt

# Backend tests (fully mocked — no network/AWS/Supabase needed)
backend/.venv/bin/python -m pytest backend/tests -q             # expect: 56 passed
backend/.venv/bin/python -m pytest backend/tests/test_chat.py -q            # one module
backend/.venv/bin/python -m pytest backend/tests/test_chat.py::test_name -q # one test

# Run the API locally (needs a .env with Supabase + AWS creds for live calls)
backend/.venv/bin/uvicorn backend.app.main:app --reload --port 8000

# Seed the vector store: data/processed/all_documents.json (358 docs) → Titan embed → Supabase
backend/.venv/bin/python -m ingestion.seed                      # safe to re-run (content_hash dedup)
backend/.venv/bin/python -m ingestion.seed --path /custom/docs.json

# Evaluation (after deploy + seed — drives the live pipeline)
backend/.venv/bin/python -m evaluation.evaluate                 # RAGAS → evaluation/evaluation_results.json
backend/.venv/bin/python -m evaluation.calibrate_threshold      # tune MIN_RERANK_SCORE

# Frontend
cd frontend && npm install && npm run dev                       # dev server on :3000
cd frontend && npm run build                                    # production build (next build)
cd frontend && npx tsc --noEmit                                 # type-check

# Deploy (see docs/DEPLOYMENT.md for the full guided walkthrough)
cd infra && bash deploy.sh                                      # sam build + sam deploy --guided
```

There is **no corpus-build script** in the rebuild: `ingestion/seed.py` loads a
pre-built `data/processed/all_documents.json` (a `[{content, metadata}]` array; the
file is gitignored but ships in the working tree). The Supabase schema/RPC is applied
once by running `backend/sql/schema.sql` in the Supabase SQL editor.

## Configuration model

Backend settings live in `backend/app/config.py` (`Settings`, pydantic-settings) — the
single source of truth for tunables, read once via `get_settings()` (lru-cached). Every
value is overridable by an environment variable; locally they load from a repo-root
`.env`. **No magic numbers in business logic** — read from `get_settings()`. In
production the SAM template (`infra/template.yaml`) sets these env vars on the Lambda;
`SUPABASE_SERVICE_KEY` is **not** an env var in prod — it is resolved at runtime from
AWS Secrets Manager (`backend/app/secrets.py`) via `SUPABASE_SECRET_ARN`. The frontend
reads only `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_COGNITO_REGION`,
`NEXT_PUBLIC_COGNITO_USER_POOL_ID`, `NEXT_PUBLIC_COGNITO_CLIENT_ID`. Env-var names are
fixed by `docs/CONTRACTS.md` — change them there first.

## Architecture

### Backend (`backend/app/`) — FastAPI, dependency-injected, deployed to Lambda via Mangum
- `main.py` — builds the module-level `app = FastAPI(...)`, CORS middleware, and the
  routes (`/health`, `/chat`, `/documents`); `handler = Mangum(app)` is the Lambda entry
  point (`app.main.handler` in the SAM template).
- `config.py` — `Settings` / `get_settings()` (see Configuration model).
- `secrets.py` — resolves `SUPABASE_SERVICE_KEY` from Secrets Manager at runtime.
- `bedrock.py` — all Amazon Bedrock calls: Titan embeddings, Llama generation
  (Converse, with the 8B fallback), Cohere `rerank`, and `apply_guardrail`.
- `db.py` — Supabase client + the `match_documents` RPC (hybrid RRF search).
- `retrieval.py` — embed → `match_documents` (RETRIEVE_K) → Cohere rerank (RERANK_TOP_N)
  → refusal floor at MIN_RERANK_SCORE.
- `guardrails.py` — local input sanitize/bounds + Bedrock Guardrail (PII mask, denied
  topics, prompt-attack, contextual grounding).
- `chat.py` — `handle_chat`: orchestrates the full per-query flow above; returns
  `answer` + `citations` + `refused`. The eval harness calls this in-process.
- `ingest.py` — `content_hash` + `ingest_items`: parse → validate → chunk (atomic Q&A/
  rate items kept whole) → Titan embed → upsert with `content_hash` dedup and
  `source_kind`. Shared by `ingestion/seed.py` (seed) and `POST /documents` (upload).
- `auth.py` — Cognito JWT verification (issuer/audience/`token_use`) and `admin`-group
  enforcement for `/documents` (defence-in-depth behind the API Gateway JWT authorizer).
- `schemas.py` — Pydantic request/response models matching `docs/CONTRACTS.md`.
- `prompts.py` — grounded prompt templates.

### Data layer — Supabase (`backend/sql/schema.sql`)
`documents` table: `content`, `metadata jsonb`, `embedding vector(1024)`, `fts`
(generated tsvector), `content_hash` UNIQUE, `source_kind` (`seed`|`upload`). HNSW index
on `embedding`, GIN on `fts`. The `match_documents(query_embedding, query_text,
match_count, rrf_k)` RPC ranks by vector cosine distance and FTS rank, then fuses with
**Reciprocal Rank Fusion**.

### Ingestion (`ingestion/seed.py`)
Loads `data/processed/all_documents.json` and calls `backend.app.ingest.ingest_items`
with `source_kind="seed"`.

### Frontend (`frontend/`) — Next.js 14 + Tailwind, Cognito via aws-amplify v6
- `lib/amplify.ts` — `Amplify.configure` for the Cognito user pool (SRP flow).
- `lib/auth.ts` — sign-in/up, token retrieval (`getIdToken`).
- `lib/api.ts` — typed `getHealth` / `postChat` / `uploadDocument`; sends the Cognito
  JWT as `Authorization: Bearer`; reads `NEXT_PUBLIC_API_BASE_URL`.
- App pages: sign-in/up, chat (history, citations, loading, refusal state), admin-only
  upload panel. Config is `next.config.mjs` (Next 14 cannot load a TS `next.config.ts`).

### Infra (`infra/`) — AWS SAM
`template.yaml` defines ~11 resources: Cognito user pool + `admin` group + Hosted UI
domain, Secrets Manager secret, Bedrock Guardrail (+ version), least-privilege Lambda
IAM role, HTTP API (Cognito JWT authorizer, CORS, throttling burst 20 / rate 10), the
Lambda function, and a $20/month Budget. `deploy.sh` wraps `sam build` +
`sam deploy --guided` and prints the post-deploy values. `samconfig.toml` saves
parameters. Stack name `amanai`, region `us-east-1`.

### Evaluation (`evaluation/`)
`evaluate.py` runs the 40-pair `golden_dataset.json` through `chat.handle_chat`
in-process and scores non-refused answers with RAGAS on a Bedrock judge.
`calibrate_threshold.py` sweeps `MIN_RERANK_SCORE` to maximise in-domain vs
out-of-domain separation. See `evaluation/README.md`.

## Conventions

- Python 3.12, FastAPI. `from __future__ import annotations` where `X | None` syntax is
  used. Type-annotate signatures; follow PEP 8.
- Components use **constructor/dependency injection** (optional collaborator args
  defaulting to real implementations) — this is the seam the tests mock. Mirror it for
  new components; never hard-code clients inside route handlers.
- Validate at the boundary (Pydantic schemas); never echo secrets in errors or logs.
  The raw user message is sanitized before it reaches the LLM or storage — preserve that
  invariant in `chat.py`/`guardrails.py`.
- Tests live in `backend/tests/`, mirror `backend/app/` module names, and are fully
  mocked (no network). Run from the repo root: `backend/.venv/bin/python -m pytest
  backend/tests -q` (56 passing).
- Do not change model IDs, env-var names, the DB schema, or API shapes without updating
  `docs/CONTRACTS.md` first.
