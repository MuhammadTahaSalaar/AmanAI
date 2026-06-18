# AmanAI — NUST Bank RAG Assistant (Serverless Rebuild)

A serverless, scale-to-zero RAG banking assistant for NUST Bank (Pakistan). The production rebuild replaces the original Llama 3.2 3B / Streamlit / ChromaDB system with a fully managed AWS stack that runs for months inside $120 of AWS credits.

> **Legacy notice:** `app.py`, `config.py`, `src/`, and `scripts/` at the repo root are the original course submission (Streamlit + ChromaDB + QLoRA). They are kept as a documented artifact but are **not part of the production system**. The production code lives in `backend/`, `frontend/`, `infra/`, and `ingestion/`.

---

## What it does

- Answers NUST Bank customer questions (savings rates, term deposits, IBFT, mortgages, remittances, Mastercard, etc.) grounded in a curated product knowledge base.
- Refuses off-domain questions (weather, politics, coding) and redacts PII (CNIC, account numbers, PINs) via Bedrock Guardrails.
- Lets authenticated admin users upload new product documents (JSON, TXT, CSV, PDF) that are immediately searchable.
- Provides source citations with every answer so the user can verify the grounding.

---

## Architecture

```
Browser
  │  (Next.js on Vercel)
  ▼
AWS Cognito                  ← sign-up / sign-in / JWT issuance
  │  JWT (Bearer token)
  ▼
API Gateway HTTP API (v2)    ← Cognito JWT authorizer, CORS, throttle
  │  /health  (no auth)
  │  /chat    (any authenticated user)
  │  /documents (admin Cognito group only)
  ▼
AWS Lambda — FastAPI + Mangum (Python 3.12, arm64, 1 024 MB, 30 s)
  ├── Amazon Bedrock Titan Embeddings v2   (embed query + documents, 1024-dim)
  ├── Amazon Bedrock Cohere Rerank 3.5     (rerank top-8 → top-4)
  ├── Amazon Bedrock Llama 3.3 70B         (generation, cross-region inference profile)
  ├── Amazon Bedrock Guardrails            (PII mask, denied topics, prompt-attack, contextual grounding)
  └── Supabase pgvector                   (hybrid FTS + vector search via RRF, persistent storage)

AWS Secrets Manager          ← SUPABASE_SERVICE_KEY
AWS Budgets                  ← $20/month alert
```

**RAG pipeline per query:** sanitize input → Guardrail input check → Titan embed → `match_documents` hybrid SQL (pgvector cosine + Postgres FTS, RRF fusion, k=8) → Cohere Rerank 3.5 (top 4) → if top score < 0.30 refuse with helpline redirect → else Llama 3.3 70B generate grounded answer → Guardrail output check → return answer + citations.

---

## Monorepo layout

```
AmanAI/
├── backend/          FastAPI application (app/, sql/, tests/)
├── frontend/         Next.js 14 App Router chat UI
├── infra/            AWS SAM template + deploy.sh
├── ingestion/        Seed script (embed + upsert 358 product docs)
├── evaluation/       RAGAS harness + golden dataset (40 Q&A pairs)
├── data/             processed/ (seed corpus JSON) + raw dataset/
├── docs/
│   ├── CONTRACTS.md  Shared API + schema + env-var contracts (source of truth)
│   ├── DEPLOYMENT.md Step-by-step go-live runbook  ← start here to deploy
│   └── COST.md       Per-query cost model + monthly estimates
│
│   ── LEGACY (original course submission, not used in production) ──
├── src/              ChromaDB / BM25 / FlashRank / Presidio RAG engine
├── app.py            Streamlit entry point
├── config.py         Old centralized config
└── scripts/          SLURM fine-tuning helpers
```

---

## Quick start — local development

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Copy and fill in env vars (Supabase URL/key, Cognito IDs, AWS region)
cp .env.example .env

uvicorn backend.app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs

# Run tests
python -m pytest -q
```

### Frontend

```bash
cd frontend
cp .env.example .env.local   # fill NEXT_PUBLIC_* vars
npm install
npm run dev                  # http://localhost:3000
```

### Seed the corpus (once, after Supabase schema is applied)

```bash
python -m ingestion.seed
```

---

## Deployment

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for the full ordered runbook (prerequisites → Bedrock model access → Supabase setup → `sam deploy` → Vercel).

---

## Cost

See **[docs/COST.md](docs/COST.md)**. Short version: ~$0.003–0.005 per query; idle cost ≈ $0 (scale-to-zero); < $5/month at student load; $120 AWS credits last 6–12+ months.

---

## Contracts

All components (backend, frontend, infra) share fixed API shapes, environment variable names, Bedrock model IDs, and the Supabase schema. **[docs/CONTRACTS.md](docs/CONTRACTS.md)** is the single source of truth — do not change names without updating all consumers.
