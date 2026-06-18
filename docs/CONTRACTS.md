# AmanAI Rebuild — Shared Contracts (source of truth)

All components MUST conform to these contracts. Region: **us-east-1**.

## Bedrock model IDs (us-east-1)
- Generation (default): `us.meta.llama3-3-70b-instruct-v1:0` (cross-region inference profile)
- Generation (fallback/cheap): `us.meta.llama3-1-8b-instruct-v1:0`
- Embeddings: `amazon.titan-embed-text-v2:0` — **1024 dims**
- Rerank: `cohere.rerank-v3-5:0` via `bedrock-agent-runtime` `rerank` API
- Guardrails: applied via `bedrock-runtime` `apply_guardrail` (and/or inline on `converse`)

## Environment variables (names are fixed)
```
AWS_REGION=us-east-1
BEDROCK_GEN_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
BEDROCK_GEN_MODEL_ID_FALLBACK=us.meta.llama3-1-8b-instruct-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_RERANK_MODEL_ID=cohere.rerank-v3-5:0
BEDROCK_GUARDRAIL_ID=
BEDROCK_GUARDRAIL_VERSION=DRAFT
EMBED_DIM=1024
RETRIEVE_K=8
RERANK_TOP_N=4
MIN_RERANK_SCORE=0.30           # refusal floor; calibrated in P6
MAX_INPUT_CHARS=2000
MAX_UPLOAD_MB=5
SUPABASE_URL=
SUPABASE_SERVICE_KEY=           # prod: from AWS Secrets Manager
COGNITO_USER_POOL_ID=
COGNITO_CLIENT_ID=
COGNITO_REGION=us-east-1
ALLOWED_ORIGINS=http://localhost:3000   # plus the Vercel origin
LOG_LEVEL=INFO
```

## Supabase / Postgres schema (pgvector)
Table `documents`:
- `id` bigint identity PK
- `content` text NOT NULL
- `metadata` jsonb NOT NULL DEFAULT '{}'
- `embedding` vector(1024)
- `fts` tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
- `content_hash` text UNIQUE  (sha256 of normalized content; dedup key)
- `source_kind` text DEFAULT 'seed'  ('seed' | 'upload')
- `created_at` timestamptz DEFAULT now()

Indexes: HNSW on `embedding` (`vector_cosine_ops`); GIN on `fts`.

RPC `match_documents(query_embedding vector(1024), query_text text, match_count int, rrf_k int DEFAULT 60)`
→ returns `table(id bigint, content text, metadata jsonb, score float)`.
Hybrid: rank docs by vector cosine distance AND by `ts_rank_cd(fts, websearch_to_tsquery('english', query_text))`; fuse with **RRF**: `score = sum(1.0/(rrf_k + rank))` across the two rankings; order by score desc; limit `match_count`.

Document metadata keys seen in seed data (preserve): `product, category, source_sheet, effective_date, rate, payment_frequency, tenor, currency, payout, source, type`.

## REST API (FastAPI; prod behind API Gateway HTTP API + Cognito JWT authorizer)
All JSON unless noted. Auth = Cognito access/ID token in `Authorization: Bearer <jwt>`.

### `GET /health` → `200`
`{ "status": "ok", "model": "<gen model id>", "docs": <int|null> }`  (no auth)

### `POST /chat`  (auth: any authenticated user)
Request:
```json
{ "message": "string (<= MAX_INPUT_CHARS)",
  "history": [ { "role": "user|assistant", "content": "string" } ] }
```
Response `200`:
```json
{ "answer": "string",
  "citations": [ { "content": "string", "product": "string|null", "source": "string|null" } ],
  "refused": false }
```
Behavior: sanitize+guardrail input → embed query (Titan) → `match_documents` (RETRIEVE_K) → Cohere rerank to RERANK_TOP_N → if top rerank score < MIN_RERANK_SCORE → `refused:true` with helpline redirect (no LLM call) → else build grounded prompt (context + recent history) → Llama generate → guardrail output → return answer + citations from the reranked context.

### `POST /documents`  (auth: **admin group only**)
Accepts `multipart/form-data` file (`.json`,`.txt`,`.csv`,`.pdf`, <= MAX_UPLOAD_MB) OR JSON `{ "documents": [ { "content": "...", "metadata": {...} } ] }`.
Response `200`: `{ "added": <int>, "skipped": <int>, "message": "string" }`
Behavior: parse → validate (non-empty, size) → chunk (keep atomic Q&A/rate items whole; free text ~600 tok / 100 overlap) → Titan embed → upsert into `documents` with `content_hash` dedup and `source_kind='upload'`. Persistent + immediately searchable.

Errors (all endpoints): `{ "detail": "message" }` with appropriate 4xx/5xx. Validate at boundary; never echo secrets.

## Auth contract
- Cognito User Pool; group `admin` grants `/documents`.
- Prod: API Gateway HTTP API **JWT authorizer** (issuer = `https://cognito-idp.us-east-1.amazonaws.com/<pool id>`, audience = app client id) validates the token before Lambda.
- Backend ALSO verifies the JWT locally (for `sam local`/uvicorn) and reads `cognito:groups` to enforce admin. Use the pool JWKS.

## Frontend contract (Next.js)
- Calls the above API with the Cognito JWT. Reads config from `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_COGNITO_*`.
- Pages/flows: sign-in/sign-up (Cognito), chat (history, citations, loading, refusal state), admin-only upload panel.
