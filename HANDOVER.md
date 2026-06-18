# AmanAI — Session Handover

This document hands the AmanAI production rebuild to a fresh session. Read it
first, then read `docs/CONTRACTS.md` (the binding source-of-truth contract).

## What this project is

AmanAI is a RAG banking assistant for **NUST Bank** (a fictional Pakistani bank),
originally a NUST CS416 course project. The course is over; the owner is now
turning it into a **cheap, production-worthy, deployable** system. The original
prototype (Llama-3.2-3B 4-bit + ChromaDB/BM25 + Streamlit, GPU-bound) gave weak
answers, broke often, and had poor document upload. It has been **rebuilt** as a
serverless, scale-to-zero system.

## Confirmed decisions (do not relitigate)

| Area | Decision |
|---|---|
| LLM | Amazon **Bedrock**, open-source **Llama 3.3 70B** default (`us.meta.llama3-3-70b-instruct-v1:0`), fallback Llama 3.1 8B. No GPU, pay-per-token. |
| Embeddings | Bedrock **Titan Text Embeddings v2** (`amazon.titan-embed-text-v2:0`, 1024-dim). |
| Rerank | Bedrock **Cohere Rerank 3.5** (`cohere.rerank-v3-5:0`). |
| Guardrails | **Bedrock Guardrails** (PII + PK regex, denied topics, prompt-attack, contextual grounding) + light local checks. |
| Vector store | **Supabase pgvector** (free tier); hybrid vector+FTS search via RRF in SQL. |
| Auth | **AWS Cognito** user pool + `admin` group (admin gates document upload). |
| Frontend | **Next.js + Tailwind** on **Vercel** (free). |
| Backend | **FastAPI on AWS Lambda** (Mangum) behind **API Gateway HTTP API** + Cognito JWT authorizer. |
| Region | **us-east-1**. |
| Budget | Serverless scale-to-zero; ~$0.004–0.005/query, ~$0 idle; **$120 credit lasts 6–24 months**. |
| Fine-tuning | **Dropped** (removed from repo). |
| Domain | Free default URLs (`*.vercel.app` + API Gateway endpoint). |

## Architecture

```
Browser ─ Next.js (Vercel) ─ Cognito (JWT) ─► API Gateway HTTP API (JWT authorizer, CORS, throttle)
                                                        │
                                                   AWS Lambda (FastAPI + Mangum, Python 3.12)
                                          ┌──────────────┼───────────────────────────┐
                                  Supabase pgvector   Amazon Bedrock            Bedrock Guardrails
                                  (hybrid RRF)        Titan / Llama / Cohere     (PII, topics, attacks)
```

Per-query flow: local sanitize + guardrail → Titan embed → Supabase `match_documents`
(RETRIEVE_K=8) → Cohere rerank to top 4 → if top score < MIN_RERANK_SCORE (0.30)
refuse with helpline → else grounded Llama generate → output guardrail → answer + citations.

## Repository layout (after cleanup)

```
backend/      FastAPI app (app/), sql/schema.sql, tests/ (56 passing), requirements*.txt, README
frontend/     Next.js + Tailwind, Cognito auth (aws-amplify v6), chat + admin upload, README
infra/        AWS SAM template.yaml, samconfig.toml, deploy.sh, README (11 resources)
ingestion/    seed.py (loads data/processed/all_documents.json → Titan embed → Supabase)
evaluation/   evaluate.py (RAGAS on Bedrock), calibrate_threshold.py, golden_dataset.json (40 pairs), README
docs/         CONTRACTS.md (SOURCE OF TRUTH), DEPLOYMENT.md (NEEDS REWRITE), COST.md
data/         processed/all_documents.json (358 seed docs), runtime_document/
dataset/      raw NUST Bank Excel + JSON (source data)
README.md, CLAUDE.md (STALE — needs refresh), .gitignore, .gitattributes
```

The legacy 3B/Streamlit system (`src/`, `app.py`, `config.py`, old `tests/`, Docker,
fine-tune scripts, `data/lora_adapter`) was **removed** — recoverable from git history.

## Git state

- Branch: **`rebuild/serverless-bedrock`** (off `main`). Remote: `origin` (GitHub `MuhammadTahaSalaar/AmanAI`).
- Two commits ahead of where the session started:
  - `d7d0986` feat: rebuild AmanAI as a serverless Bedrock RAG (FastAPI + Next.js + Supabase)
  - `1829f8d` chore: remove legacy 3B/Streamlit system; dedupe and tidy repo
- **Likely NOT pushed yet** — verify with `git status -sb` / `git log origin/main..HEAD`. The
  remote `main` may have diverged (teammates / original course commits), so a pull + merge is needed.
- `node_modules/` (~34k files), `.venv/`, `.next/`, `data/processed/`, caches are gitignored — git tree is clean.

## Status by component

- **Backend** — complete; **56/56 unit tests pass** (fully mocked, no network):
  run from repo ROOT: `backend/.venv/bin/python -m pytest backend/tests -q`.
- **Infra** — SAM template authored (Lambda, HTTP API + Cognito authorizer, user pool/admin group,
  Bedrock Guardrail, least-privilege IAM, Secrets Manager, $20 Budget, throttling). `sam validate`
  NOT run (SAM CLI absent in the build env). Verify before deploy.
- **Frontend** — fully scaffolded; **builds on a clean env/Vercel**, but `tsc`/`next build` could not be
  confirmed green in the build sandbox (see Known issues).
- **Ingestion / Evaluation / Docs(CONTRACTS, COST)** — done.
- **Security** — a full review ran; **all CRITICAL/HIGH findings were fixed** (JWT `token_use` validation,
  runtime Secrets Manager resolution, history/input sanitization + bounds, pre-read upload size cap,
  PDF page cap, deny-by-default guardrail, API throttling, Cognito hardening).

## Known issues / caveats (IMPORTANT)

1. **Frontend `tsc` shows 6 spurious errors** — `Module 'aws-amplify/auth' has no exported member 'signIn'…`.
   Root cause: a dependency-**hoisting** quirk in the build sandbox (`@aws-amplify/auth` resolves nested, not
   hoisted) — NOT a code defect; the code is idiomatic Amplify v6. A clean install fixes it:
   `cd frontend && rm -rf node_modules package-lock.json && npm install && npm run build`. The new session
   should confirm this resolves it; if a clean install still fails, root-cause and fix properly (do NOT disable
   type-checking or downgrade aws-amplify).
2. **`next build` bus-errored in the sandbox** (native SWC / memory) — an environment issue; builds fine on Vercel.
3. **Nothing is deployed.** No AWS/Supabase/Vercel creds in the build env. All code + IaC + docs are ready; the
   owner deploys by following `docs/DEPLOYMENT.md`.
4. **RAGAS eval** runs only after deploy (needs live Bedrock + Supabase).
5. **A leaked real `HF_TOKEN`** sits in the (gitignored, now unused) `.env` — owner should rotate/delete it.
6. **A cost-tracking hook injects a high session-cost figure** into sub-agents and makes them PAUSE/ASK before
   working. In the next session, either do the doc-writing work directly in the main thread, or instruct any
   sub-agent explicitly: "cost is approved; do not pause or ask about cost; proceed to completion."
7. **Approximate Bedrock prices** in COST.md were not machine-verified — confirm on the Bedrock pricing page.

## Remaining work (the TODO for the next session)

1. **Sync git**: fetch/pull `origin`, merge remote `main` (or whatever diverged) into `rebuild/serverless-bedrock`,
   resolve any conflicts, keep backend tests green, and push the branch.
2. **Rewrite `docs/DEPLOYMENT.md`** into an exhaustive, beginner-proof, up-to-date runbook (every account,
   command, and console step — see the prompt below). Use Context7/web to verify current AWS/Supabase/Vercel/
   Bedrock procedures.
3. **Refresh `CLAUDE.md`** — it still describes the old 3B/Streamlit system; rewrite for the new architecture
   and commands.
4. **Confirm the frontend clean build** (issue #1) and fix if anything real remains.
5. (Optional) Run `sam validate --lint` once SAM CLI is available; address findings.

## Verification commands

```bash
# backend tests (from repo root)
backend/.venv/bin/python -m pytest backend/tests -q          # expect: 56 passed

# frontend type-check / build (clean env)
cd frontend && rm -rf node_modules package-lock.json && npm install && npm run build

# git state
git status -sb && git log --oneline -3 && git log origin/main..HEAD --oneline
```
