# AmanAI Backend

FastAPI + AWS Bedrock + Supabase serverless RAG banking assistant.

## Local development

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Copy `.env.example` (or set env vars directly):

```
AWS_REGION=us-east-1
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>
COGNITO_USER_POOL_ID=us-east-1_XXXXX
COGNITO_CLIENT_ID=<client-id>
BEDROCK_GUARDRAIL_ID=           # optional
ALLOWED_ORIGINS=http://localhost:3000
```

Start the server:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

## Run tests

```bash
cd backend
python -m pytest -q
```

## Seed the database

```bash
python -m ingestion.seed
# or with custom path:
python -m ingestion.seed --path data/processed/all_documents.json
```

## Schema setup

Run `backend/sql/schema.sql` against your Supabase project once:

```bash
psql "$SUPABASE_DB_URL" -f backend/sql/schema.sql
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | None | Health check + doc count |
| POST | /chat | User | RAG chat |
| POST | /documents | Admin | Upload documents |
