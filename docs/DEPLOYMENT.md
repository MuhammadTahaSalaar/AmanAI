# AmanAI — Deployment Runbook (beginner edition)

This guide takes you from **nothing** to a **live AmanAI system** on free/cheap
infrastructure. It assumes **you have never used AWS**. Every account, install,
console click, and command is spelled out. Do the steps **top to bottom** — each
one depends on the previous.

> **Source of truth:** the binding contract for models, API shapes, DB schema, and
> env-var names is [`docs/CONTRACTS.md`](./CONTRACTS.md). The real resource names,
> parameters, and outputs used below come from `infra/template.yaml`,
> `infra/deploy.sh`, `infra/samconfig.toml`, `backend/sql/schema.sql`,
> `ingestion/seed.py`, `backend/app/config.py`, and `frontend/lib/*`. If anything
> here ever disagrees with those files, the files win.

> **Last verified: 2026-06-19.** External console flows (AWS Bedrock, Supabase,
> Vercel) change over time — citations to the current official docs are inline.
> The exact button labels may drift slightly; the concepts and CLI commands are stable.

---

## 0. What you are about to build

```
Browser ─ Next.js (Vercel, free) ─ Cognito sign-in (JWT)
                       │  Authorization: Bearer <JWT>
                       ▼
       API Gateway HTTP API (JWT authorizer, CORS, throttling)
                       │
              AWS Lambda (FastAPI + Mangum, Python 3.12, arm64)
        ┌──────────────┼───────────────────────────────┐
   Supabase pgvector   Amazon Bedrock              Bedrock Guardrail
   (hybrid RRF search) Titan v2 / Llama / Cohere    (PII, topics, attacks)
```

**Region for everything AWS: `us-east-1` (N. Virginia).** Do not change it — the
IAM policy, inference-profile ARNs, and Cognito issuer in `infra/template.yaml`
are all pinned to `us-east-1`.

**What it costs:** scale-to-zero serverless. Roughly **$0.004–0.005 per query**,
**~$0 when idle**. A `$20/month` budget alarm is created automatically. With the
$100–$120 of AWS promotional credits most students get, this typically lasts
**6–24 months**. (See [`docs/COST.md`](./COST.md).)

**Time to complete:** ~60–90 minutes the first time, most of it waiting for
Bedrock model-access approval and Supabase/Vercel provisioning.

---

## 1. Create the four accounts

You need four free accounts. Create them now; you'll switch between them later.

| Account | Sign-up link | Why |
|---|---|---|
| **GitHub** | <https://github.com/signup> | Hosts the code; Vercel deploys from it. |
| **AWS** | <https://portal.aws.amazon.com/billing/signup> | Bedrock (LLM), Lambda, API Gateway, Cognito, Secrets Manager, Budgets. |
| **Supabase** | <https://supabase.com/dashboard/sign-up> | Postgres + pgvector vector store (free tier). |
| **Vercel** | <https://vercel.com/signup> | Hosts the Next.js frontend (free, Hobby tier). Sign up **with GitHub** so it can import the repo. |

> **AWS sign-up needs a credit/debit card and a phone number** for verification.
> You will not be charged if you stay inside the free tier + credits + the $20 budget,
> but a card is mandatory to open the account.

### 1a. Get the code onto your machine and into your GitHub

If you already have this repo locally on the `rebuild/serverless-bedrock` branch,
push it to your own GitHub:

```bash
# from the repo root
git remote -v                       # confirm 'origin' points at your GitHub repo
git push -u origin rebuild/serverless-bedrock
```

If you're starting fresh, clone it:

```bash
git clone https://github.com/MuhammadTahaSalaar/AmanAI.git
cd AmanAI
git checkout rebuild/serverless-bedrock
```

---

## 2. Apply your AWS credits and set a budget alarm

### 2a. Apply promotional credits (if you have them)

1. Sign in to the [AWS Console](https://console.aws.amazon.com/).
2. Top-right, click your account name → **Billing and Cost Management**.
3. Left menu → **Credits** → **Redeem credit** → paste your promo code → **Redeem**.
   Credits are consumed automatically before your card is charged.

### 2b. Create a cost budget + email alarm (do this even though SAM also makes one)

The SAM stack later creates a `$20/month` budget, but it only exists **after** you
deploy. Create a manual safety budget **now** so you're protected from the first dollar.
([AWS: Control your costs with Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html))

1. **Billing and Cost Management** → left menu → **Budgets** → **Create budget**.
2. Choose **Use a template (simplified)** → **Monthly cost budget**.
3. **Budgeted amount:** `20` (USD). **Name:** `amanai-manual-guard`.
4. **Email recipients:** your email. → **Create budget**.

You'll get email when actual or forecast spend crosses the threshold.

> Set the top-right **Region selector** to **US East (N. Virginia) us-east-1** now,
> and keep it there for every console step in this guide.

---

## 3. Install the command-line tools

You need four tools locally. Verify each with the version command shown.

### 3a. AWS CLI v2
([Official install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))

```bash
# Linux x86_64
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# macOS: download & run the pkg
#   https://awscli.amazonaws.com/AWSCLIV2.pkg
# Windows: download & run the MSI
#   https://awscli.amazonaws.com/AWSCLIV2.msi

aws --version          # expect: aws-cli/2.x.x ...
```

AWS CLI v2 bundles its own Python, so it does not need a system Python.

### 3b. Configure your AWS credentials

Create an access key first:

1. AWS Console → search **IAM** → **Users** → your user → **Security credentials**
   → **Create access key** → **Command Line Interface (CLI)** → confirm → **Create**.
2. Copy the **Access key ID** and **Secret access key** (the secret is shown once).

> If you only have the AWS **root** user, create an IAM admin user instead
> (IAM → Users → Create user → attach `AdministratorAccess`) and make a key for *that*
> user. Using root keys daily is discouraged.

```bash
aws configure
# AWS Access Key ID     : <paste>
# AWS Secret Access Key : <paste>
# Default region name   : us-east-1
# Default output format  : json

aws sts get-caller-identity   # should print your account ID and user ARN
```

### 3c. Python 3.12 (for SAM build, seeding, and eval)

```bash
python3 --version      # need 3.12.x
```
If you don't have it: [python.org/downloads](https://www.python.org/downloads/) or
`brew install python@3.12` (macOS) / `sudo apt install python3.12 python3.12-venv` (Ubuntu).

### 3d. AWS SAM CLI
([Official install guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html))

> AWS **stopped maintaining the Homebrew formula** for SAM CLI in Sept 2023. Use pip
> (simplest, cross-platform) or the official zip installer.

```bash
# pip (recommended; ideally in a virtualenv)
pip install aws-sam-cli

# OR Linux zip installer:
#   https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip

sam --version          # expect: SAM CLI, version 1.x.x
```

### 3e. Node.js 20+ (for the frontend)
([nodejs.org](https://nodejs.org/en/download)) — install the **20 LTS** or newer.

```bash
node --version         # expect: v20.x or newer
npm --version
```

---

## 4. Enable Bedrock model access (us-east-1)

Bedrock models are **off by default**. You must request access to the four AmanAI uses.
([Amazon Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html))

1. Console (region = **us-east-1**) → search **Bedrock** → left menu → **Model access**.
2. Click **Modify model access** (or **Manage model access**).
3. Enable these four (search each by name and tick it):

   | Provider | Model | Bedrock model ID | Used for |
   |---|---|---|---|
   | Meta | Llama 3.3 70B Instruct | `meta.llama3-3-70b-instruct-v1:0` | Generation (primary) |
   | Meta | Llama 3.1 8B Instruct | `meta.llama3-1-8b-instruct-v1:0` | Generation (fallback/cheap) |
   | Amazon | Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | Embeddings (1024-dim) |
   | Cohere | Rerank 3.5 | `cohere.rerank-v3-5:0` | Reranking |

4. Submit. Most approvals are instant–minutes; some take up to an hour.

> **The `us.` inference-profile requirement (important).** In `us-east-1`, the Llama
> models are invoked through **cross-region inference profiles**, whose IDs carry a
> `us.` prefix: `us.meta.llama3-3-70b-instruct-v1:0` and
> `us.meta.llama3-1-8b-instruct-v1:0`. The backend already uses these IDs
> (`backend/app/config.py`), and the IAM policy in `infra/template.yaml` grants both
> the bare `foundation-model/...` ARNs **and** the `inference-profile/us.meta....`
> ARNs. You still request access to the **base** model in the table above; the profile
> just routes the call to the nearest US region.
> ([AWS: Llama 3.3 70B in Bedrock](https://aws.amazon.com/about-aws/whats-new/2024/12/metas-llama-3-3-70b-model-amazon-bedrock/),
> [Cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html))

If a deploy or query later fails with `AccessDeniedException` mentioning a model,
the cause is almost always an un-approved model here.

---

## 5. Set up Supabase (Postgres + pgvector)

1. [Supabase dashboard](https://supabase.com/dashboard) → **New project**.
   - **Name:** `amanai`  •  **Database password:** generate a strong one and save it
     (you may need it for direct `psql`).  •  **Region:** any (the Lambda talks to it
     over the public HTTPS URL; closest to you is fine).  •  Plan: **Free**.
   - Click **Create new project** and wait ~2 minutes for it to provision.

2. **Run the schema.** Left menu → **SQL Editor** → **New query**. Open
   `backend/sql/schema.sql` from the repo, paste its **entire** contents, and click
   **Run**. This single script:
   - runs `CREATE EXTENSION IF NOT EXISTS vector;` (enables **pgvector** — no separate
     toggle needed),
   - creates the `documents` table (`id, content, metadata jsonb, embedding vector(1024),
     fts tsvector, content_hash UNIQUE, source_kind, created_at`),
   - creates the **HNSW** index on `embedding` and the **GIN** index on `fts`,
   - creates the `match_documents(query_embedding, query_text, match_count, rrf_k)`
     RPC that does hybrid vector+full-text search fused with **Reciprocal Rank Fusion**.

   You should see "Success. No rows returned."

3. **Copy your two connection values.** Left menu → **Project Settings** → **API Keys**
   (and the **Data API**/**General** page for the URL).
   ([Supabase: understanding API keys](https://supabase.com/docs/guides/getting-started/api-keys))
   - **Project URL** — looks like `https://<project-ref>.supabase.co`. This becomes
     `SUPABASE_URL`.
   - **Service role key** — the backend needs a key that **bypasses Row Level Security**
     to write/read all rows. In the **API Keys** screen:
     - **Legacy path (works today):** open the **Legacy API Keys** tab and copy the
       **`service_role`** secret (a long JWT). This becomes `SUPABASE_SERVICE_KEY`.
     - **New path:** Supabase is migrating to **publishable/secret** keys. A new
       **secret key** (`sb_secret_...`) is the drop-in replacement for `service_role`
       and works the same way with the backend. Either is fine.
       (Legacy `service_role` keys are slated for deprecation by **end of 2026**, so a
       new secret key is the more future-proof choice.)
       ([Supabase: migrating to new API keys](https://supabase.com/docs/guides/getting-started/migrating-to-new-api-keys))

   > **Treat the service role / secret key like a root password.** It bypasses RLS.
   > Never put it in frontend code, `NEXT_PUBLIC_*`, or git. It lives only in AWS
   > Secrets Manager (prod) and your local `.env` (seeding/eval).

Keep the **Project URL** and **service/secret key** handy — you need them in Steps 6 and 7.

---

## 6. Deploy the AWS backend with SAM

### 6a. First decide your Vercel URL (to avoid a chicken-and-egg)

CORS and the Cognito sign-in callback need your eventual Vercel URL. You won't know it
for certain until Step 8, but Vercel URLs are predictable: **`https://<project-name>.vercel.app`**.
If you'll name the Vercel project `amanai`, your origin will be `https://amanai.vercel.app`
(the template's default). If it ends up different, you'll redeploy in Step 8c — that's
expected and cheap.

### 6b. Run the guided deploy

```bash
cd infra
bash deploy.sh
```

`deploy.sh` runs `sam build` then `sam deploy --guided`. On the **first** run SAM asks a
series of questions. Answer them as follows (press Enter to accept a shown default):

| Prompt | What to enter | Notes |
|---|---|---|
| `Stack Name` | `amanai` | The CloudFormation stack name. |
| `AWS Region` | `us-east-1` | Must be us-east-1. |
| `Parameter Environment` | `prod` | `dev` or `prod`; suffixes resource names. |
| `Parameter VercelOrigin` | `https://amanai.vercel.app` | Your Vercel URL from 6a (update in 8c if different). |
| `Parameter BudgetAlertEmail` | `you@example.com` | Gets the $20 budget alerts. |
| `Parameter CognitoHostedUiDomainPrefix` | `amanai-auth-<yourname>` | **Globally unique** across all AWS — add your name/initials. |
| `Parameter SupabaseUrl` | `https://<project-ref>.supabase.co` | From Step 5. **Set it here** (the clean way — see note below). |
| `Confirm changes before deploy` | `Y` | Lets you review the resource diff each deploy. |
| `Allow SAM CLI IAM role creation` | `Y` | The stack creates a least-privilege Lambda role; required. |
| `Disable rollback` | `N` | Keep rollback on so failed deploys clean up. |
| `<Function> may not have authorization defined, Is this okay?` | `Y` | This is the **public `/health`** route — intentional. |
| `Save arguments to configuration file` | `Y` | Writes your answers to `infra/samconfig.toml`. |
| `SAM configuration file` / `environment` | accept defaults | `samconfig.toml` / `default`. |

> **Why set `SupabaseUrl` as a SAM parameter (not via the Lambda CLI):** `SUPABASE_URL`
> is a CloudFormation **parameter** wired into the Lambda's environment by the template.
> Setting it here keeps the template the single source of truth. **Do NOT** use
> `aws lambda update-function-configuration --environment "Variables={SUPABASE_URL=...}"`
> to add it after the fact — that command **replaces the entire environment map** and
> would wipe the model IDs, Cognito IDs, and Guardrail ID the template set. If you ever
> must change the URL later, change the `SupabaseUrl` parameter and re-deploy.

The deploy takes ~3–6 minutes (Cognito, the Bedrock Guardrail, IAM, API Gateway, Lambda,
Secrets Manager, and the Budget all get created — 11 resources).

### 6c. Capture the stack Outputs

When the deploy finishes, CloudFormation prints an **Outputs** table. `deploy.sh` also
re-prints the important ones. Record these (exact Output keys from `infra/template.yaml`):

| Output key | Example value | You'll use it for |
|---|---|---|
| `ApiUrl` | `https://abc123.execute-api.us-east-1.amazonaws.com/prod` | Frontend `NEXT_PUBLIC_API_BASE_URL`; smoke tests |
| `UserPoolId` | `us-east-1_AbCdEf123` | Frontend; admin-user CLI; eval |
| `UserPoolClientId` | `1a2b3c4d5e6f7g8h9i0j` | Frontend `NEXT_PUBLIC_COGNITO_CLIENT_ID` |
| `HostedUiDomain` | `https://amanai-auth-yourname.auth.us-east-1.amazoncognito.com` | Optional hosted sign-in page |
| `GuardrailId` | `abcd1234efgh` | Reference; already wired into the Lambda |
| `GuardrailVersion` | `1` | Reference |
| `SupabaseSecretArn` | `arn:aws:secretsmanager:us-east-1:...:secret:amanai/prod/supabase-service-key-XXXX` | Setting the secret in Step 7 |

You can re-print them any time:

```bash
aws cloudformation describe-stacks --stack-name amanai --region us-east-1 \
  --query "Stacks[0].Outputs" --output table
```

---

## 7. Put the Supabase service key into Secrets Manager

The template created the secret with a placeholder. Set its real value now (the backend
reads `SUPABASE_SERVICE_KEY` from this secret at runtime). Use the exact command
`deploy.sh` printed, with the secret id from `SupabaseSecretArn` in Step 6c:

```bash
aws secretsmanager put-secret-value \
  --secret-id "amanai/prod/supabase-service-key" \
  --secret-string '{"SUPABASE_SERVICE_KEY":"<PASTE_YOUR_SUPABASE_SERVICE_OR_SECRET_KEY>"}' \
  --region us-east-1
```

> The JSON key **must** be exactly `SUPABASE_SERVICE_KEY` (that's what
> `backend/app/secrets.py` looks up). `SUPABASE_URL` is already set (Step 6b parameter),
> and the model IDs / Cognito IDs / Guardrail ID were set by the template — nothing else
> to configure on the Lambda.

Verify (prints metadata, not the secret value):

```bash
aws secretsmanager describe-secret --secret-id "amanai/prod/supabase-service-key" --region us-east-1
```

---

## 8. Deploy the frontend on Vercel

### 8a. Import the repo

1. [Vercel dashboard](https://vercel.com/dashboard) → **Add New…** → **Project**.
2. **Import** your GitHub `AmanAI` repository (authorize Vercel for the repo if asked).
3. **Project Name:** `amanai` (this is what makes the URL `https://amanai.vercel.app`;
   if you pick a different name, note the resulting URL for Step 8c).
4. **Root Directory:** click **Edit** and set it to **`frontend`**. This is essential —
   the Next.js app lives in `frontend/`, not the repo root.
   ([Vercel monorepos](https://vercel.com/docs/monorepos))
5. **Framework Preset:** Vercel auto-detects **Next.js** (leave as is). Build command,
   output, and install command can stay on defaults.

### 8b. Add the frontend environment variables

In the import screen (or later under **Project → Settings → Environment Variables**),
add these. Values come from the Step 6c stack Outputs. **Only these four are read by the
app** (`frontend/lib/api.ts` + `frontend/lib/amplify.ts`):
([Vercel environment variables](https://vercel.com/docs/environment-variables))

| Variable | Value (from stack Outputs) |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | the `ApiUrl` output, e.g. `https://abc123.execute-api.us-east-1.amazonaws.com/prod` |
| `NEXT_PUBLIC_COGNITO_REGION` | `us-east-1` |
| `NEXT_PUBLIC_COGNITO_USER_POOL_ID` | the `UserPoolId` output |
| `NEXT_PUBLIC_COGNITO_CLIENT_ID` | the `UserPoolClientId` output |

> **Note on the "domain"/"hosted UI" variable.** `infra/deploy.sh` and `frontend/.env.example`
> mention `NEXT_PUBLIC_COGNITO_HOSTED_UI` / `NEXT_PUBLIC_COGNITO_DOMAIN`. The current app
> code does **not** read it (sign-in uses Amplify's SRP flow directly against the user
> pool, not the hosted UI redirect). Setting it is harmless but **not required**. The
> four variables above are sufficient.

Set each variable's scope to **Production** (and Preview/Development if you want preview
deploys to work). Then click **Deploy**. The build runs `next build`; it produces the
app and finishes in ~1–2 minutes.

> The frontend build was verified locally: `tsc --noEmit` passes and `next build`
> generates all routes. The config is `frontend/next.config.mjs` (Next.js 14 cannot load
> a TypeScript `next.config.ts`).

### 8c. Point the backend at the real Vercel URL

After the Vercel deploy, note your actual production URL (e.g. `https://amanai.vercel.app`
or `https://amanai-<hash>.vercel.app`). If it **differs** from what you entered for
`VercelOrigin` in Step 6b, update the backend so CORS and the Cognito callback/logout URLs
include it:

```bash
cd infra
sam deploy --config-file samconfig.toml \
  --parameter-overrides \
    "Environment=prod" \
    "SupabaseUrl=https://<project-ref>.supabase.co" \
    "BudgetAlertEmail=you@example.com" \
    "CognitoHostedUiDomainPrefix=amanai-auth-yourname" \
    "VercelOrigin=https://<your-real-vercel-url>"
```

This updates the API Gateway `AllowOrigins` and the Cognito app client `CallbackURLs`/
`LogoutURLs` to your real origin. (Both `http://localhost:3000` and the Vercel origin are
always allowed, so local dev keeps working.)

> **Tip:** to keep one stable URL, assign a Vercel **production domain alias** (Project →
> Settings → Domains) like `amanai.vercel.app` and use that as `VercelOrigin` from the start.

---

## 9. Seed the knowledge corpus

This embeds the 358 NUST Bank documents and upserts them into Supabase. Run it **from the
repo root** with AWS credentials (for Bedrock Titan embeddings) and Supabase env vars set.

1. Create a local environment for the backend deps:

   ```bash
   # from repo root
   python3 -m venv backend/.venv          # if it doesn't already exist
   backend/.venv/bin/pip install -r backend/requirements.txt
   ```

2. Create a `.env` in the repo root (read by `backend/app/config.py` via pydantic-settings):

   ```bash
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=<your key>
   AWS_SECRET_ACCESS_KEY=<your secret>
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_SERVICE_KEY=<your supabase service/secret key>
   BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
   ```

   > `.env` is gitignored — never commit it.

3. Run the seeder (loads `data/processed/all_documents.json`, the 358-doc corpus that
   ships in the working tree):

   ```bash
   backend/.venv/bin/python -m ingestion.seed
   ```

   Expected log tail: `Seeding complete — added: 358, skipped (duplicates): 0`.
   It is **safe to re-run** — `content_hash` dedup means re-runs add 0 and skip 358.

   Custom corpus path (optional): `python -m ingestion.seed --path /path/to/docs.json`
   (the JSON must be a top-level array of `{ "content": "...", "metadata": {...} }`).

Confirm in Supabase: **Table Editor → documents** should show 358 rows with `source_kind = 'seed'`.

---

## 10. Create the first admin user

Any signed-in user can chat. Only members of the Cognito **`admin`** group can upload
documents (`POST /documents`). Create yourself as admin with the `UserPoolId` from Step 6c:

```bash
# 1. Create the user with a temporary password
aws cognito-idp admin-create-user \
  --user-pool-id <UserPoolId> \
  --username you@example.com \
  --temporary-password "TempPass1!" \
  --region us-east-1

# 2. Add them to the admin group (grants document upload)
aws cognito-idp admin-add-user-to-group \
  --user-pool-id <UserPoolId> \
  --username you@example.com \
  --group-name admin \
  --region us-east-1
```

> The pool password policy (from `infra/template.yaml`) requires **≥12 chars with
> upper, lower, number, and symbol** — your temporary and permanent passwords must satisfy
> it. On first sign-in via the frontend you'll be prompted to set a permanent password.

---

## 11. End-to-end smoke tests

Run in order once Steps 1–10 are done.

| # | Test | Expected |
|---|---|---|
| 1 | `curl https://<ApiUrl>/health` | `{"status":"ok","model":"us.meta.llama3-3-70b-instruct-v1:0","docs":358}` (public, no auth) |
| 2 | Open the Vercel URL → **Sign up** a new user | Cognito emails a verification code; entering it logs you in |
| 3 | Ask **"What is the profit rate on a 1-year NUST term deposit?"** | A grounded answer with **citations**; `refused: false` |
| 4 | Ask **"What is the weather in Islamabad?"** | Polite refusal / helpline redirect; `refused: true` (off-domain gate + guardrail) |
| 5 | Ask something with a fake CNIC like **"my CNIC is 35202-1234567-1, what's my balance?"** | PII is masked/blocked; no account data leaked |
| 6 | Sign in as the **admin** user → upload a small product `.json`/`.txt` via the admin panel | `{"added":N,"skipped":0}`; then ask a fact only in that file and confirm it's answered |

> A `401` on `/chat` means the JWT isn't being sent — sign out/in. A `403` on
> `/documents` means the user isn't in the `admin` group (re-run Step 10b). A CORS error
> in the browser console means the Vercel origin isn't in `VercelOrigin` (redo Step 8c).

---

## 12. (Optional) Run the RAGAS evaluation

Only works **after** deploy + seed (it drives the real pipeline). From the repo root:

```bash
backend/.venv/bin/pip install -r evaluation/requirements.txt

# Full RAGAS scoring of the 40-pair golden set → evaluation/evaluation_results.json
backend/.venv/bin/python -m evaluation.evaluate

# Calibrate the refusal threshold (no LLM calls) → evaluation/threshold_calibration.json
backend/.venv/bin/python -m evaluation.calibrate_threshold
```

`evaluate.py` loads `evaluation/golden_dataset.json` (40 Q&A pairs: 26 in-domain, 3
multi-turn, 6 out-of-domain, 5 PII), calls the chat pipeline in-process, and scores
non-refused answers on faithfulness / answer_relevancy / context_precision /
context_recall using a Bedrock judge. If the recommended threshold from
`calibrate_threshold` differs from `0.30`, set `MIN_RERANK_SCORE` accordingly (update the
SAM `MIN_RERANK_SCORE` env in `infra/template.yaml` and re-deploy). See
[`evaluation/README.md`](../evaluation/README.md) for the result-file schemas.

---

## 13. Re-deploying after code changes

- **Backend / infra change:** `cd infra && sam build && sam deploy --config-file samconfig.toml`
  (it reuses your saved parameters; add `--parameter-overrides ...` to change any).
- **Frontend change:** just `git push` — Vercel auto-redeploys the connected branch.
- **Backend tests before deploying** (from repo root): `backend/.venv/bin/python -m pytest backend/tests -q` → expect **56 passed**.

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `sam deploy` fails: `AccessDeniedException` naming a model | Bedrock model not yet approved | Step 4 — approve all four models in **Bedrock → Model access** (us-east-1); wait for "Access granted". |
| `/health` shows `"docs": 0` or `null` | Corpus not seeded, or wrong Supabase URL/key | Re-run Step 9; confirm `documents` has 358 rows; check the secret in Step 7. |
| `/chat` returns `500` mentioning Supabase | `SUPABASE_SERVICE_KEY` wrong/placeholder, or `match_documents` RPC missing | Re-run Step 7 with the real key; re-run `schema.sql` (Step 5.2). |
| `/chat` returns `500` mentioning a model / inference profile | Llama profile not accessible | Confirm Llama 3.3 70B **and** 3.1 8B approved; IDs use the `us.` prefix (Step 4 note). |
| Every answer is a refusal (`refused: true`) | `MIN_RERANK_SCORE` too high, or Cohere Rerank not approved | Approve `cohere.rerank-v3-5:0`; run `calibrate_threshold` and lower `MIN_RERANK_SCORE` if needed. |
| Browser console: CORS / blocked by policy | Vercel origin not allowed | Step 8c — redeploy backend with the exact `VercelOrigin`. |
| Sign-in fails / "User pool client does not exist" | Wrong `NEXT_PUBLIC_COGNITO_*` in Vercel | Re-check against `UserPoolId` / `UserPoolClientId` outputs; redeploy Vercel after editing env vars. |
| `403` uploading a document | User not in `admin` group | Step 10b `admin-add-user-to-group`. |
| Lambda lost its env vars (model IDs, etc.) | Someone ran `lambda update-function-configuration --environment` | Re-deploy with SAM (Step 6b) to restore the full env from the template. Never set single vars via that CLI. |
| `CognitoHostedUiDomainPrefix` deploy error: domain taken | Prefix not globally unique | Re-deploy with a more unique prefix (add your name). |
| Budget alert email never arrives | SNS/email not confirmed | Check spam; confirm the address in **Billing → Budgets**. |

---

## 15. Teardown (delete everything, stop all charges)

```bash
# 1. Delete the whole AWS stack (Lambda, API GW, Cognito, Guardrail, IAM, Budget, secret)
aws cloudformation delete-stack --stack-name amanai --region us-east-1
aws cloudformation wait stack-delete-complete --stack-name amanai --region us-east-1

# 2. The Secrets Manager secret has a 7-day recovery window by default. To purge now:
aws secretsmanager delete-secret \
  --secret-id "amanai/prod/supabase-service-key" \
  --force-delete-without-recovery \
  --region us-east-1
```

- **Supabase:** Dashboard → your project → **Project Settings → General → Delete project**.
- **Vercel:** Project → **Settings → (bottom) Delete Project**.
- **AWS manual budget** from Step 2b: **Billing → Budgets →** delete `amanai-manual-guard`.

After teardown, idle cost returns to **$0**.

---

## Appendix A — Resource & name reference (from the IaC)

| Thing | Value | Defined in |
|---|---|---|
| CloudFormation stack | `amanai` | `infra/samconfig.toml` |
| Region | `us-east-1` | everywhere |
| Lambda function | `amanai-backend-${Environment}` (e.g. `amanai-backend-prod`) | `template.yaml` |
| Lambda handler / runtime | `app.main.handler` / Python 3.12, arm64, 1024 MB, 30 s | `template.yaml` |
| Cognito user pool | `amanai-userpool-${Environment}` | `template.yaml` |
| Cognito app client | `amanai-spa-client-${Environment}` (no secret, SRP) | `template.yaml` |
| Admin group | `admin` | `template.yaml` |
| Secrets Manager secret | `amanai/${Environment}/supabase-service-key` | `template.yaml` |
| Bedrock Guardrail | `amanai-guardrail-${Environment}` | `template.yaml` |
| Budget | `amanai-monthly-budget-${Environment}`, $20/mo, alerts 80%/100% actual + 100% forecast | `template.yaml` |
| API throttling | burst 20, rate 10 req/s | `template.yaml` |
| Routes | `GET /health` (public), `POST /chat` (JWT), `POST /documents` (JWT + admin) | `template.yaml`, `CONTRACTS.md` |

## Appendix B — Backend environment variables (the contract)

Set automatically by `template.yaml` on the Lambda; defaults live in `backend/app/config.py`
and the contract in `docs/CONTRACTS.md`.

```
AWS_REGION=us-east-1
BEDROCK_GEN_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
BEDROCK_GEN_MODEL_ID_FALLBACK=us.meta.llama3-1-8b-instruct-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_RERANK_MODEL_ID=cohere.rerank-v3-5:0
BEDROCK_GUARDRAIL_ID=<from stack>        BEDROCK_GUARDRAIL_VERSION=<from stack>
EMBED_DIM=1024  RETRIEVE_K=8  RERANK_TOP_N=4  MIN_RERANK_SCORE=0.30
MAX_INPUT_CHARS=2000  MAX_UPLOAD_MB=5
SUPABASE_URL=<your project URL>          SUPABASE_SECRET_ARN=<from stack>   # key pulled from Secrets Manager at runtime
COGNITO_USER_POOL_ID=<from stack>  COGNITO_CLIENT_ID=<from stack>  COGNITO_REGION=us-east-1
ALLOWED_ORIGINS=http://localhost:3000,<your Vercel origin>
LOG_LEVEL=INFO
```

---

## Sources (verified 2026-06-19)

- AWS Bedrock — [Model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) · [Cross-region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) · [Llama 3.3 70B on Bedrock](https://aws.amazon.com/about-aws/whats-new/2024/12/metas-llama-3-3-70b-model-amazon-bedrock/)
- AWS CLI v2 — [Install/update guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- AWS SAM CLI — [Install guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- AWS Budgets — [Managing costs with Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html)
- Supabase — [Understanding API keys](https://supabase.com/docs/guides/getting-started/api-keys) · [Migrating to new API keys](https://supabase.com/docs/guides/getting-started/migrating-to-new-api-keys) · [pgvector / AI & vectors](https://supabase.com/docs/guides/database/extensions/pgvector)
- Vercel — [Monorepos / Root Directory](https://vercel.com/docs/monorepos) · [Environment variables](https://vercel.com/docs/environment-variables) · [Next.js on Vercel](https://vercel.com/docs/frameworks/full-stack/nextjs)
