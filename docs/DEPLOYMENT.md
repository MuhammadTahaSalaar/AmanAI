# AmanAI — Deployment Runbook

Ordered steps to go from a fresh clone to a live production system. Follow them top-to-bottom; each step depends on the previous one completing successfully.

---

## Prerequisites

| Tool | Notes |
|---|---|
| AWS account | Apply the $120 promotional credits before spending begins |
| AWS CLI v2 | `aws configure` with Access Key ID, Secret Access Key, region `us-east-1`, output `json` |
| SAM CLI | `pip install aws-sam-cli` or `brew install aws-sam-cli` |
| Python 3.12 | Required for SAM build and local backend dev |
| Node.js 18+ | Required for the frontend |
| Supabase account | Free tier at [supabase.com](https://supabase.com) |
| Vercel account | Free tier at [vercel.com](https://vercel.com) |

---

## Step 1 — Enable Bedrock model access (us-east-1)

In the AWS Console: **Bedrock → Model access → Manage model access** → enable:

| Model ID | Purpose |
|---|---|
| `meta.llama3-1-8b-instruct-v1:0` | Generation fallback / cheap |
| `meta.llama3-3-70b-instruct-v1:0` | Generation primary |
| `amazon.titan-embed-text-v2:0` | Embeddings (1024-dim) |
| `cohere.rerank-v3-5:0` | Reranking |

> Model access approval usually takes a few minutes but can take up to an hour. The SAM deploy will fail with `AccessDeniedException` until all four are approved.

---

## Step 2 — Set up Supabase

1. Create a new project in the Supabase dashboard (choose any region; the Lambda connects over the public URL).
2. In **Database → Extensions**, enable the `vector` extension.
3. Open **SQL Editor** and run the full contents of `backend/sql/schema.sql`. This creates the `documents` table (with HNSW index and GIN FTS index) and the `match_documents` hybrid-search RPC.
4. Copy two values from **Project Settings → API**:
   - **Project URL** — looks like `https://<ref>.supabase.co`
   - **Service role key** (the `service_role` JWT, not the `anon` key)

Keep these handy; you will need them in Steps 4 and 5.

---

## Step 3 — Build and deploy the AWS stack

```bash
cd infra
bash deploy.sh
```

On the first run `deploy.sh` calls `sam build && sam deploy --guided`, which prompts for four parameters:

| Parameter | Example value |
|---|---|
| `VercelOrigin` | `https://your-project.vercel.app` |
| `BudgetAlertEmail` | `you@example.com` |
| `CognitoHostedUiDomainPrefix` | `amanai-auth-yourname` (must be globally unique) |
| `Environment` | `prod` |

SAM saves your answers to `infra/samconfig.toml`. Future re-deploys need only:

```bash
sam build && sam deploy --config-file infra/samconfig.toml
```

At the end of a successful deploy, CloudFormation prints **Outputs** — copy them. You need `ApiEndpoint`, `UserPoolId`, `UserPoolClientId`, and `HostedUiUrl`.

---

## Step 4 — Wire Supabase credentials into AWS

### A. Store the service key in Secrets Manager

```bash
aws secretsmanager put-secret-value \
  --secret-id "amanai/prod/supabase-service-key" \
  --secret-string '{"SUPABASE_SERVICE_KEY":"eyJ...your_service_role_key..."}' \
  --region us-east-1
```

The SAM template creates the secret; this command sets its value.

### B. Set the Supabase project URL on the Lambda

```bash
aws lambda update-function-configuration \
  --function-name amanai-backend-prod \
  --environment "Variables={SUPABASE_URL=https://<your-project-ref>.supabase.co}" \
  --region us-east-1
```

The remaining environment variables (model IDs, Cognito IDs, Guardrail ID) are set automatically by the SAM template.

---

## Step 5 — Seed the knowledge corpus

With AWS credentials and Supabase env vars set locally (or in `.env`):

```bash
python -m ingestion.seed
```

This embeds and upserts the 358 NUST Bank product documents from `data/processed/all_documents.json` into Supabase. Content-hash deduplication means it is safe to re-run.

---

## Step 6 — Create the first admin user

```bash
# 1. Create the user (they will receive a temporary password)
aws cognito-idp admin-create-user \
  --user-pool-id <UserPoolId from stack output> \
  --username admin@example.com \
  --temporary-password "TempPass1!" \
  --region us-east-1

# 2. Grant admin privileges (required for POST /documents)
aws cognito-idp admin-add-user-to-group \
  --user-pool-id <UserPoolId from stack output> \
  --username admin@example.com \
  --group-name admin \
  --region us-east-1
```

The user must sign in once through the frontend or Cognito Hosted UI to set a permanent password.

---

## Step 7 — Deploy the frontend on Vercel

1. In the Vercel dashboard, click **Add New Project → Import Git Repository** and select this repo.
2. Set **Root Directory** to `frontend`.
3. Add the following **Environment Variables** (values come from the stack Outputs printed in Step 3):

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://<api-id>.execute-api.us-east-1.amazonaws.com/prod` |
| `NEXT_PUBLIC_COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `NEXT_PUBLIC_COGNITO_CLIENT_ID` | Cognito User Pool Client ID |
| `NEXT_PUBLIC_COGNITO_REGION` | `us-east-1` |
| `NEXT_PUBLIC_COGNITO_DOMAIN` | Hosted UI base URL (e.g. `amanai-auth-yourname.auth.us-east-1.amazoncognito.com`) |

4. Click **Deploy**. Vercel auto-detects Next.js.

Subsequent pushes to `main` trigger automatic redeployment.

---

## Step 8 — Smoke tests

Run these in order after deployment completes.

| Test | Expected result |
|---|---|
| `curl https://<api>/health` | `{"status":"ok","model":"us.meta.llama3-3-70b-instruct-v1:0","docs":358}` |
| Sign up a new user in the frontend | Cognito confirmation email arrives; user can sign in |
| Sign in and send "What is the profit rate on a 1-year term deposit?" | Answer with citations; `refused: false` |
| Send "What is the weather in Islamabad?" | Answer redirects to helpline or is a polite refusal; `refused: true` |
| Sign in as admin, upload a new product JSON file via the admin panel | `{"added":N,"skipped":0}` response; immediately query a fact only in that file and confirm the answer reflects it |

---

## Step 9 — Budget alarm

The SAM template creates an AWS Budgets alert that emails `BudgetAlertEmail` at 80% and 100% of a $20 monthly threshold, and at 100% of a $20 forecasted spend. Confirm in **AWS Billing → Budgets** that the `amanai-budget` entry is active.

---

## Teardown

```bash
# Delete all AWS resources created by SAM
aws cloudformation delete-stack --stack-name amanai --region us-east-1

# Force-delete the Secrets Manager secret (otherwise it lingers 7 days)
aws secretsmanager delete-secret \
  --secret-id amanai/prod/supabase-service-key \
  --force-delete-without-recovery \
  --region us-east-1
```

For Supabase: go to **Project Settings → General → Delete project**.

For Vercel: go to **Project Settings → Advanced → Delete project**.
