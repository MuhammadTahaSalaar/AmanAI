# AmanAI – Infrastructure as Code (AWS SAM)

Scale-to-zero serverless RAG on AWS. Estimated cost: **< $5/month** at student load; hard budget alert at $20.

---

## Prerequisites

### 1. Enable Bedrock Model Access (us-east-1)

In the AWS Console → **Bedrock → Model access** → request access for:

| Model | Purpose |
|---|---|
| `meta.llama3-3-70b-instruct-v1:0` | Generation (primary) |
| `meta.llama3-1-8b-instruct-v1:0` | Generation (fallback) |
| `amazon.titan-embed-text-v2:0` | Embeddings (1024-dim) |
| `cohere.rerank-v3-5:0` | Reranking |

> Model access may take a few minutes. Deploy will fail with `AccessDeniedException` until access is approved.

### 2. Install CLI Tools

```bash
# AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# SAM CLI
pip install aws-sam-cli
# or: brew install aws-sam-cli

# Configure credentials
aws configure
# Enter: Access Key ID, Secret Access Key, us-east-1, json
```

### 3. Python 3.12 (for local dev / SAM build)

```bash
python3 --version   # should be 3.12.x
```

---

## Deploy

```bash
# From repo root:
bash infra/deploy.sh
```

On **first run** you will be prompted (`--guided`) for:

| Parameter | Example value |
|---|---|
| `VercelOrigin` | `https://your-project.vercel.app` |
| `BudgetAlertEmail` | `you@example.com` |
| `CognitoHostedUiDomainPrefix` | `amanai-auth-yourname` (globally unique) |
| `Environment` | `prod` |

SAM saves answers to `infra/samconfig.toml` for future deploys.

Subsequent deploys (no prompt):
```bash
sam build && sam deploy --config-file infra/samconfig.toml
```

---

## Post-Deploy Steps

The deploy script prints these values — follow the on-screen instructions, or do them manually:

### A. Store the Supabase Service Key

```bash
aws secretsmanager put-secret-value \
  --secret-id "amanai/prod/supabase-service-key" \
  --secret-string '{"SUPABASE_SERVICE_KEY":"eyJ...your_key..."}' \
  --region us-east-1
```

### B. Set the Supabase URL

Update `SUPABASE_URL` in the Lambda environment (or re-deploy with it set in template.yaml):

```bash
aws lambda update-function-configuration \
  --function-name amanai-backend-prod \
  --environment "Variables={SUPABASE_URL=https://<your-project-ref>.supabase.co}" \
  --region us-east-1
```

> The remaining env vars (model IDs, guardrail ID, Cognito IDs) are set automatically by the template.

### C. Create the First Admin User

```bash
# 1. Create the user
aws cognito-idp admin-create-user \
  --user-pool-id <UserPoolId from stack output> \
  --username admin@example.com \
  --temporary-password "TempPass1!" \
  --region us-east-1

# 2. Add to admin group (grants /documents access)
aws cognito-idp admin-add-user-to-group \
  --user-pool-id <UserPoolId from stack output> \
  --username admin@example.com \
  --group-name admin \
  --region us-east-1
```

The user must then sign in (via frontend or Hosted UI) to set a permanent password.

### D. Configure Vercel

In your Vercel project → Settings → Environment Variables, add:

| Variable | Value (from stack Outputs) |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | `https://<api-id>.execute-api.us-east-1.amazonaws.com/prod` |
| `NEXT_PUBLIC_COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `NEXT_PUBLIC_COGNITO_CLIENT_ID` | Cognito User Pool Client ID |
| `NEXT_PUBLIC_COGNITO_REGION` | `us-east-1` |
| `NEXT_PUBLIC_COGNITO_HOSTED_UI` | Hosted UI base URL |

---

## Architecture

```
Browser (Vercel Next.js)
  │  Cognito JWT
  ▼
API Gateway HTTP API (v2)
  │  /health → no auth
  │  /chat   → Cognito JWT authorizer
  │  /documents → Cognito JWT authorizer (admin enforced in-app)
  ▼
Lambda (Python 3.12, arm64, 1024 MB, 30s)
  ├── Bedrock Titan Embed v2   (embeddings)
  ├── Bedrock Cohere Rerank 3.5 (reranking)
  ├── Bedrock Llama 3.3 70B    (generation, cross-region profile)
  ├── Bedrock Guardrail        (PII mask + topic deny + content filter)
  └── Supabase (pgvector RPC)  (hybrid search + storage)

Secrets Manager → SUPABASE_SERVICE_KEY
Cognito User Pool → auth + admin group
Budgets → $20/month alert
```

---

## Cost Estimate (Student Load)

| Service | Estimated Monthly Cost |
|---|---|
| Lambda (arm64, ~1000 req/day × 5s avg) | ~$0.50 |
| API Gateway HTTP API | ~$0.10 |
| Bedrock Titan Embed (1000 req) | ~$0.08 |
| Bedrock Llama 3.3 70B (1000 req, ~500 tok avg) | ~$1.50 |
| Bedrock Cohere Rerank | ~$0.10 |
| Secrets Manager | ~$0.40 |
| Cognito (< 50k MAU free tier) | $0.00 |
| **Total** | **~$3–5/month** |

Budget alert fires at $20 (80% and 100% actual, 100% forecasted).

---

## Teardown

```bash
aws cloudformation delete-stack --stack-name amanai --region us-east-1
# Note: Secrets Manager secret has a 7-day recovery window by default.
# To force-delete: aws secretsmanager delete-secret --secret-id amanai/prod/supabase-service-key --force-delete-without-recovery
```
