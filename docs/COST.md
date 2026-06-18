# AmanAI — Cost Model

All prices are approximate. Confirm current rates at the [AWS Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) before making financial decisions.

---

## Per-query breakdown

One `/chat` call runs: Guardrail input check → Titan embed (query) → Cohere Rerank (8 candidates → 4) → Llama 3.3 70B generate → Guardrail output check.

| Component | Approximate cost | Notes |
|---|---|---|
| Titan Embeddings v2 (query) | ~$0.00002 | 1 call, ~50 tokens |
| Cohere Rerank 3.5 | ~$0.002 | ~$2 per 1 000 queries |
| Llama 3.3 70B generation | ~$0.002 | ~500 input + ~300 output tokens; cross-region profile pricing |
| Bedrock Guardrails | ~$0.0001–0.00015 | ~$0.10–0.15 per 1 000 text units |
| **Total per query** | **~$0.004–0.005** | All-in estimate |

At low traffic (student / demo use) the dominant cost is Llama generation. Cohere Rerank is the second-largest item.

---

## Monthly estimates

| Traffic level | Queries / month | Estimated cost |
|---|---|---|
| Idle (no queries) | 0 | ~$0 |
| Light (demo / student) | ~1 000 | ~$4–5 |
| Moderate | ~5 000 | ~$20–25 |
| Heavy | ~20 000 | ~$80–100 |

A budget alarm fires at $20/month (see [DEPLOYMENT.md](DEPLOYMENT.md) Step 9).

---

## Why idle cost is essentially $0

| Service | Idle cost |
|---|---|
| AWS Lambda | $0 — billed per invocation only (scale-to-zero) |
| API Gateway HTTP API | $0 — billed per request |
| Amazon Bedrock | $0 — billed per token/request, no reservation |
| Supabase | $0 — free tier (500 MB storage, no compute charge at rest) |
| Vercel | $0 — free tier (static + serverless functions) |
| AWS Cognito | $0 — free up to 50 000 MAU |
| Secrets Manager | ~$0.40/month — the one fixed cost (one secret, $0.40/secret/month) |

The only unavoidable fixed charge is the Secrets Manager secret storing the Supabase service key (~$0.40/month). Everything else is strictly pay-per-use.

---

## How $120 lasts 6–12+ months

At the light-traffic estimate of ~$5/month (including the fixed Secrets Manager charge), $120 lasts approximately **24 months**. Even at moderate usage (~$20/month) it lasts **6 months**. The $20 budget alarm provides an early warning well before credits are exhausted.

To extend runway further:
- Switch to the fallback model (`us.meta.llama3-1-8b-instruct-v1:0`) via the `BEDROCK_GEN_MODEL_ID_FALLBACK` env var for low-stakes queries — cost drops by roughly 8–10x on generation.
- Reduce `RETRIEVE_K` (default 8) to lower the Cohere Rerank call size.

---

## Pricing references

- Bedrock on-demand pricing (generation, embeddings, rerank): https://aws.amazon.com/bedrock/pricing/
- Bedrock Guardrails pricing: included on the same page under "Guardrails"
- Supabase free tier: https://supabase.com/pricing
- Vercel free tier: https://vercel.com/pricing
- Cognito free tier: https://aws.amazon.com/cognito/pricing/
