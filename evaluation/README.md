# AmanAI Evaluation Harness v2

Evaluation tooling for the rebuilt RAG backend (`backend/app/`).  
Drives questions through the live pipeline in-process and scores them with RAGAS using an AWS Bedrock judge model.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Same environment as the backend |
| AWS credentials | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN` — must have Bedrock `InvokeModel` + `Rerank` permissions |
| Supabase | `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` pointing at a populated instance |
| Backend deps | `pip install -r backend/requirements.txt` |
| Eval deps | `pip install -r evaluation/requirements.txt` |

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

BEDROCK_GEN_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
BEDROCK_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_RERANK_MODEL_ID=cohere.rerank-v3-5:0
BEDROCK_GUARDRAIL_ID=          # leave blank to skip guardrails
MIN_RERANK_SCORE=0.30          # refusal floor — calibrate with calibrate_threshold.py

SUPABASE_URL=https://<project>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>

# Optional: override the judge model used by RAGAS
RAGAS_JUDGE_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
RAGAS_EMBED_MODEL_ID=amazon.titan-embed-text-v2:0
```

---

## Running the evaluation

All commands are run from the **repo root** (`/data/Nust/.../AmanAI`).

### Full RAGAS evaluation

```bash
python -m evaluation.evaluate_v2
```

What it does:

1. Loads `evaluation/golden_dataset_v2.json` (40 Q&A pairs).
2. Calls `backend.app.chat.handle_chat` in-process for every question.
3. Scores non-refused answers with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) using `ChatBedrockConverse` + `BedrockEmbeddings` as the judge.
4. Saves results to `evaluation/evaluation_results_v2.json`.

If AWS credentials or RAGAS are unavailable the script degrades gracefully — it still saves raw answers and contexts and logs a clear warning.

### Threshold calibration

```bash
python -m evaluation.calibrate_threshold
```

What it does:

1. Runs the retrieval pipeline (embed → match_documents → Cohere rerank) for every question in the golden set **without** calling the LLM.
2. Sweeps `MIN_RERANK_SCORE` over `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`.
3. For each threshold, reports the pass-rate for in-domain questions vs out-of-domain questions, and the gap between them.
4. Recommends the threshold that **maximises the gap** (best separation between in-domain and out-of-domain).
5. Saves full data to `evaluation/threshold_calibration.json`.

After running, update `MIN_RERANK_SCORE` in your `.env` to the recommended value.

---

## Reading the results

### `evaluation_results_v2.json` structure

```json
{
  "meta": {
    "total_samples": 40,
    "ragas_scored": true,
    "judge_model": "us.meta.llama3-3-70b-instruct-v1:0"
  },
  "aggregate": {
    "faithfulness": 0.87,
    "answer_relevancy": 0.82,
    "context_precision": 0.79,
    "context_recall": 0.75
  },
  "per_question": [
    {
      "index": 0,
      "question": "...",
      "ground_truth": "...",
      "answer": "...",
      "contexts": ["..."],
      "refused": false,
      "domain": "in_domain",
      "ragas_scores": {
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
        "context_precision": 0.80,
        "context_recall": 0.72
      }
    }
  ]
}
```

- **refused** `true` means the refusal gate or guardrail fired; RAGAS scores are omitted for these.
- **domain** is `in_domain`, `out_of_domain`, or `pii`; out-of-domain and pii samples are expected to have `refused: true`.

### `threshold_calibration.json` structure

```json
{
  "sweep": [
    { "threshold": 0.1, "in_domain_pass_rate": 0.97, "out_domain_pass_rate": 0.60, "gap": 0.37 },
    ...
  ],
  "recommended": 0.3,
  "gap": 0.85,
  "in_domain_scores": [...],
  "out_domain_scores": [...]
}
```

---

## Golden dataset

`golden_dataset_v2.json` contains **40 pairs** across four categories:

| Category | Count | Notes |
|---|---|---|
| In-domain (single-turn) | 26 | Savings, term deposits, IBFT, NUST4Car, Imarat, Sahar, Mortgage, Mastercard, Home Remittances |
| Multi-turn follow-ups | 3 | Carry a `history` array with prior turns |
| Out-of-domain | 6 | Weather, coding, politics, sports, hacking — expected to be refused |
| PII-bearing | 5 | CNIC, account numbers, PINs, passwords — expected to be refused/redacted |

The schema matches `golden_dataset.json` (the original 20 pairs) with two additional fields:

- `"domain"`: `"in_domain"` | `"out_of_domain"` | `"pii"`
- `"history"`: list of `{"role": "user"|"assistant", "content": "..."}` (empty list for single-turn)
