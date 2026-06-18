"""AmanAI Evaluation Harness v2.

Loads the golden dataset, drives each question through the NEW backend
pipeline (backend.app.chat.handle_chat), and computes RAGAS metrics using
a Bedrock judge model.  Saves per-question scores + aggregate means to
evaluation/evaluation_results.json.

Usage (from repo root):
    python -m evaluation.evaluate

Requirements: see evaluation/requirements.txt
AWS credentials + SUPABASE_* env vars must be set before running.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so that `backend.app.*` resolves.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAGAS_JUDGE_MODEL_ID = os.environ.get(
    "RAGAS_JUDGE_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0"
)
RAGAS_EMBED_MODEL_ID = os.environ.get(
    "RAGAS_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Golden dataset loader
# ---------------------------------------------------------------------------

def load_golden_dataset() -> list[dict[str, Any]]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pipeline runner — imports backend in-process
# ---------------------------------------------------------------------------

def run_pipeline(sample: dict[str, Any]) -> tuple[str, list[str], bool]:
    """Call handle_chat and return (answer, context_texts, refused)."""
    from backend.app.chat import handle_chat
    from backend.app.schemas import ChatRequest, HistoryTurn

    history_raw: list[dict[str, str]] = sample.get("history", [])
    history = [HistoryTurn(role=t["role"], content=t["content"]) for t in history_raw]

    request = ChatRequest(message=sample["question"], history=history)
    response = handle_chat(request)

    context_texts: list[str] = [c.content for c in response.citations]
    return response.answer, context_texts, response.refused


# ---------------------------------------------------------------------------
# RAGAS judge setup
# ---------------------------------------------------------------------------

def _build_ragas_llm() -> Any:
    """Return a LangChain-compatible LLM backed by AWS Bedrock."""
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=RAGAS_JUDGE_MODEL_ID,
        region_name=AWS_REGION,
    )


def _build_ragas_embeddings() -> Any:
    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        model_id=RAGAS_EMBED_MODEL_ID,
        region_name=AWS_REGION,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    golden = load_golden_dataset()
    logger.info("Loaded %d evaluation samples from %s", len(golden), GOLDEN_PATH)

    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[str] = []
    contexts_list: list[list[str]] = []
    refused_flags: list[bool] = []
    per_question: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1: run each question through the pipeline
    # ------------------------------------------------------------------
    for i, sample in enumerate(golden):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")
        logger.info(
            "Running [%d/%d]: %s", i + 1, len(golden), question[:80]
        )

        try:
            answer, contexts, refused = run_pipeline(sample)
        except Exception as exc:
            logger.error("Pipeline error on sample %d: %s", i + 1, exc)
            answer = ""
            contexts = []
            refused = False

        questions.append(question)
        answers.append(answer)
        ground_truths.append(ground_truth)
        contexts_list.append(contexts)
        refused_flags.append(refused)

        per_question.append(
            {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
                "refused": refused,
                "domain": sample.get("domain", "in_domain"),
            }
        )

    # ------------------------------------------------------------------
    # Step 2: RAGAS evaluation
    # ------------------------------------------------------------------
    ragas_available = False
    aggregate: dict[str, float] = {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ragas_available = True
    except ImportError as exc:
        logger.warning(
            "RAGAS / datasets not installed (%s). "
            "Install with: pip install -r evaluation/requirements.txt. "
            "Saving raw predictions only.",
            exc,
        )

    if ragas_available:
        try:
            judge_llm = _build_ragas_llm()
            judge_embeddings = _build_ragas_embeddings()
        except Exception as exc:
            logger.warning(
                "Could not initialise Bedrock judge (%s). "
                "Check AWS credentials and region. "
                "Saving raw predictions only.",
                exc,
            )
            ragas_available = False

    if ragas_available:
        # Filter out refused answers — RAGAS cannot score refusals meaningfully
        non_refused_indices = [
            idx for idx, r in enumerate(refused_flags) if not r
        ]
        if not non_refused_indices:
            logger.warning("All samples were refused; RAGAS scoring skipped.")
            ragas_available = False
        else:
            eval_data = {
                "question": [questions[i] for i in non_refused_indices],
                "answer": [answers[i] for i in non_refused_indices],
                "ground_truth": [ground_truths[i] for i in non_refused_indices],
                "contexts": [contexts_list[i] for i in non_refused_indices],
            }
            eval_dataset = Dataset.from_dict(eval_data)

            try:
                # Configure RAGAS to use Bedrock models
                for metric in [
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]:
                    metric.llm = judge_llm
                    if hasattr(metric, "embeddings"):
                        metric.embeddings = judge_embeddings

                results = evaluate(
                    eval_dataset,
                    metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_precision,
                        context_recall,
                    ],
                )

                aggregate = {k: float(v) for k, v in results.items()}
                logger.info("=== RAGAS Aggregate Results ===")
                for metric_name, score in aggregate.items():
                    logger.info("  %-25s %.4f", metric_name, score)

                # Attach per-question RAGAS scores back to per_question list
                results_df = results.to_pandas()
                for row_pos, orig_idx in enumerate(non_refused_indices):
                    row = results_df.iloc[row_pos]
                    per_question[orig_idx]["ragas_scores"] = {
                        col: float(row[col])
                        for col in [
                            "faithfulness",
                            "answer_relevancy",
                            "context_precision",
                            "context_recall",
                        ]
                        if col in row.index
                    }

            except Exception as exc:
                logger.error(
                    "RAGAS evaluate() failed (%s). "
                    "Check AWS credentials / model access. "
                    "Saving raw predictions only.",
                    exc,
                )
                ragas_available = False

    # ------------------------------------------------------------------
    # Step 3: Persist results
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "meta": {
            "golden_dataset": str(GOLDEN_PATH),
            "total_samples": len(golden),
            "ragas_scored": ragas_available,
            "judge_model": RAGAS_JUDGE_MODEL_ID if ragas_available else None,
            "embed_model": RAGAS_EMBED_MODEL_ID if ragas_available else None,
        },
        "aggregate": aggregate,
        "per_question": per_question,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Results saved to %s", RESULTS_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
