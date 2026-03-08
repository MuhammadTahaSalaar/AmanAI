"""Prepare fine-tuning data from the processed NUST Bank documents.

Converts ETL output into instruction-following pairs in Alpaca format:
  {"instruction": ..., "input": ..., "output": ...}

These are then formatted into the Llama-3.2 chat template for QLoRA training.
Run this BEFORE finetune.py.

Usage:
    python -m src.llm.prepare_finetune_data
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import config
from src.data_processing.etl_pipeline import ETLPipeline
from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FINETUNE_DATA_PATH = config.PROCESSED_DATA_DIR / "finetune_dataset.json"
SYSTEM_INSTRUCTION = (
    "You are AmanAI, NUST Bank's helpful customer service assistant. "
    "Answer questions accurately based on NUST Bank's products and policies. "
    "If you don't know the answer, say so politely and direct the user to contact the bank."
)


def _qa_pairs_from_documents(documents: list[Document]) -> list[dict]:
    """Convert documents into instruction-following pairs."""
    pairs: list[dict] = []

    for doc in documents:
        content = doc.content.strip()
        if not content:
            continue

        doc_type = doc.metadata.get("type", "")
        product = doc.metadata.get("product", doc.metadata.get("category", "NUST Bank"))

        if doc_type == "qa_pair":
            # Already a Q&A pair
            lines = content.split("\n", 1)
            if len(lines) == 2:
                question_line = lines[0].replace("Q:", "").strip()
                answer_line = lines[1].replace("A:", "").strip()
                pairs.append(
                    {
                        "instruction": question_line,
                        "input": "",
                        "output": answer_line,
                        "source": product,
                    }
                )

        elif doc_type == "rate_info" or "rate" in content.lower():
            # Rate info → generate a factual Q&A
            pairs.append(
                {
                    "instruction": f"Tell me about {product} rates or charges.",
                    "input": "",
                    "output": content,
                    "source": product,
                }
            )

        elif doc_type in ("description", "descriptive_block"):
            # Product description → ask about the product
            pairs.append(
                {
                    "instruction": f"What can you tell me about {product}?",
                    "input": "",
                    "output": content,
                    "source": product,
                }
            )

        else:
            # Generic: use content as answer to open-ended query
            if len(content) > 40:
                pairs.append(
                    {
                        "instruction": f"What does NUST Bank offer regarding {product}?",
                        "input": "",
                        "output": content,
                        "source": product,
                    }
                )

    return pairs


def _format_as_chat(pair: dict) -> dict:
    """Format an Alpaca pair into Llama 3.2 chat format for Unsloth."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": pair["instruction"]},
        {"role": "assistant", "content": pair["output"]},
    ]
    return {"messages": messages, "source": pair.get("source", "")}


def prepare() -> Path:
    """Run the full data preparation pipeline.

    Returns:
        Path to the saved JSONL fine-tuning dataset.
    """
    logger.info("Running ETL pipeline to get all documents...")
    pipeline = ETLPipeline()
    documents = pipeline.run()
    logger.info("ETL produced %d documents", len(documents))

    pairs = _qa_pairs_from_documents(documents)
    logger.info("Generated %d instruction pairs", len(pairs))

    # Shuffle for training stability
    random.seed(42)
    random.shuffle(pairs)

    # Format for Unsloth / ChatML
    formatted = [_format_as_chat(p) for p in pairs]

    # Split into train / validation (90/10)
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save as JSONL (one JSON object per line — standard for HuggingFace datasets)
    train_path = config.PROCESSED_DATA_DIR / "finetune_train.jsonl"
    val_path = config.PROCESSED_DATA_DIR / "finetune_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Saved %d train samples → %s", len(train_data), train_path)
    logger.info("Saved %d val samples → %s", len(val_data), val_path)
    return train_path


if __name__ == "__main__":
    prepare()
