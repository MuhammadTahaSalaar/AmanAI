"""Prepare fine-tuning data from the processed NUST Bank documents.

Converts ETL output into high-quality instruction-following pairs with diverse
question phrasings and natural-sounding answers in Llama 3.2 chat format.

Run this BEFORE finetune.py.

Usage:
    python -m src.llm.prepare_finetune_data
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import config
from src.data_processing.etl_pipeline import ETLPipeline
from src.data_processing.base_processor import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FINETUNE_DATA_PATH = config.PROCESSED_DATA_DIR / "finetune_dataset.json"

# Use the SAME system instruction style as the inference prompt so the model
# learns to follow the grounding rules it will encounter at inference time.
SYSTEM_INSTRUCTION = (
    "You are AmanAI, NUST Bank's customer service assistant. "
    "Answer questions accurately using ONLY the information provided about NUST Bank's products and policies. "
    "Never invent facts, rates, or eligibility criteria. "
    "If a user describes a person by age or demographic, recommend the matching product: "
    "under 18 → Little Champs Account, 55+ → NUST Waqaar Account, "
    "freelancers → NUST Freelancer Digital Account, women → NUST Sahar Account, "
    "new to banking → NUST Asaan Digital Account, non-resident → Roshan Digital Account. "
    "If you don't have the answer, say so politely and direct the user to call +92 (51) 111 000 494."
)


def _clean_answer(text: str) -> str:
    """Clean raw document content into a natural answer format."""
    # Strip raw Q:/A: prefixes and [ProductName] tags
    text = re.sub(r"^\[.*?\]\s*", "", text)
    text = re.sub(r"^Q:\s*.*?\n", "", text)
    text = re.sub(r"^A:\s*", "", text)
    # Convert decimal rates to percentages (e.g., 0.19 → 19%)
    text = re.sub(
        r"\b0\.(\d{2})\b",
        lambda m: f"{int(m.group(1))}%",
        text,
    )
    return text.strip()


def _generate_question_variants(base_question: str, product: str, doc_type: str) -> list[str]:
    """Generate diverse question phrasings from a base question."""
    variants = [base_question]
    q_lower = base_question.lower()

    # Add product-contextual variants
    if doc_type == "rate_info" or "rate" in q_lower or "profit" in q_lower:
        variants.extend([
            f"What is the profit rate for {product}?",
            f"Tell me about {product} rates.",
            f"How much profit does {product} offer?",
        ])
    elif doc_type in ("description", "descriptive_block"):
        variants.extend([
            f"What is {product}?",
            f"Tell me about {product}.",
            f"What can you tell me about {product}?",
            f"Give me details on {product}.",
        ])
    elif "eligib" in q_lower or "open" in q_lower or "require" in q_lower:
        variants.extend([
            f"Who can open a {product}?",
            f"What are the requirements for {product}?",
            f"Am I eligible for {product}?",
        ])
    elif "document" in q_lower:
        variants.extend([
            f"What documents do I need for {product}?",
            f"What papers are required for {product}?",
        ])

    return variants


# Demographic→product training examples to teach age/persona reasoning
_DEMOGRAPHIC_EXAMPLES: list[dict] = [
    # Children / minors
    {"q": "Which account is suitable for my 8 year old child?", "a": "The Little Champs Account is designed specifically for minors below 18 years of age. Your 8-year-old child would be eligible. A parental or legal guardian is required to open this account.", "product": "Little Champs Account"},
    {"q": "My brother is 14, which account should I open for him?", "a": "Since your brother is 14 years old, the Little Champs Account would be the best fit. It is designed for minors below 18 years of age and offers a debit card, chequebook, and profit payment.", "product": "Little Champs Account"},
    {"q": "Can I open an account for my 12 year old daughter?", "a": "Yes, the Little Champs Account is perfect for your 12-year-old daughter. It is designed for minors below 18 years of age. You would need to act as the parental/legal guardian to open it.", "product": "Little Champs Account"},
    {"q": "What account do you recommend for a teenager?", "a": "For a teenager (under 18 years of age), the Little Champs Account is the recommended option. It is specifically designed for minors and offers attractive features including a debit card and profit payment.", "product": "Little Champs Account"},
    {"q": "Account for someone who is 17 years old?", "a": "Since the person is 17 years old (under 18), the Little Champs Account is the appropriate choice. It is designed specifically for minors.", "product": "Little Champs Account"},
    # Senior citizens
    {"q": "My grandmother is 60, which account is best for her?", "a": "The NUST Waqaar Account is designed specifically for senior citizens aged 55 and above. Since your grandmother is 60, she is eligible for this account.", "product": "NUST Waqaar Account"},
    {"q": "Best account for a retired person aged 58?", "a": "The NUST Waqaar Account is designed for senior citizens aged 55 and above. At 58 years old, a retired person qualifies for this account.", "product": "NUST Waqaar Account"},
    {"q": "Account for my grandfather who is 70?", "a": "The NUST Waqaar Account is ideal for your grandfather. It is designed specifically for senior citizens aged 55 and above.", "product": "NUST Waqaar Account"},
    {"q": "Which account for someone who is exactly 55 years old?", "a": "The NUST Waqaar Account is designed for senior citizens aged 55 and above. A 55-year-old meets the eligibility criteria.", "product": "NUST Waqaar Account"},
    # Freelancers
    {"q": "Best account for a freelance web developer?", "a": "The NUST Freelancer Digital Account is exclusively designed for freelancers. It offers current accounts in PKR and foreign currencies (USD, GBP, EUR, AED) to receive international payments directly.", "product": "NUST Freelancer Digital Account"},
    {"q": "I'm a freelancer, which account should I open?", "a": "The NUST Freelancer Digital Account is designed exclusively for freelancers like you. It allows you to receive payments in PKR and foreign currencies including USD, GBP, EUR, and AED.", "product": "NUST Freelancer Digital Account"},
    # New to banking
    {"q": "I've never had a bank account before, where do I start?", "a": "The NUST Asaan Digital Account is perfect for you. It is a simple digital banking account designed for individuals who are new to banking or the unbanked segment.", "product": "NUST Asaan Digital Account"},
    {"q": "What account is best for someone new to banking?", "a": "The NUST Asaan Digital Account is designed specifically for individuals who are new to banking. It offers a simple and accessible way to start your banking journey.", "product": "NUST Asaan Digital Account"},
    # Women
    {"q": "What accounts do you have for women?", "a": "The NUST Sahar Account is designed specifically for women. It offers Current, Savings, Term Deposit, and Asaan Current account variants.", "product": "NUST Sahar Account"},
    # Out-of-domain
    {"q": "Can you help me with Bitcoin trading?", "a": "I can only assist with NUST Bank banking queries. For other questions, please use an appropriate service. For banking help, call +92 (51) 111 000 494.", "product": "OOD"},
    {"q": "What is the weather today?", "a": "I can only help with NUST Bank products and services. For other inquiries, please try another service.", "product": "OOD"},
    # Helpline/delete account
    {"q": "How do I delete my account?", "a": "To delete or deactivate your account, please call NUST Bank's helpline at +92 (51) 111 000 494 for assistance.", "product": "General"},
    {"q": "What is the NUST Bank helpline number?", "a": "You can reach NUST Bank's helpline at +92 (51) 111 000 494.", "product": "General"},
]


def _qa_pairs_from_documents(documents: list[Document]) -> list[dict]:
    """Convert documents into diverse instruction-following pairs."""
    pairs: list[dict] = []

    for doc in documents:
        content = doc.content.strip()
        if not content or len(content) < 20:
            continue

        doc_type = doc.metadata.get("type", "")
        product = doc.metadata.get("product", doc.metadata.get("category", "NUST Bank"))
        cleaned = _clean_answer(content)

        if not cleaned or len(cleaned) < 15:
            continue

        if doc_type == "qa_pair":
            lines = content.split("\n", 1)
            if len(lines) == 2:
                question_line = lines[0].replace("Q:", "").strip()
                answer_line = _clean_answer(lines[1])
                if question_line and answer_line and len(answer_line) > 15:
                    # Generate question variants
                    variants = _generate_question_variants(question_line, product, doc_type)
                    for q in variants:
                        pairs.append({
                            "instruction": q,
                            "input": "",
                            "output": answer_line,
                            "source": product,
                        })

        elif doc_type == "rate_info" or "rate" in content.lower():
            for q in [
                f"Tell me about {product} rates or charges.",
                f"What is the profit rate for {product}?",
                f"What rates does {product} offer?",
            ]:
                pairs.append({
                    "instruction": q,
                    "input": "",
                    "output": cleaned,
                    "source": product,
                })

        elif doc_type in ("description", "descriptive_block"):
            for q in [
                f"What can you tell me about {product}?",
                f"What is {product}?",
                f"Tell me about {product}.",
            ]:
                pairs.append({
                    "instruction": q,
                    "input": "",
                    "output": cleaned,
                    "source": product,
                })

        else:
            if len(cleaned) > 40:
                pairs.append({
                    "instruction": f"What does NUST Bank offer regarding {product}?",
                    "input": "",
                    "output": cleaned,
                    "source": product,
                })

    # Add demographic→product reasoning examples
    for ex in _DEMOGRAPHIC_EXAMPLES:
        pairs.append({
            "instruction": ex["q"],
            "input": "",
            "output": ex["a"],
            "source": ex["product"],
        })

    return pairs


def _format_as_chat(pair: dict) -> dict:
    """Format an Alpaca pair into Llama 3.2 chat format for Unsloth."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": pair["instruction"]},
        {"role": "assistant", "content": pair["output"]},
    ]
    return {"messages": messages, "source": pair.get("source", "")}


def _validate_pairs(pairs: list[dict]) -> list[dict]:
    """Validate and filter training pairs for quality."""
    valid = []
    seen: set[str] = set()
    for p in pairs:
        q = p["instruction"].strip()
        a = p["output"].strip()
        # Skip empty or too-short
        if not q or not a or len(a) < 15:
            continue
        # Skip duplicates (same question+answer)
        key = f"{q.lower()}||{a[:100].lower()}"
        if key in seen:
            continue
        seen.add(key)
        valid.append(p)
    return valid


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
    logger.info("Generated %d raw instruction pairs", len(pairs))

    # Validate and deduplicate
    pairs = _validate_pairs(pairs)
    logger.info("After validation: %d instruction pairs", len(pairs))

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
