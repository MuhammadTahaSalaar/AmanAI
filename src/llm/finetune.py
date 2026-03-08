"""QLoRA Fine-Tuning Script for AmanAI using Unsloth.

Fine-tunes Llama 3.2 3B Instruct on the NUST Bank dataset using
Parameter-Efficient Fine-Tuning (LoRA rank=16) via the Unsloth library.

Requirements:
    - CUDA GPU with ≥16GB VRAM (run on Hydra cluster)
    - unsloth installed (see scripts/setup_hydra.sh)
    - Fine-tuning data prepared via: python -m src.llm.prepare_finetune_data

Usage:
    python -m src.llm.finetune [--epochs N] [--output-dir PATH]

Outputs:
    - LoRA adapter weights saved to data/lora_adapter/
    - Training loss logged to data/training_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# PyTorch 2.10's torch._inductor has an internal import bug where
# mm_scaled.py references 'scaled_mm_configs' which doesn't exist in
# mm_common.py.  torch.compile triggers this path at import time via
# unsloth's @torch.compile decorators (both unsloth_zoo and models/llama.py).
# Disabling dynamo makes torch.compile a no-op, which is safe — unsloth's
# speed gains come from its custom CUDA kernels, not torch.compile.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Unsloth must be imported before transformers to apply kernel patches
try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except Exception as exc:
    import traceback
    print(f"[ERROR] Unsloth import failed ({type(exc).__name__}): {exc}")
    traceback.print_exc()
    sys.exit(1)

import torch
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # Base (non-quantized) for training
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0  # Unsloth recommends 0 for training stability
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

OUTPUT_DIR = str(config.DATA_DIR / "lora_adapter")
LOG_FILE = str(config.DATA_DIR / "training_log.json")


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """Load a .jsonl file as a HuggingFace Dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def formatting_func(examples: dict, tokenizer) -> list[str]:
    """Format chat examples into template strings for SFT.

    Unsloth/TRL calls this in two ways:
      - Validation probe: a single unbatched row  → messages is a list of dicts
      - Training:         a batched dict           → messages is a list of lists
    Always returns a list[str] as required by Unsloth's SFTTrainer.
    """
    messages = examples["messages"]
    # Single (unbatched) example: messages = [{role, content}, ...]
    if messages and isinstance(messages[0], dict):
        return [tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )]
    # Batched: messages = [[{role, content}, ...], ...]
    return [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
        )
        for convo in messages
    ]


def finetune(num_epochs: int = 3, output_dir: str = OUTPUT_DIR) -> None:
    """Run the full QLoRA fine-tuning pipeline.

    Args:
        num_epochs: Number of training epochs.
        output_dir: Directory to save LoRA adapter weights.
    """
    train_path = config.PROCESSED_DATA_DIR / "finetune_train.jsonl"
    val_path = config.PROCESSED_DATA_DIR / "finetune_val.jsonl"

    if not train_path.exists():
        logger.error("Training data not found at %s", train_path)
        logger.error("Run: python -m src.llm.prepare_finetune_data")
        sys.exit(1)

    logger.info("Loading model + LoRA scaffolding via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply Llama 3.2 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    logger.info("Loading fine-tuning datasets...")
    train_dataset = load_jsonl_as_dataset(train_path)
    val_dataset = load_jsonl_as_dataset(val_path)
    logger.info("Train: %d samples | Val: %d samples", len(train_dataset), len(val_dataset))

    # Compute warmup_steps from ratio (warmup_ratio removed in transformers 5.3)
    steps_per_epoch = max(1, len(train_dataset) // (TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_steps=warmup_steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",  # Disable wandb/mlflow for academic use
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=True,  # Sequence packing for efficiency
    )

    logger.info("Starting fine-tuning for %d epochs...", num_epochs)
    trainer_stats = trainer.train()

    # Save final LoRA adapter
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("LoRA adapter saved to %s", output_dir)

    # Save training log
    log_data = {
        "model": MODEL_NAME,
        "epochs": num_epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "final_train_loss": trainer_stats.training_loss,
        "adapter_path": output_dir,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info("Training log saved to %s", LOG_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmanAI QLoRA Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory for LoRA adapter output",
    )
    args = parser.parse_args()
    finetune(num_epochs=args.epochs, output_dir=args.output_dir)
