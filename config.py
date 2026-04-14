"""AmanAI - Centralized Configuration

All magic numbers, file paths, model names, and hyperparameters are defined here.
No hardcoded values in business logic.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")

DATASET_DIR = PROJECT_ROOT / "dataset"
EXCEL_FILE = DATASET_DIR / "NUST Bank-Product-Knowledge.xlsx"
FAQ_JSON_FILE = DATASET_DIR / "funds_transfer_app_features_faq.json"

# ── LLM ───────────────────────────────────────────────────────────────────────
# GPU model: 4-bit NF4 quantized via bitsandbytes (requires CUDA)
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
)
# CPU fallback: loaded in float32 without quantization
CPU_FALLBACK_MODEL = os.getenv(
    "CPU_FALLBACK_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "data/lora_adapter")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "nust_bank_docs")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# ── Guardrails ────────────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "3"))

# ── Rate Sheet ────────────────────────────────────────────────────────────────
RATE_SHEET_NAME = "Rate Sheet July 1 2024"
RATE_SHEET_EFFECTIVE_DATE = "July 1st, 2024"

# ── Sheets to Skip ────────────────────────────────────────────────────────────
SKIP_SHEETS = {"Main", "Sheet1"}

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = "AmanAI — NUST Bank Customer Service"
APP_ICON = "🏦"

# ── Authentication ────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
GUEST_USER = "guest"
ADMIN_USER = "admin"
