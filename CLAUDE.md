# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AmanAI is a RAG-based LLM customer-service chatbot for NUST Bank (Pakistan), delivered as a Streamlit app. The pipeline is: **user → safety guardrails → hybrid retrieval (BM25 + ChromaDB) → FlashRank rerank → Llama 3.2 (+ optional QLoRA adapter) → output PII scrub → user.** It runs in two hardware modes — a GPU mode (4-bit quantized 3B model) targeting the VUB **Hydra SLURM cluster**, and a CPU fallback mode (float32 1B model) for laptops.

## Commands

```bash
# Local setup (after `conda create -n amanai python=3.10 && conda activate amanai`)
bash setup.sh                              # pip install + downloads spaCy en_core_web_lg

# Build the knowledge base (ETL → data/processed/*.json). Run once before first app launch.
python -m src.data_processing.etl_pipeline

# Run the app
streamlit run app.py                       # serves on :8501

# Tests
pytest tests/ -v                           # full suite
pytest tests/test_guardrails.py -v         # one module
pytest tests/test_guardrails.py::TestSafetyManager -v   # one class
pytest --cov=src --cov-report=term-missing # with coverage

# Evaluation (Ragas against evaluation/golden_dataset.json → evaluation/evaluation_results.json)
python evaluation/evaluate.py

# Fine-tuning (GPU required; two steps)
python -m src.llm.prepare_finetune_data    # ETL docs → data/processed/finetune_{train,val}.jsonl
python -m src.llm.finetune --epochs 3 --output-dir data/lora_adapter
```

**Hydra (SLURM) workflow** — never run training/serving directly on a login node:
```bash
bash scripts/setup_hydra.sh    # ONCE on a login node: creates mamba env, installs torch+unsloth
sbatch scripts/finetune.sh     # QLoRA fine-tune job (prepares data + trains)
sbatch scripts/run_app.sh      # serves Streamlit on a GPU node; SSH-tunnel :8501 to view
```

**Docker** runs CPU-only by design — `docker-compose.yml` forces `EMBEDDING_DEVICE=cpu` and caps memory at 8G. `docker compose up` serves the app on :8501.

## Configuration model

`config.py` is the single source of truth — **no magic numbers in business logic**, everything reads from `config`. Every value is overridable via environment variables loaded from `.env` (see `.env.example`). Note that defaults in `config.py`, the README table, and `.env.example` intentionally differ (e.g. `CHUNK_SIZE` is 256 in `config.py` but 800 in `.env.example`); the **effective** value at runtime is the env var if set, else the `config.py` default. When changing tunables, edit `config.py` and/or `.env`, not the call sites.

Key env vars: `LLM_MODEL_NAME` (GPU 4-bit model), `CPU_FALLBACK_MODEL`, `LORA_ADAPTER_PATH` (set this to `data/lora_adapter` to load a fine-tuned adapter), `EMBEDDING_DEVICE` (`cuda`/`cpu`), `HF_TOKEN` (required for gated Llama models), `BM25_WEIGHT`/`VECTOR_WEIGHT`, `RETRIEVAL_TOP_K`/`RERANK_TOP_K`.

## Architecture

The app (`app.py`) wires components together; all heavy objects are cached via `@st.cache_resource` so models load once per process. Components are dependency-injected (constructors accept optional collaborators, defaulting to real implementations) — this is the seam used by tests.

**Data layer** (`src/data_processing/`) — an ETL pipeline (`etl_pipeline.py`) orchestrates four sources into a unified `list[Document]` (`base_processor.Document`, content + metadata):
- Rate sheet (`rate_sheet_processor.py`) — Excel "Rate Sheet July 1 2024" → natural-language rate sentences.
- Product FAQ sheets (`faq_sheet_processor.py`) — remaining Excel sheets (skips `SKIP_SHEETS`) → Q&A docs.
- App FAQ JSON (`json_processor.py`).
- Runtime documents — any JSON dropped in `data/runtime_document/` (FAQ-category format), loaded at startup.

**Retrieval layer** (`src/rag_engine/`) — `Embedder` (BAAI/bge-small-en-v1.5, 384-dim) feeds `VectorStore` (persistent ChromaDB at `data/chroma_db`). `BM25Retriever` is **in-memory and rebuilt every session** from the ETL docs (it is not persisted — this is why `app.py` re-runs ETL/indexing on load). `HybridRetriever` fuses both via Reciprocal Rank Fusion (`BM25_WEIGHT`/`VECTOR_WEIGHT`). `Reranker` (FlashRank cross-encoder) produces the final top-k; `MIN_RELEVANCE_SCORE` (reranker.py) is the out-of-domain threshold.

**Orchestration** (`rag_chain.py`) is the most nuanced file — read it before changing retrieval behavior. It does, in order:
1. **Query augmentation** — prefixes "NUST Bank" and, for follow-up questions lacking a product name, injects the product from the previous turn's question (regex `_PRODUCT_RE`).
2. **Two-tier OOD gate** — if the query contains any `_BANKING_KEYWORDS`, it's trusted on-topic and reranked with the augmented query; otherwise it's reranked with the *original* query and rejected with `_OOD_RESPONSE` if the top score is below `MIN_RELEVANCE_SCORE`.
3. **Product filtering** — when exactly one product is named, drops chunks from other products so the small model can't mix product data (skipped for multi-product comparisons).
4. Builds context + prompt (`prompt_templates.py`), generates, and forces a grounding/helpline fallback when no context was found.

**LLM** (`src/llm/model_loader.py`) — `ModelLoader.load()` auto-selects GPU (4-bit NF4 via `BitsAndBytesConfig`) vs CPU (float32 fallback model) from `torch.cuda.is_available()`. A LoRA adapter is merged **only on GPU** (it was trained for the GPU model; CPU loads skip it). `generate()` strips `<|...|>` control markers from output. If the model fails to load, `app.py` degrades to **retrieval-only mode** (shows reranked context, no generation).

**Guardrails** (`src/guardrails/`) — `SafetyManager` runs input through, in order: control-char strip → length check → empty check → `JailbreakDetector` (regex) → `SemanticSafetyDetector` → `PIIAnonymizer` (Presidio + custom CNIC/IBAN regex). **Critical PII invariant:** the raw user message is shown in the UI but only the **sanitized** (PII-scrubbed) text is stored in chat history and ever passed to the LLM — preserve this when touching `app.py`'s message flow. `sanitize_output()` re-scrubs the LLM response. All blocks are recorded by `AuditLogger` to `logs/security_audit.log`.

**Auth** (`src/auth/`) — simple guest/admin login (`AuthManager`, password from `ADMIN_PASSWORD`). Admins can upload `.json`/`.txt` docs at runtime (`SessionDocumentManager`); uploads are indexed into the live retriever via `RAGChain.update_retriever_with_documents()` and persist for the session only.

## Conventions

- Python 3.10. `from __future__ import annotations` at the top of modules using `X | None` syntax.
- All modules log via `src.utils.logger.setup_logger(__name__)` — do not use `print()` in `src/` (the ETL `__main__` block is the one intentional exception).
- New retrieval/guardrail/LLM components follow the constructor-injection pattern (optional collaborator args) to stay testable; mirror it.
- Tests live in `tests/`, mirror `src/` module names, and rely on `conftest.py` putting the project root on `sys.path`. Markers are registered in `pytest.ini` under `--strict-markers`: each module sets a module-level `pytestmark` (`unit` or `integration`), so select subsets with `pytest -m unit` / `pytest -m integration`. New test modules must set `pytestmark` and use only registered markers.
