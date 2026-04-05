# 🏦 AmanAI — NUST Bank Customer Service Agent

A production-grade, RAG-infused LLM agent for NUST Bank (Pakistan). Built with Llama 3.2 3B, ChromaDB, hybrid BM25+Vector retrieval, FlashRank re-ranking, and comprehensive safety guardrails.

## Architecture

```
User → Streamlit UI → Safety Manager → Hybrid Retriever → Reranker → LLM → Safety Output → User
                                          ├── BM25 (sparse)
                                          └── ChromaDB (dense)
```

**Key Components:**
- **LLM**: Llama 3.2 3B Instruct (4-bit NF4 quantized via bitsandbytes)
- **Embeddings**: BAAI/bge-small-en-v1.5 (384-dim)
- **Vector Store**: ChromaDB (persistent, local)
- **Retrieval**: BM25 + Dense vector hybrid with Reciprocal Rank Fusion
- **Re-ranking**: FlashRank cross-encoder
- **Guardrails**: PII anonymization (Presidio + custom CNIC/IBAN regex), jailbreak detection
- **Fine-tuning**: QLoRA via Unsloth (LoRA rank=16, alpha=32)
- **UI**: Streamlit chat interface
- **Evaluation**: Ragas framework (faithfulness, relevancy, precision, recall)

## Project Structure

```
AmanAI/
├── app.py                          # Streamlit entry point
├── config.py                       # Centralized configuration
├── Dockerfile                      # Container build file
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
├── dataset/                        # Raw data files
├── data/                           # Processed data, vector store & LoRA adapters
├── scripts/
│   ├── setup_hydra.sh              # One-time Hydra env setup (mamba)
│   ├── finetune.slurm              # SLURM job: QLoRA fine-tuning on GPU
│   └── run_app.slurm               # SLURM job: Streamlit app on GPU node
├── src/
│   ├── data_processing/            # ETL pipeline & document processors
│   ├── rag_engine/                 # Embedder, vector store, BM25, hybrid retriever, reranker
│   ├── llm/
│   │   ├── model_loader.py         # GPU (4-bit NF4) + CPU (float32) model loading
│   │   ├── prompt_templates.py     # System & RAG prompt builders
│   │   ├── prepare_finetune_data.py # Converts ETL docs → instruction pairs
│   │   └── finetune.py             # Unsloth QLoRA fine-tuning script
│   ├── guardrails/                 # PII anonymizer, jailbreak detector, safety manager
│   └── utils/                      # Logging utilities
├── tests/                          # pytest test suite (34 tests)
├── evaluation/                     # Golden dataset & Ragas evaluation
└── documents/                      # Project documentation
```

## Quick Start

**Add this section to your README:**
> ## System Requirements
> * **OS:** Windows 10/11 with WSL2 (Windows Subsystem for Linux) **highly recommended**.
> * **GPU:** NVIDIA GPU with at least 8GB VRAM (RTX 3070 or better).
> * **Software:** Ensure NVIDIA drivers are installed on Windows; CUDA will be handled by the Python environment.
>
> ## Installation
> 1. Open your Ubuntu/WSL terminal.
> 2. Ensure you have Conda installed.
> 3. Create the environment: `conda create -n amanai python=3.10 -y && conda activate amanai`.
> 4. Run `bash setup.sh`.

## Fine-Tuning

The fine-tuning pipeline runs on Hydra (SLURM cluster) and uses Unsloth for efficient QLoRA training:

```bash
# Step 1: Prepare training data from ETL output
python -m src.llm.prepare_finetune_data
# Output: data/processed/finetune_train.jsonl, finetune_val.jsonl

# Step 2: Fine-tune (GPU required)
python -m src.llm.finetune --epochs 3 --output-dir data/lora_adapter
# Or via SLURM: sbatch scripts/finetune.slurm
```

After fine-tuning, set `LORA_ADAPTER_PATH=data/lora_adapter` in `.env` to load the adapter.

## Data Pipeline

The ETL pipeline processes three data sources:

1. **Rate Sheet** (`NUST Bank-Product-Knowledge.xlsx` → "Rate Sheet July 1 2024"): Savings rates, term deposit rates, FCY rates — converted to natural language sentences.
2. **Product FAQ Sheets** (remaining Excel sheets): Q&A pairs and descriptive text for each banking product.
3. **App FAQ JSON** (`funds_transfer_app_features_faq.json`): Mobile app features and fund transfer FAQs.

All documents are chunked, embedded, and indexed into both ChromaDB (dense) and BM25 (sparse) stores.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_guardrails.py -v
```

## Evaluation

```bash
# Run Ragas evaluation against golden dataset
python evaluation/evaluate.py
```

Results are saved to `evaluation/evaluation_results.json`.

## Configuration

All settings are centralized in `config.py` and can be overridden via environment variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | HuggingFace model ID |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `BM25_WEIGHT` | `0.4` | BM25 weight in hybrid retrieval |
| `VECTOR_WEIGHT` | `0.6` | Vector weight in hybrid retrieval |
| `RETRIEVAL_TOP_K` | `5` | Candidates from retrieval |
| `RERANK_TOP_K` | `2` | Final documents after re-ranking |

## License

Academic project — NUST University, 8th Semester LLM Course.
