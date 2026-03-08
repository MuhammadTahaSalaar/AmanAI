"""AmanAI — NUST Bank Customer Service Chatbot (Streamlit UI)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import torch

import config
from src.data_processing.base_processor import Document
from src.data_processing.etl_pipeline import ETLPipeline
from src.rag_engine.embedder import Embedder
from src.rag_engine.vector_store import VectorStore
from src.rag_engine.bm25_retriever import BM25Retriever
from src.rag_engine.hybrid_retriever import HybridRetriever
from src.rag_engine.reranker import Reranker
from src.rag_engine.rag_chain import RAGChain
from src.llm.model_loader import ModelLoader
from src.guardrails.safety_manager import SafetyManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="centered",
)


# ── Initialization (cached so components load once) ──────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def _load_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="Initializing vector store...")
def _load_vector_store(_embedder: Embedder) -> VectorStore:
    return VectorStore(embedder=_embedder)


@st.cache_resource(show_spinner="Loading LLM (this may take a moment)...")
def _load_model() -> ModelLoader | None:
    """Load model; returns None if loading fails (e.g., in pure CPU demo mode)."""
    try:
        loader = ModelLoader()
        loader.load()
        return loader
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load LLM: %s", exc)
        return None


@st.cache_resource(show_spinner="Initializing re-ranker...")
def _load_reranker() -> Reranker:
    return Reranker()


@st.cache_resource(show_spinner="Initializing safety guardrails...")
def _load_safety() -> SafetyManager:
    return SafetyManager()


def _run_etl_and_index(vector_store: VectorStore) -> list[Document]:
    """Run ETL pipeline and index documents in both vector and BM25 stores."""
    pipeline = ETLPipeline()
    documents = pipeline.run()

    # Index into vector store if empty
    if vector_store.count == 0:
        vector_store.add_documents(documents)
        logger.info("Indexed %d documents into vector store", len(documents))
    else:
        logger.info("Vector store already has %d documents; skipping indexing", vector_store.count)

    return documents


def _build_rag_chain(
    documents: list[Document],
    vector_store: VectorStore,
    model_loader: ModelLoader,
    reranker: Reranker,
) -> RAGChain:
    """Assemble the full RAG chain from component parts."""
    bm25 = BM25Retriever()
    bm25.index(documents)

    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
    )

    return RAGChain(
        retriever=hybrid,
        reranker=reranker,
        model_loader=model_loader,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render sidebar with info and document upload."""
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        st.markdown(
            "Welcome to **AmanAI**, your intelligent banking assistant.\n\n"
            "Ask me about savings accounts, term deposits, fund transfers, "
            "profit rates, and more."
        )
        st.divider()

        # Document upload for real-time ingestion
        st.subheader("Upload New Documents")
        st.caption(
            "Add new FAQs or policy documents to the knowledge base instantly. "
            "Supports JSON (structured Q&A) or plain text (.txt)."
        )
        uploaded = st.file_uploader(
            "Choose a file",
            type=["json", "txt"],
            key="doc_upload",
        )
        if uploaded is not None:
            _handle_upload(uploaded)

        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def _handle_upload(uploaded_file) -> None:
    """Process an uploaded JSON or TXT file and add to the knowledge base."""
    try:
        raw = uploaded_file.read()
        filename = uploaded_file.name
        new_docs: list[Document] = []

        if filename.endswith(".json"):
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "content" in item:
                        new_docs.append(Document.from_dict(item))
            elif isinstance(data, dict) and "content" in data:
                new_docs.append(Document.from_dict(data))

        elif filename.endswith(".txt"):
            text = raw.decode("utf-8").strip()
            if text:
                # Split on double-newlines to create paragraph-level documents
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    new_docs.append(
                        Document(
                            content=para,
                            metadata={"source": filename, "type": "uploaded_document"},
                        )
                    )

        if new_docs:
            vs = st.session_state.get("vector_store")
            if vs:
                vs.add_documents(new_docs)
                # Also re-index BM25 with new docs appended
                if "documents" in st.session_state:
                    st.session_state["documents"].extend(new_docs)
                    # Trigger BM25 rebuild on next interaction
                    if "rag_chain" in st.session_state:
                        del st.session_state["rag_chain"]
                st.sidebar.success(
                    f"✅ Added {len(new_docs)} document(s) from **{filename}**. "
                    "You can now ask questions about this content."
                )
                logger.info("Uploaded %d docs from %s", len(new_docs), filename)
        else:
            st.sidebar.warning("No valid documents found in the uploaded file.")
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        st.sidebar.error(f"Failed to parse uploaded file: {e}")


# ── Main Chat Interface ─────────────────────────────────────────────────────
def main() -> None:
    """Main application entry point."""
    _render_sidebar()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Hardware info banner
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    model_label = config.LLM_MODEL_NAME if torch.cuda.is_available() else config.CPU_FALLBACK_MODEL
    st.caption(f"🖥️ Running on: **{gpu_info}** | Model: `{model_label}`")

    # Load components
    embedder = _load_embedder()
    vector_store = _load_vector_store(embedder)
    st.session_state["vector_store"] = vector_store

    model_loader = _load_model()
    reranker = _load_reranker()
    safety = _load_safety()

    if model_loader is None:
        st.warning(
            "⚠️ LLM could not be loaded on this machine. "
            "Responses will show retrieved context only. "
            "For full generation, run on Hydra (`sbatch scripts/run_app.slurm`)."
        )

    # Run ETL and index on first load
    if "documents" not in st.session_state:
        with st.spinner("Processing NUST Bank knowledge base..."):
            documents = _run_etl_and_index(vector_store)
            st.session_state["documents"] = documents
    else:
        documents = st.session_state["documents"]

    # Build RAG chain (BM25 is in-memory, rebuilt per session)
    if "rag_chain" not in st.session_state:
        if model_loader is not None:
            rag_chain = _build_rag_chain(documents, vector_store, model_loader, reranker)
            st.session_state["rag_chain"] = rag_chain
        else:
            st.session_state["rag_chain"] = None
    rag_chain = st.session_state["rag_chain"]

    # Display chat header
    st.title(f"{config.APP_ICON} AmanAI")
    st.markdown("*Your intelligent NUST Bank customer service assistant*")
    st.divider()

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me about NUST Bank products and services...")

    if user_input:
        # Display user message (raw input visible to user)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Safety check — sanitized is PII-scrubbed version used for LLM/history
        is_safe, sanitized, rejection = safety.validate_input(user_input)

        # Store sanitized input in history so PII is never passed to the LLM
        # or recalled in subsequent turns (show raw to user above but store clean)
        st.session_state["messages"].append({"role": "user", "content": sanitized or user_input})

        retrieved_docs: list = []
        if not is_safe:
            response = rejection
        elif rag_chain is None:
            # No LLM: return retrieved context directly
            with st.spinner("Searching knowledge base..."):
                bm25 = BM25Retriever()
                bm25.index(documents)
                hybrid = HybridRetriever(vector_store=vector_store, bm25_retriever=bm25)
                reranker_tmp = _load_reranker()
                # Augment query for better retrieval
                retrieval_q = sanitized
                if "nust" not in sanitized.lower():
                    retrieval_q = f"NUST Bank {sanitized}"
                candidates = hybrid.retrieve(retrieval_q)
                reranked, _ = reranker_tmp.rerank(retrieval_q, candidates)
                if reranked:
                    ctx_parts = [f"**[{i+1}]** {d.content}" for i, d in enumerate(reranked[:3])]
                    response = (
                        "*(LLM not available — showing retrieved context)*\n\n"
                        + "\n\n".join(ctx_parts)
                    )
                    retrieved_docs = reranked
                else:
                    response = (
                        "I'm sorry, I couldn't find relevant information for your query. "
                        "Please contact NUST Bank at +92 (51) 111 000 494."
                    )
                    retrieved_docs = []
        else:
            # Generate response with full RAG pipeline
            with st.spinner("Searching knowledge base..."):
                raw_response, retrieved_docs = rag_chain.query(
                    user_query=sanitized,
                    chat_history=st.session_state["messages"][:-1],
                )
                response = safety.sanitize_output(raw_response)

        # Display assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            # Show retrieved chunks in a collapsible expander
            if retrieved_docs:
                with st.expander("🔍 Retrieved Knowledge Base Chunks", expanded=False):
                    for i, doc in enumerate(retrieved_docs, 1):
                        source = doc.metadata.get(
                            "product",
                            doc.metadata.get("source_sheet", "Unknown"),
                        )
                        category = doc.metadata.get("category", "")
                        label = f"**[{i}] {source}**"
                        if category:
                            label += f"  ·  {category}"
                        st.markdown(label)
                        st.caption(doc.content)
                        st.divider()


if __name__ == "__main__":
    main()
