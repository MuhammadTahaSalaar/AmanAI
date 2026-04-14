"""AmanAI — NUST Bank Customer Service Chatbot (Streamlit UI)."""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from io import StringIO

# Suppress non-critical warnings for cleaner startup
# ONNX Runtime GPU discovery (expected in WSL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.quantizers")
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", message=".*__path__.*models.*image_processing.*")

# Suppress Presidio language registry warnings
presidio_logger = logging.getLogger("presidio-analyzer")
presidio_logger.setLevel(logging.ERROR)


# Custom stderr wrapper to filter transformers __path__ messages
class FilteredStderr:
    """Filters out harmless transformers __path__ messages from stderr."""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
    
    def write(self, message: str) -> int:
        """Filter and write to stderr."""
        # Skip the transformers __path__ aliasing messages completely
        if "Accessing `__path__`" in message and "image_processing" in message:
            return len(message)
        # Skip GPU discovery errors from ONNX  
        if "GPU device discovery failed" in message:
            return len(message)
        # Pass everything else through
        self.original_stderr.write(message)
        return len(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)


# Apply stderr filter before any imports
sys.stderr = FilteredStderr(sys.stderr)

import streamlit as st
import torch

import config
from src.auth.auth_manager import AuthManager
from src.auth.session_manager import SessionDocumentManager
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


# ── Authentication & Session Management ──────────────────────────────────────
def _render_login_screen() -> None:
    """Render the login screen."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title(f"{config.APP_ICON} AmanAI")
        st.markdown("## Login")
        st.caption("*NUST Bank Customer Service Chatbot*")
        st.divider()
        
        # Username selection
        username = st.selectbox(
            "Select Account",
            [config.GUEST_USER, config.ADMIN_USER],
            format_func=lambda x: f"{x.capitalize()} Account",
        )
        
        # Password input (only for admin)
        password = ""
        if username == config.ADMIN_USER:
            password = st.text_input("Password", type="password")
        
        # Login button
        if st.button("Login", use_container_width=True, type="primary"):
            auth_manager = AuthManager()
            is_valid, error_msg = auth_manager.authenticate(username, password)
            
            if is_valid:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["is_admin"] = auth_manager.is_admin(username)
                st.session_state["session_docs"] = SessionDocumentManager()
                st.session_state["messages"] = []
                logger.info("User %s logged in", username)
                st.rerun()
            else:
                st.error(f" {error_msg}")
        
        st.divider()
        st.caption("**Guest:** No password required\n**Admin:** Password required")


# ── Sidebar ──────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render sidebar with info, user details, document upload, and logout."""
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        
        # User info
        username = st.session_state.get("username", "Unknown")
        is_admin = st.session_state.get("is_admin", False)
        user_badge = "👤 Admin" if is_admin else "👁️ Guest"
        st.markdown(f"**{user_badge}** · `{username}`")
        st.divider()
        
        st.markdown(
            "Welcome to **AmanAI**, your intelligent banking assistant.\n\n"
            "Ask me about savings accounts, term deposits, fund transfers, "
            "profit rates, and more."
        )
        st.divider()

        # Document upload for admin only
        if is_admin:
            st.subheader("📤 Add Documents (Admin)")
            st.caption(
                "Upload new FAQs or documents to the knowledge base. "
                "Changes persist for this session only. "
                "Supports JSON (structured Q&A) or plain text (.txt)."
            )
            uploaded = st.file_uploader(
                "Choose a file",
                type=["json", "txt"],
                key="doc_upload",
            )
            if uploaded is not None:
                _handle_admin_upload(uploaded)
            
            sess_docs = st.session_state.get("session_docs")
            if sess_docs and sess_docs.get_document_count() > 0:
                st.success(
                    f" {sess_docs.get_document_count()} session document(s) loaded"
                )
            st.divider()

        # Chat history clear
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
        
        st.divider()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = ""
            st.session_state["is_admin"] = False
            st.session_state["session_docs"] = None
            st.session_state["messages"] = []
            logger.info("User logged out")
            st.rerun()


def _handle_admin_upload(uploaded_file) -> None:
    """Process an uploaded JSON or TXT file for admin and add to session documents."""
    sess_docs = st.session_state.get("session_docs")
    if not sess_docs:
        st.sidebar.error("Session document manager not initialized")
        return

    # Track processed files to prevent re-processing on rerun
    if "processed_uploads" not in st.session_state:
        st.session_state["processed_uploads"] = set()
    
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_id in st.session_state["processed_uploads"]:
        return  # Skip duplicate processing
    
    try:
        # Save to temp file and parse
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Record count before parsing so we can get only the newly added docs
        prev_count = sess_docs.get_document_count()

        success, message = sess_docs.parse_and_add_file(tmp_path)
        
        # Clean up
        Path(tmp_path).unlink()
        
        if success:
            st.session_state["processed_uploads"].add(file_id)
            st.sidebar.success(f" {message}")
            logger.info("Admin uploaded document: %s", uploaded_file.name)
            
            # Get only the newly added documents (avoid re-indexing old ones)
            all_session_docs = sess_docs.get_documents()
            new_documents = all_session_docs[prev_count:]
            
            # Update RAG chain's retriever indexes with only the new documents
            if new_documents and "rag_chain" in st.session_state and st.session_state["rag_chain"] is not None:
                try:
                    st.session_state["rag_chain"].update_retriever_with_documents(new_documents)
                    logger.info("Updated RAG chain retriever with %d new session documents", len(new_documents))
                except Exception as e:
                    logger.error("Failed to update RAG chain retriever: %s", str(e))
        else:
            st.sidebar.error(f" {message}")
    except Exception as e:
        st.sidebar.error(f" Error uploading file: {str(e)}")
        logger.error("Admin upload failed: %s", str(e))


# ── Main Chat Interface ─────────────────────────────────────────────────────
def main() -> None:
    """Main application entry point."""
    # Check if user is authenticated
    if not st.session_state.get("authenticated", False):
        _render_login_screen()
        return

    # User is authenticated, show chat interface
    _render_sidebar()

    # Load components (cached)
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
        )
    else:
        device_info = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.success(
            f"LLM loaded successfully on {device_info} | "
            f"Model: Llama 3.2 3B + QLoRA LoRA adapter"
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
                
                # Augment with session documents
                session_docs_list = st.session_state.get("session_docs")
                if session_docs_list:
                    candidates.extend(session_docs_list.get_documents())
                
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
                    session_documents=None,  # Session documents are already indexed via upload handler
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
