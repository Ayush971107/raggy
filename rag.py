# app.py
# ---------------------------------------------------------------
# Streamlit RAG over any PDF using LlamaIndex + OpenAI
# - Loads OPENAI_API_KEY from .env (no settings UI)
# - Hardcoded params at top
# - Robust response + source handling across LlamaIndex versions
# ---------------------------------------------------------------

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ---------- Hardcoded config ----------
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MAX_CHARS_PER_CHUNK_DISPLAY = 1500

# ---------- Env / API key ----------
load_dotenv()  # reads .env from current directory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Create a .env with OPENAI_API_KEY=... in this folder.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # ensure sub-libs see it

# ---------- LlamaIndex global settings ----------
Settings.llm = OpenAI(model=MODEL_NAME, temperature=0)
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="PDF RAG (LlamaIndex + OpenAI)", page_icon="📄", layout="wide")
st.title("📄🔎 PDF RAG (LlamaIndex + OpenAI)")
st.caption(
    f"Model: {MODEL_NAME} · Embeddings: {EMBED_MODEL} · "
    f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}, top-k: {TOP_K}"
)

# Session state
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Upload
st.subheader("1) Upload PDF")
uploaded_pdf = st.file_uploader("Drop a .pdf here", type=["pdf"])

# Build index
st.subheader("2) Build the index")
build_btn = st.button("⚒️ Build / Rebuild RAG Index", disabled=(uploaded_pdf is None))

def _node_text(n):
    """Robustly extract text from NodeWithScore across LlamaIndex versions."""
    try:
        # Newer versions often expose .text directly
        if hasattr(n, "text") and n.text:
            return n.text
        # Older: NodeWithScore.node.get_content()
        node = getattr(n, "node", None)
        if node is not None:
            if hasattr(node, "get_content"):
                return node.get_content(metadata_mode="none")
            # Fallback: some nodes expose .text
            if hasattr(node, "text"):
                return node.text
    except Exception:
        pass
    return ""

def _response_text(resp):
    """Handle different response interfaces: .response, .text, or str(resp)."""
    for attr in ("response", "text"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if isinstance(val, str) and val.strip():
                return val
    return str(resp)

def build_index_from_pdf(pdf_bytes: bytes, filename: str):
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(pdf_bytes)

        # Parse PDF into LlamaIndex Documents
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()

        # Build vector index with splitter transform
        index = VectorStoreIndex.from_documents(docs, transformations=[splitter])

    # Create RAG query engine
    st.session_state.query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
    )
    st.success("Index built successfully. Proceed to ask a question.")

if build_btn and uploaded_pdf:
    try:
        build_index_from_pdf(uploaded_pdf.read(), uploaded_pdf.name)
    except Exception as e:
        st.error("Failed to build the index.")
        st.exception(e)

# Ask questions
st.subheader("3) Ask a question")
question = st.text_input("Your question", placeholder="e.g., Summarize the main findings in section 3…")
ask_btn = st.button("💬 Ask")

if ask_btn:
    if st.session_state.query_engine is None:
        st.warning("Please upload a PDF and build the index first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking…"):
                response = st.session_state.query_engine.query(question)

            # Answer
            st.markdown("### 🧠 Answer")
            st.write(_response_text(response))

        except Exception as e:
            st.error("Query failed.")
            st.exception(e)
