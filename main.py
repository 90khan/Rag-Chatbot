import os
import shutil
import streamlit as st

from app.file_processor import process_file
from app.vectorstore import VectorStore
from app.rag_pipeline import RAGPipeline
from app.graph_pipeline import GraphRAGPipeline

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="RAG + GraphRAG Chatbot", page_icon="🤖")

# ------------------------
# Login Check
# ------------------------
def check_credentials(username, password):
    env_user = os.getenv("APP_USER", "admin")
    env_pass = os.getenv("APP_PASS", "password")
    return username == env_user and password == env_pass

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.authenticated = True
            st.success("✅ Logged in!")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ------------------------
# Title
# ------------------------
st.title("📚 RAG + 🧠 GraphRAG Chatbot")

# ------------------------
# Vector Stores
# ------------------------
if "user_vectorstore" not in st.session_state:
    st.session_state.user_vectorstore = VectorStore(
        persist_path="data/user_store", load=True
    )

if "db_vectorstore" not in st.session_state:
    st.session_state.db_vectorstore = VectorStore(
        persist_path="data/db_store", load=True
    )

# ------------------------
# Sidebar Settings
# ------------------------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Number of docs to retrieve", 1, 50, 3)
max_tokens = st.sidebar.slider("Max tokens", 50, 500, 200, 10)

sources = st.sidebar.multiselect(
    "Sources", ["User PDF", "Database"], default=["User PDF", "Database"]
)
pipeline_sources = []
if "User PDF" in sources:
    pipeline_sources.append("user")
if "Database" in sources:
    pipeline_sources.append("db")

use_bm25 = st.sidebar.checkbox("Use BM25", value=False)
use_reranker = st.sidebar.checkbox("Use Reranker (MiniLM-6)", value=False)

# ------------------------
# HyDE Advanced Settings
# ------------------------
hyde_top_k = st.sidebar.slider("HyDE retrieval top_k", 1, 20, 3)
pseudo_max_tokens = st.sidebar.slider("Pseudo-answer max tokens", 10, 200, 50)

# ------------------------
# Retrieval Method Selection
# ------------------------
retrieval_method = st.sidebar.radio("Select Retrieval Method", ["RAG", "HyDE"])
compare_method = st.sidebar.checkbox("Compare RAG vs HyDE", value=False)

# ------------------------
# RAG Pipeline
# ------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(
        user_vectorstore=st.session_state.user_vectorstore,
        db_vectorstore=st.session_state.db_vectorstore,
        chunk_size=400,
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2" if use_reranker else None,
        use_bm25=use_bm25,
    )
else:
    st.session_state.rag.use_bm25 = use_bm25
    # Update reranker
    if use_reranker and not st.session_state.rag.reranker:
        from sentence_transformers import CrossEncoder
        st.session_state.rag.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
    elif not use_reranker:
        st.session_state.rag.reranker = None

# ------------------------
# GraphRAG Pipeline
# ------------------------
if "graph_rag" not in st.session_state:
    st.session_state.graph_rag = GraphRAGPipeline(st.session_state.rag)

st.sidebar.subheader("🧠 GraphRAG Options")
enable_graph_rag = st.sidebar.checkbox("Enable Knowledge Graph (GraphRAG)", value=False)

if enable_graph_rag:
    if st.sidebar.button("Build Knowledge Graph"):
        with st.spinner("🔄 Building Knowledge Graph..."):
            st.session_state.graph_rag.build_knowledge_graph()
        st.sidebar.success("✅ Knowledge Graph built!")

# ------------------------
# File Upload Section
# ------------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader(
    "📂 Upload PDFs or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="file_uploader_widget",
)

if uploaded_files:
    st.session_state.uploaded_files.extend(uploaded_files)
    for file in uploaded_files:
        process_file(
            file,
            st.session_state.user_vectorstore,
            rag_pipeline=st.session_state.rag,
            persist=True,
        )
    st.success("✅ Files uploaded and processed successfully!")

# ------------------------
# Document Management
# ------------------------
st.sidebar.subheader("🗑️ Document Management")

if st.sidebar.button("Clear Uploaded Documents"):
    st.session_state.user_vectorstore = VectorStore(
        persist_path="data/user_store", load=False
    )
    st.session_state.rag.user_vectorstore = st.session_state.user_vectorstore
    if os.path.exists("data/user_store"):
        shutil.rmtree("data/user_store")
    os.makedirs("data/user_store", exist_ok=True)
    st.session_state.uploaded_files = []
    st.success("✅ All uploaded documents cleared!")

# ------------------------
# Cache Management
# ------------------------
if st.sidebar.button("Clear Cache"):
    st.session_state.rag.clear_cache()
    st.sidebar.success("✅ Cache cleared!")

# ------------------------
# QA Section
# ------------------------
st.subheader("💬 Ask a Question")

user_input = st.text_input("Type your question here:")

if user_input:
    with st.spinner("🤖 Generating answer..."):
        if enable_graph_rag:
            # Use GraphRAG pipeline
            response = st.session_state.graph_rag.query(
                user_input, top_k=top_k, max_length=max_tokens
            )
            st.write("🤖 **GraphRAG Answer:**")
            st.write(response)
        elif compare_method:
            # Compare RAG vs HyDE
            rag_response = st.session_state.rag.answer(
                user_input,
                top_k=top_k,
                max_length=max_tokens,
                sources=pipeline_sources,
            )
            hyde_response = st.session_state.rag.hyde_answer(
                user_input,
                top_k=hyde_top_k,
                max_length=max_tokens,
                sources=pipeline_sources,
                pseudo_max_tokens=pseudo_max_tokens,
            )
            st.write("🤖 **RAG Answer:**")
            st.write(rag_response)
            st.write("🤖 **HyDE Answer:**")
            st.write(hyde_response)
        else:
            # Use selected method (RAG or HyDE)
            if retrieval_method == "RAG":
                response = st.session_state.rag.answer(
                    user_input,
                    top_k=top_k,
                    max_length=max_tokens,
                    sources=pipeline_sources,
                )
            else:  # HyDE
                response = st.session_state.rag.hyde_answer(
                    user_input,
                    top_k=hyde_top_k,
                    max_length=max_tokens,
                    sources=pipeline_sources,
                    pseudo_max_tokens=pseudo_max_tokens,
                )
            st.write("🤖 **Answer:**")
            st.write(response)
