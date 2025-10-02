import os
import shutil
import streamlit as st
from app.file_processor import process_file
from app.vectorstore import VectorStore
from app.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG QA Chatbot", page_icon="🤖")

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
            st.success("Logged in!")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

st.title("📚 RAG QA Chatbot")

# --- VectorStores ---
if "user_vectorstore" not in st.session_state:
    st.session_state.user_vectorstore = VectorStore(
        persist_path="data/user_store", load=True
    )
if "db_vectorstore" not in st.session_state:
    st.session_state.db_vectorstore = VectorStore(
        persist_path="data/db_store", load=True
    )

# --- Sidebar Settings ---
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

# --- RAG Pipeline init/update ---
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
    # Reranker güncelle
    if use_reranker and not st.session_state.rag.reranker:
        from sentence_transformers import CrossEncoder
        st.session_state.rag.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )
    elif not use_reranker:
        st.session_state.rag.reranker = None

# --- File uploader session_state ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader(
    "Upload PDFs or DOCs",
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
    st.success("✅ Files uploaded and processed!")

# --- Document Management ---
st.sidebar.subheader("🗑️ Document Management")
if st.sidebar.button("Clear Uploaded Documents"):
    st.session_state.user_vectorstore = VectorStore(
        persist_path="data/user_store", load=False
    )
    st.session_state.rag.user_vectorstore = st.session_state.user_vectorstore
    if os.path.exists("data/user_store"):
        shutil.rmtree("data/user_store")
    os.makedirs("data/user_store", exist_ok=True)
    # File uploader sıfırlama
    st.session_state.uploaded_files = []
    st.success("✅ All uploaded documents cleared!")

# --- Cache Management ---
if st.sidebar.button("Clear Cache"):
    st.session_state.rag.clear_cache()
    st.sidebar.success("✅ Cache cleared!")

# --- QA Input ---
user_input = st.text_input("Ask a question:")
if user_input:
    response = st.session_state.rag.answer(
        user_input,
        top_k=top_k,
        max_length=max_tokens,
        sources=pipeline_sources,
    )
    st.write("🤖", response)
