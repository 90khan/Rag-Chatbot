import os
import streamlit as st
from app.file_processor import process_file
from app.vectorstore import VectorStore
from app.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG QA Chatbot", page_icon="🤖")

# --- Simple Auth ---
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
# -------------------

st.title("📚 Multilanguage RAG QA Chatbot")

# Initialize vectorstore + pipeline
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()
    st.session_state.rag = RAGPipeline(st.session_state.vectorstore)

# File upload
uploaded_files = st.file_uploader("Upload PDFs or DOCs", type=["pdf", "docx"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        process_file(file, st.session_state.vectorstore)
    st.success("✅ Files uploaded and processed!")

# --- Sidebar: Generation & Retrieval Settings ---
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Number of documents to retrieve (top_k)", min_value=1, max_value=50, value=10)
max_length = st.sidebar.slider("Max tokens for answer", min_value=50, max_value=500, value=200, step=10)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
do_sample = st.sidebar.checkbox("Enable sampling (do_sample)", value=True)

# Chat input
user_input = st.text_input("Ask a question (TR/EN/DE):")
if user_input:
    response = st.session_state.rag.answer(
        user_input,
        top_k=top_k,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample
    )
    st.write("🤖", response)
