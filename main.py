import streamlit as st
from app.file_processor import process_file
from app.vectorstore import VectorStore
from app.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG QA Chatbot", page_icon="🤖")
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

# Chat input
user_input = st.text_input("Ask a question (TR/EN/DE):")
if user_input:
    response = st.session_state.rag.answer(user_input)
    st.write("🤖", response)
