# 📚 Multilingual RAG QA Chatbot

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot with a Streamlit UI.  
Users can upload PDF or DOCX files; the app extracts text, stores embeddings in FAISS,  
and answers questions using a generation model augmented with retrieved context.

---

## ✨ Features

* Upload multiple PDFs or DOCX files
* FAISS-backed vector store with multilingual sentence-transformers embeddings
* Retrieval-Augmented Generation (RAG) using Hugging Face transformers
* Simple caching of previously asked questions (optional)
* Dockerized for easy deployment
* Multilingual support: Turkish, English, German

---

## 🚀 Quick start (local)

```bash
# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run Streamlit app
streamlit run main.py

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run with Docker

```bash
# build Docker image
docker build -t rag-chatbot .

# run Docker container
docker run -p 8501:8501 rag-chatbot

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```
app/
 ├── file_processor.py   # file parsing & chunking
 ├── vectorstore.py      # FAISS wrapper
 ├── rag_pipeline.py     # retrieval + generation pipeline + cache
 ├── utils.py            # helpers
main.py                  # Streamlit UI entry point
Dockerfile
requirements.txt
```


