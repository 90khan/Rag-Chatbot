# 📚 RAG QA Chatbot

This project is a **Retrieval-Augmented Generation (RAG)** based **Question-Answering Chatbot**  
with a Streamlit web interface.  
Users can upload **PDF or DOCX** files; the app extracts text, stores embeddings in **FAISS**,  
and generates answers using a Hugging Face model augmented with retrieved context.  

---

## ✨ Features

* 📂 **File upload:** Upload multiple PDF/DOCX documents  
* 🧩 **Text processing:** Cleaning + chunking  
* 🔎 **Search engine:** FAISS + optional BM25 hybrid retrieval  
* 🤖 **RAG pipeline:** Hugging Face Flan-T5 for context-aware answers  
* 🔁 **Cache:** Disk-based caching (shelve) for repeated queries  
* 🏷️ **Metadata:** Responses include document names for traceability  
* 🔑 **Authentication:** Simple username/password login (environment variables)  
* 🐳 **Dockerized:** Easy to build and deploy

---

## 🚀 Quick Start (Local)

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py

# Default login credentials
Username: admin
Password: password

(Change by setting environment variables APP_USER and APP_PASS)

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run with Docker

# docker-compose run 
docker-compose up -d --build

# docker-compose down
docker-compose down

Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📂 Project Structure

```text
app/
├── file_processor.py # File parsing & chunking
├── vectorstore.py # FAISS wrapper
├── rag_pipeline.py # Retrieval + generation pipeline + cache
├── utils.py # Helpers
main.py # Streamlit UI entry point
Dockerfile # Docker image definition
docker-compose.yml # Multi-container/project orchestration
.env # Environment variables (user credentials)
requirements.txt
```


