📚 RAG + GraphRAG QA Chatbot

This project is a Retrieval-Augmented Generation (RAG) based Question-Answering Chatbot with a Streamlit web interface, now enhanced with GraphRAG — a knowledge graph feature for context-aware reasoning.
Users can upload PDF or DOCX files; the app extracts text, stores embeddings in FAISS, builds a knowledge graph, and generates answers using a Hugging Face model augmented with retrieved context.

✨ Features

📂 File Upload: Upload multiple PDF/DOCX documents

🧩 Text Processing: Cleaning + chunking

🔎 Search Engine: FAISS + optional BM25 hybrid retrieval

🤖 RAG Pipeline: Hugging Face Flan-T5 for context-aware answers

🧠 GraphRAG Pipeline: Knowledge graph built from uploaded documents to enhance reasoning and retrieval

🔁 Cache: Disk-based caching (shelve) for repeated queries

🏷️ Metadata: Responses include document names for traceability

🔑 Authentication: Simple username/password login (environment variables)

🐳 Dockerized: Easy to build and deploy

🚀 Quick Start (Local)
1️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run main.py

4️⃣ Default login credentials
Username: admin
Password: password


Change credentials by setting environment variables:

export APP_USER=your_username
export APP_PASS=your_password


Open your browser at: http://localhost:8501

🐳 Run with Docker
Build and start containers
docker-compose up -d --build

Stop containers
docker-compose down


Open your browser at: http://localhost:8501

📂 Project Structure
app/
├── file_processor.py      # File parsing & chunking
├── vectorstore.py         # FAISS wrapper
├── rag_pipeline.py        # Retrieval + generation pipeline + cache
├── graph_builder.py       # Extract entities & relations from text
├── graph_pipeline.py      # GraphRAG: integrate knowledge graph with RAG pipeline
├── utils.py               # Helper functions
main.py                    # Streamlit UI entry point
Dockerfile                 # Docker image definition
docker-compose.yml         # Multi-container/project orchestration
.env                       # Environment variables (user credentials)
requirements.txt           # Python dependencies

⚙️ Environment Variables
Variable	Description	Default
APP_USER	Username for login	admin
APP_PASS	Password for login	password
OPENAI_API_KEY	(Optional) Key for enhanced model usage	-
🧠 Tech Stack

Python 3.10+

Streamlit

FAISS

Hugging Face Transformers

spaCy (for entity extraction)

NetworkX (for graph building and querying)

Docker / Docker Compose

🧪 Example Usage

Upload one or more .pdf or .docx files.

Ask a question about the content (e.g., "Summarize the introduction section").

The chatbot retrieves relevant context:

RAG: via vector embeddings (FAISS + optional BM25)

GraphRAG: optionally uses the knowledge graph to find related entities and improve answers

The app generates context-aware answers using Hugging Face Flan-T5.

🧠 GraphRAG Usage

Enable the Knowledge Graph in the sidebar.

Click Build Knowledge Graph to extract entities and relations from uploaded documents.

GraphRAG allows queries to consider related entities, not just direct vector search.

This improves retrieval for complex questions and reasoning over multiple documents.

🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request for new features or bug fixes.

📜 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

💡 Acknowledgements

Special thanks to:

Hugging Face

Streamlit

FAISS

spaCy

NetworkX
