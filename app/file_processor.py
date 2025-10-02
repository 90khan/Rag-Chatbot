import os, tempfile
from PyPDF2 import PdfReader
import docx
from app.utils import chunk_text, clean_text

def process_file(uploaded_file, vectorstore, rag_pipeline=None, persist=True):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    text = ""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif uploaded_file.name.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    else:
        os.remove(file_path)
        raise ValueError("Only PDF or DOCX supported")

    os.remove(file_path)
    text = clean_text(text)
    if not text.strip():
        return "⚠️ No text could be extracted"

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    metadata_list = [{"doc_name": uploaded_file.name} for _ in chunks]
    vectorstore.add_texts(chunks, metadata_list=metadata_list)

    if persist and hasattr(vectorstore, "save"):
        vectorstore.save()
    if rag_pipeline is not None:
        rag_pipeline.refresh_bm25()

    return f"✅ {uploaded_file.name} processed, {len(chunks)} chunks added"
