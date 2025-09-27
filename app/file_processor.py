import os
import tempfile
from PyPDF2 import PdfReader
import docx
from app.utils import chunk_text, clean_text, detect_language

def process_file(uploaded_file, vectorstore):
    # Dosyayı geçici olarak kaydet
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
        document = docx.Document(file_path)
        for para in document.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    else:
        os.remove(file_path)
        raise ValueError("❌ Sadece PDF veya DOCX dosyaları destekleniyor.")

    # Geçici dosyayı sil
    os.remove(file_path)

    # Metni temizle
    text = clean_text(text)
    if not text.strip():
        return "⚠️ Bu dosyadan metin çıkarılamadı."

    # Dil tespiti yap
    lang = detect_language(text)

    # Chunk & store in vector DB
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    vectorstore.add_texts(chunks)

    return f"✅ {uploaded_file.name} ({lang}) dosyasından {len(chunks)} parça eklendi."

