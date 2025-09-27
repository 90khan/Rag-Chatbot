import re
from langdetect import detect, DetectorFactory

# Dil tespitinde deterministik sonuç almak için
DetectorFactory.seed = 0

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """ Metni anlamlı parçalara böler """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def clean_text(text: str) -> str:
    """ Basit metin temizleme """
    return " ".join(text.split())

def detect_language(text: str) -> str:
    """ Metnin dilini tespit eder """
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"
