import re

def clean_text(text: str) -> str:
    return " ".join(text.split())

def chunk_text(text: str, chunk_size=300, overlap=50) -> list[str]:
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
