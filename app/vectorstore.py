import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self):
        # Multilanguage embeddings
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []

    def add_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        return results
