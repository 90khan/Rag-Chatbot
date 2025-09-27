import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self):
        # Multilanguage embeddings
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity için IP kullanıyoruz
        self.texts = []

    def normalize_embeddings(self, embeddings):
        # L2 normalize her embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def add_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = self.normalize_embeddings(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self.normalize_embeddings(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        return results

