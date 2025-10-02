import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

class VectorStore:
    def __init__(self, persist_path=None, load=False):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []  # her eleman {"text": ..., "meta": {...}}
        self.persist_path = persist_path
        if persist_path and load:
            self.load()

    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms

    def add_texts(self, texts, metadata_list=None):
        if not texts:
            return
        if metadata_list is None:
            metadata_list = [{} for _ in texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = self.normalize_embeddings(embeddings).astype(np.float32)
        self.index.add(embeddings)

        for t, m in zip(texts, metadata_list):
            self.texts.append({"text": t, "meta": m})

    def search(self, query, top_k=3):
        if not self.texts:
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = self.normalize_embeddings(query_embedding).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        if len(indices) == 0 or len(indices[0]) == 0:
            return []
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

    def save(self):
        if not self.persist_path:
            raise ValueError("persist_path is not set for VectorStore")
        os.makedirs(self.persist_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_path, "index.faiss"))
        with open(os.path.join(self.persist_path, "texts.json"), "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.persist_path:
            raise ValueError("persist_path is not set for VectorStore")
        index_path = os.path.join(self.persist_path, "index.faiss")
        texts_path = os.path.join(self.persist_path, "texts.json")
        if os.path.exists(index_path) and os.path.exists(texts_path):
            self.index = faiss.read_index(index_path)
            with open(texts_path, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
        else:
            print(f"[VectorStore] No persisted index found at {self.persist_path}. Starting fresh.")
