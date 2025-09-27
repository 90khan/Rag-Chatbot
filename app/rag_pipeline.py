import torch
from transformers import pipeline
import shelve
import hashlib

class ShelveCache:
    def __init__(self, filename="cache.db"):
        self.filename = filename

    def _hash_key(self, key):
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key):
        hashed_key = self._hash_key(key)
        with shelve.open(self.filename) as db:
            return db.get(hashed_key)

    def set(self, key, value):
        hashed_key = self._hash_key(key)
        with shelve.open(self.filename) as db:
            db[hashed_key] = value

class RAGPipeline:
    def __init__(self, vectorstore, cache_enabled=True):
        self.vectorstore = vectorstore
        self.cache = ShelveCache()
        self.cache_enabled = cache_enabled

        # MPS (Apple Silicon) check
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Multilanguage model answering
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if self.device == "mps" else -1
        )

    def build_prompt(self, context, query):
        """Prompt sabit olarak İngilizce"""
        return f"""Answer the question based on the following context. 
Provide a concise, accurate answer. 
Do not repeat the question. Do not just repeat words from the context; paraphrase and summarize when possible. 
If the answer is not in the context, say 'Information not found in the provided documents.'

Context:
{context}

Question:
{query}

Answer:"""

    def answer(self, query, top_k=3, max_length=200, temperature=0.7, do_sample=True):
        if self.cache_enabled:
            cached = self.cache.get(query)
            if cached:
                return cached

        # Retrieve top_k documents
        retrieved_docs = self.vectorstore.search(query, top_k=top_k)
        context = "\n".join(retrieved_docs)

        # Build prompt
        prompt = self.build_prompt(context, query)

        # Generate
        output = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample
        )
        answer_text = output[0]["generated_text"]

        # Cache
        if self.cache_enabled:
            self.cache.set(query, answer_text)

        return answer_text
