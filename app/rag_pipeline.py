import torch
from transformers import pipeline, AutoTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import shelve, hashlib, os

class ShelveCache:
    def __init__(self, filename="cache.db"):
        self.filename = filename

    def _hash_key(self, key):
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, key_dict):
        raw_key = str(key_dict)
        hashed_key = self._hash_key(raw_key)
        with shelve.open(self.filename) as db:
            return db.get(hashed_key)

    def set(self, key_dict, value):
        raw_key = str(key_dict)
        hashed_key = self._hash_key(raw_key)
        with shelve.open(self.filename) as db:
            db[hashed_key] = value

class RAGPipeline:
    def __init__(self, user_vectorstore, db_vectorstore=None, cache_enabled=True, use_bm25=False,
                 reranker_model=None, chunk_size=400, max_model_tokens=512):
        self.user_vectorstore = user_vectorstore
        self.db_vectorstore = db_vectorstore
        self.cache = ShelveCache()
        self.cache_enabled = cache_enabled
        self.use_bm25 = use_bm25
        self.chunk_size = chunk_size
        self.max_model_tokens = max_model_tokens

        self.bm25_user = None
        self.bm25_db = None
        if use_bm25:
            self.refresh_bm25()

        self.reranker = CrossEncoder(reranker_model, device="cpu") if reranker_model else None

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        device_id = 0 if self.device == "mps" else -1
        self.generator = pipeline("text2text-generation",
                                  model="google/flan-t5-base",
                                  device=device_id)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def refresh_bm25(self):
        if self.user_vectorstore and self.user_vectorstore.texts:
            self.bm25_user = BM25Okapi([doc["text"].split() for doc in self.user_vectorstore.texts])
        if self.db_vectorstore and self.db_vectorstore.texts:
            self.bm25_db = BM25Okapi([doc["text"].split() for doc in self.db_vectorstore.texts])

    def build_prompt(self, context, query):
        return f"""Answer the question using only the following context.
If the answer is not in the context, say "No such as information".

Context:
{context}

Question: {query}
Answer:"""

    def retrieve(self, query, top_k, sources):
        docs = []
        if "user" in sources and self.user_vectorstore:
            if self.use_bm25 and self.bm25_user:
                scores = self.bm25_user.get_scores(query.split())
                indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                docs += [self.user_vectorstore.texts[i] for i in indices]
            else:
                docs += self.user_vectorstore.search(query, top_k)
        if "db" in sources and self.db_vectorstore:
            if self.use_bm25 and self.bm25_db:
                scores = self.bm25_db.get_scores(query.split())
                indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                docs += [self.db_vectorstore.texts[i] for i in indices]
            else:
                docs += self.db_vectorstore.search(query, top_k)
        return docs

    def rerank_docs(self, query, docs, top_k):
        if not self.reranker or not docs:
            return docs[:top_k]
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        return sorted_docs[:top_k]

    def safe_chunk(self, text):
        tokens = self.tokenizer.encode(text, truncation=False)
        chunks = []
        for i in range(0, len(tokens), self.max_model_tokens):
            chunk_text = self.tokenizer.decode(tokens[i:i+self.max_model_tokens], skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    def answer(self, query, top_k=3, max_length=200, temperature=0.0, do_sample=False,
               sources=["user","db"], concat_chunks=True):
        doc_ids = [d["meta"].get("doc_name","") for d in (self.user_vectorstore.texts if self.user_vectorstore else [])]
        doc_hash = hashlib.sha256("".join(doc_ids).encode()).hexdigest()

        cache_key = {"query": query, "sources": sources, "top_k": top_k, "max_length": max_length,
                     "temperature": temperature, "do_sample": do_sample, "doc_hash": doc_hash}

        if self.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        retrieved_docs = self.retrieve(query, top_k, sources)
        retrieved_docs = self.rerank_docs(query, retrieved_docs, max(1, top_k))

        if not retrieved_docs:
            return "No such as information"

        all_chunks = []
        for doc in retrieved_docs:
            all_chunks.extend(self.safe_chunk(doc["text"]))

        if concat_chunks:
            context = "\n\n".join(
                [f"[{doc['meta'].get('doc_name','unknown')}] {doc['text']}" for doc in retrieved_docs[:3]]
            )
            prompt = self.build_prompt(context, query)
            output = self.generator(prompt, max_new_tokens=max_length)
            combined = output[0].get("generated_text","").strip()
        else:
            combined = ""
            for doc in retrieved_docs[:3]:
                context = f"[{doc['meta'].get('doc_name','unknown')}] {doc['text']}"
                prompt = self.build_prompt(context, query)
                output = self.generator(prompt, max_new_tokens=max_length)
                combined += output[0].get("generated_text","").strip() + " "

        if self.cache_enabled:
            self.cache.set(cache_key, combined)

        return combined

    def hyde_answer(self, query, top_k=3, max_length=200, temperature=0.0, do_sample=False,
                    sources=["user","db"], pseudo_max_tokens=50):
        """
        HyDE approach:
        1. Generate hypothetical answer
        2. Embed hypothetical answer to retrieve relevant documents
        3. Generate final answer using retrieved docs

        Parameters:
            pseudo_max_tokens: Max tokens to generate the pseudo-answer
            top_k: Number of documents to retrieve using pseudo-answer
        """
        # Step 1: Generate hypothetical answer
        hypo_prompt = f"Generate a concise answer for the question, without external context:\nQuestion: {query}\nAnswer:"
        hypo_output = self.generator(hypo_prompt, max_new_tokens=pseudo_max_tokens)
        pseudo_answer = hypo_output[0].get("generated_text","").strip()

        # Step 2: Retrieve documents using pseudo-answer
        retrieved_docs = self.retrieve(pseudo_answer, top_k, sources)
        retrieved_docs = self.rerank_docs(pseudo_answer, retrieved_docs, max(1, top_k))

        if not retrieved_docs:
            return "No such as information"

        # Step 3: Generate final answer using retrieved docs
        context = "\n\n".join(
            [f"[{doc['meta'].get('doc_name','unknown')}] {doc['text']}" for doc in retrieved_docs[:3]]
        )
        final_prompt = self.build_prompt(context, query)
        output = self.generator(final_prompt, max_new_tokens=max_length)
        final_answer = output[0].get("generated_text","").strip()

        # Cache key
        doc_ids = [d["meta"].get("doc_name","") for d in (self.user_vectorstore.texts if self.user_vectorstore else [])]
        doc_hash = hashlib.sha256("".join(doc_ids).encode()).hexdigest()
        cache_key = {"query": query, "method": "hyde", "sources": sources, "top_k": top_k,
                     "max_length": max_length, "pseudo_max_tokens": pseudo_max_tokens,
                     "temperature": temperature, "do_sample": do_sample,
                     "doc_hash": doc_hash}

        if self.cache_enabled:
            self.cache.set(cache_key, final_answer)

        return final_answer

    def clear_cache(self):
        if os.path.exists(self.cache.filename):
            try:
                os.remove(self.cache.filename)
                print(f"[Cache] {self.cache.filename} deleted.")
            except Exception as e:
                print(f"[Cache] Could not clear cache: {e}")
