import torch
from transformers import pipeline

class RAGPipeline:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

        # MPS (Apple Silicon) check
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Multilanguage model answering (Flan-T5-base)
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if self.device == "mps" else -1
        )

    def answer(self, query):
        # Retrieve
        retrieved_docs = self.vectorstore.search(query)
        context = "\n".join(retrieved_docs)

        # Augmented prompt
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Generate
        output = self.generator(prompt, max_length=200, do_sample=True)
        return output[0]["generated_text"]
