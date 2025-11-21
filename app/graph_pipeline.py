from app.graph_builder import GraphBuilder

class GraphRAGPipeline:
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.graph_builder = GraphBuilder()
        self.graph = None

    def build_knowledge_graph(self):
        print("[GraphRAG] Building knowledge graph...")
        all_docs = []
        if self.rag_pipeline.user_vectorstore:
            all_docs.extend(self.rag_pipeline.user_vectorstore.texts)
        if self.rag_pipeline.db_vectorstore:
            all_docs.extend(self.rag_pipeline.db_vectorstore.texts)
        self.graph = self.graph_builder.build_graph(all_docs)
        print(f"[GraphRAG] Graph built with {len(self.graph.nodes())} nodes.")

    def query(self, query, top_k=3, max_length=200):
        if self.graph is None:
            return "Graph not built yet. Please build it first."
        # Sorgudan entity’leri çıkar
        entities, _ = self.graph_builder.extract_entities_relations(query)
        if not entities:
            return self.rag_pipeline.answer(query, top_k=top_k, max_length=max_length)
        related_nodes = set()
        for ent in entities:
            related_nodes.update(self.graph_builder.query_related_entities(ent, depth=2))
        # Graph node’larına göre belge bul
        related_texts = []
        for doc in self.rag_pipeline.user_vectorstore.texts:
            if any(node in doc["text"] for node in related_nodes):
                related_texts.append(doc)
        if not related_texts:
            return self.rag_pipeline.answer(query, top_k=top_k, max_length=max_length)
        # RAG pipeline üzerinden yanıt oluştur
        context = "\n\n".join([d["text"] for d in related_texts[:3]])
        prompt = self.rag_pipeline.build_prompt(context, query)
        output = self.rag_pipeline.generator(prompt, max_new_tokens=max_length)
        return output[0]["generated_text"]
