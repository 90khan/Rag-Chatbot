import spacy
import networkx as nx

class GraphBuilder:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except:
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        self.graph = nx.Graph()

    def extract_entities_relations(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        relations = []
        for token in doc:
            if token.dep_ in ("nsubj", "dobj", "pobj") and token.head.pos_ == "VERB":
                subj = token.text
                rel = token.head.lemma_
                obj = " ".join([child.text for child in token.head.children if child.dep_ == "dobj"])
                if subj and obj:
                    relations.append((subj, rel, obj))
        return entities, relations

    def build_graph(self, documents):
        for doc in documents:
            text = doc.get("text", "")
            entities, relations = self.extract_entities_relations(text)
            for ent in entities:
                self.graph.add_node(ent)
            for subj, rel, obj in relations:
                self.graph.add_edge(subj, obj, relation=rel)
        return self.graph

    def query_related_entities(self, entity, depth=1):
        if entity not in self.graph:
            return []
        neighbors = list(nx.single_source_shortest_path_length(self.graph, entity, cutoff=depth).keys())
        return neighbors
