import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension:int):
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []
        self.embeddings = None

    def add_vectors(self, texts:list[str],embeddings:np.ndarray):
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.embeddings = embeddings
    
    def search(self,query_embedding:np.ndarray,top_k=5):
        scores, indices = self.index.search(query_embedding,top_k)
        return {
        "texts": [self.texts[i] for i in indices[0]],
        "scores": scores[0],
        "indices": indices[0]
    }      

    

