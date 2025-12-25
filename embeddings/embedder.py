from sentence_transformers import SentenceTransformer
import numpy as np
class Embedder:
    def __init__(self,model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.astype(np.float32) #We convert to NumPy because embeddings are data artifacts, not model internals — and NumPy is the universal format for storage, search, and retrieval.