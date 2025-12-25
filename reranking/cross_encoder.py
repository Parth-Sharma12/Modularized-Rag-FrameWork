from sentence_transformers import CrossEncoder
class Reranker:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query:str, documents:list[str],top_n=3):
        pairs = [(query,doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        ranked = sorted(
            zip(documents,scores), # This is "x"
            key=lambda x:x[1], # Sort on the basis of key which is score here
            reverse = True
        )
        return ranked[0:top_n]