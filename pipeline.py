from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from reranking.cross_encoder import Reranker
from context.builder import build_context
from generation.answer_generator import AnswerGenerator

class ModularRAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.embedder = Embedder()
        self.vector_store = vector_store
        self.reranker = Reranker()
        self.generator = AnswerGenerator()

    def run(self, query: str):
        # 1. Embed query
        query_embedding = self.embedder.embed([query])

        # 2. Retrieve candidates
        search_result = self.vector_store.search(query_embedding)

        retrieved_texts = search_result["texts"]
        retrieved_scores = search_result["scores"]
        retrieved_indices = search_result["indices"]  # for visualization

        # 3. Re-rank top-N results
        reranked = self.reranker.rerank(query, retrieved_texts, top_n=3)

        # 4. Build context from reranked
        context = build_context(reranked)

        # 5. Generate answer
        answer = self.generator.generate(query, context)

        # 6. Return all for UI
        return {
            "retrieved": list(zip(retrieved_texts, retrieved_scores)),  # for UI loop
            "reranked": reranked,         # use actual reranked variable
            "context": context,           # use actual context variable
            "answer": answer,
            "retrieved_indices": retrieved_indices  # for vector-space plot
        }
