from ingestion.loader import load_documents
from ingestion.chunker import chunk_doc
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from pipeline import ModularRAGPipeline

# Build vector store first
docs = load_documents("data/documents.txt")
chunks = chunk_doc(docs)

embedder = Embedder()
embeddings = embedder.embed(chunks)
print("TYPE of embeddings %s\n",type(embeddings))
vector_store = VectorStore(embeddings.shape[1])
vector_store.add_vectors(chunks,embeddings)

# Run Modular RAG
rag = ModularRAGPipeline(vector_store)

query = "Why was my insurance claim rejected?"
result = rag.run(query)

print("\n=== Retrieved (Vector DB) ===")
for d, s in result["retrieved"]:
    print(f"{s:.4f} | {d}")

print("\n=== Re-ranked ===")
for d, s in result["reranked"]:
    print(f"{s:.4f} | {d}")

print("\n=== Final Answer ===")
print(result["answer"])
