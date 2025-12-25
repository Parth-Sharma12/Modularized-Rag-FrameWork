from embeddings.embedder import Embedder
from ingestion.chunker import chunk_doc
from ingestion.loader import load_documents
from retrieval.vector_store import VectorStore
from reranking.cross_encoder import Reranker

def main():
    docs = load_documents("data\documents.txt")
    chunks = chunk_doc(docs)
    
    print("Chunks created %s\n",chunks)

#-------

    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    print("Embeddings created %s\n",embeddings)

#-------
 
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_vectors(chunks,embeddings)

#-------

    input_query = "Why was my insurance claim rejected?"
    embedded_input = embedder.embed([input_query])
    results = vector_store.search(embedded_input)

#-------
    reranker = Reranker()
    ranked = reranker.rerank(input_query,[text for text,score in results])
#-------

    print("\n=== Vector DB Results ===")
    for text, score in ranked:
        print(f"Score: {score:.4f} | Text: {text}")

if __name__ == "__main__":
    main()