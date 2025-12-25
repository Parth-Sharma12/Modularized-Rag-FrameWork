# 🧩 Modular RAG Visualizer

 **Live Demo:** [Click here]((https://drive.google.com/drive/folders/1AeOBonHmU9iy4q4-5UdeerngSCi0Nx3j?usp=drive_link))  


---

## Overview

Large Language Models (LLMs) can hallucinate when answering questions outside their training data or without sufficient context. This project demonstrates **Modular Retrieval-Augmented Generation (RAG)** to reduce hallucinations by grounding LLM responses in relevant external knowledge.  

The **interactive demo** allows you to visualize each step of the RAG process and understand how retrieved chunks are scored, re-ranked, and used to generate final answers.

---

## Features

- **Vector DB Retrieval:** Quickly retrieve relevant chunks using a FAISS vector store.
- **Vector Space Visualization:** Visualize embeddings in 2D using PCA, with retrieved chunks highlighted.
- **Cross-Encoder Re-ranking:** Rank retrieved chunks for higher relevance.
- **Context Builder:** Assemble top chunks into context for the language model.
- **Answer Generation:** Produce precise answers using the context.
- **Noise Reduction Funnel:** See how irrelevant chunks are progressively filtered.

---

## Tech Stack

- **Python** for backend logic.
- **Streamlit** for interactive UI.
- **Matplotlib** for charts and visualization.
- **Pandas & Numpy** for data manipulation.
- **Sentence Transformers** for embeddings.
- **FAISS** for vector store and similarity search.
- **Cross-Encoder** for reranking.
- **PCA** for 2D vector space visualization.

---

## Installation

```bash
# Clone the repository
git clone ADD_REPO_LINK_HERE
cd modular-rag-visualizer

# Create environment and install dependencies
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt

# Create a .env file and add your Gemini API key
echo "GEMINI_API_KEY='your_api_key_here'" > .env

streamlit run app.py
Enter your question in the input box.

The system retrieves relevant chunks from the document.

Visualize retrieved scores and vector space.

Cross-encoder re-ranks chunks for precision.

Context is built and the final answer is generated.

Observe the noise reduction funnel for clarity.


modular-rag-visualizer/
│
├── app.py                 # Streamlit app
├── pipeline.py            # Modular RAG pipeline
├── embeddings/
│   └── embedder.py        # SentenceTransformer embeddings
├── retrieval/
│   └── vector_store.py    # FAISS vector storage
├── reranking/
│   └── cross_encoder.py   # Cross-encoder reranker
├── context/
│   └── builder.py         # Context builder for LLM
├── generation/
│   └── answer_generator.py # Gemini / LLM answer generator
├── ingestion/
│   ├── loader.py          # Document loader
│   └── chunker.py         # Document chunker
├── data/
│   └── documents.txt      # Demo text data
├── requirements.txt
└── .env                   # API key (not tracked in Git)
```
