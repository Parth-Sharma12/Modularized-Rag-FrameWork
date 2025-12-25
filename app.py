import streamlit as st
from pipeline import ModularRAGPipeline
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from ingestion.loader import load_documents
from ingestion.chunker import chunk_doc

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# ------------------ Global Style ------------------
plt.style.use("seaborn-v0_8-whitegrid")

# ------------------ Visualization Helpers ------------------
def plot_scores(title, labels, scores):
    df = pd.DataFrame({
        "Chunk": labels,
        "Score": scores
    })

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(df["Chunk"], df["Score"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Score")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)


def plot_vector_space_conceptual(embeddings, top_k):
    """
    Conceptual PCA visualization.
    Does NOT rely on FAISS indices.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(5, 4))

    # All chunks
    ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        alpha=0.25,
        label="All Chunks"
    )

    # Conceptual highlight: first top_k vectors
    ax.scatter(
        reduced[:top_k, 0],
        reduced[:top_k, 1],
        marker="X",
        s=80,
        label="Top-K Retrieved (Conceptual)"
    )

    ax.set_title("Vector Space (PCA Projection)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)

    st.caption(
        "⚠️ Conceptual visualization: illustrates retrieval geometry without FAISS indices."
    )

    plt.tight_layout()
    st.pyplot(fig)

def plot_vector_space_true(embeddings, query_embedding, retrieved_indices):
    """
    Proper PCA visualization of vector space:
    - All embeddings as background
    - Retrieved embeddings highlighted
    - Query embedding in red
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(
        np.vstack([embeddings, query_embedding])
    )

    doc_points = reduced[:-1]
    query_point = reduced[-1]

    fig, ax = plt.subplots(figsize=(5, 4))

    # All chunks
    ax.scatter(
        doc_points[:, 0],
        doc_points[:, 1],
        alpha=0.25,
        label="All Chunks"
    )

    # Highlight retrieved vectors
    ax.scatter(
        doc_points[retrieved_indices, 0],
        doc_points[retrieved_indices, 1],
        s=80,
        marker="X",
        label="Retrieved Chunks"
    )

    # Query vector
    ax.scatter(
        query_point[0],
        query_point[1],
        s=120,
        color="red",
        label="Query"
    )

    ax.set_title("Vector Space (PCA Projection)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)



# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Modular RAG Visualizer",
    layout="wide"
)

st.title("🧩 Modular RAG Visualizer")
st.caption("Seeing why Vector Databases alone are not enough")

# ------------------ Build Pipeline ------------------
@st.cache_resource
def build_pipeline():
    docs = load_documents("data/documents.txt")
    chunks = chunk_doc(docs)

    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    vs = VectorStore(embeddings.shape[1])
    vs.add_vectors(chunks, embeddings)

    pipeline = ModularRAGPipeline(vs)

    # store embeddings only for visualization (UI purpose)
    pipeline._all_embeddings = embeddings
    return pipeline

rag = build_pipeline()

# ------------------ Query Input ------------------
query = st.text_input(
    "🔍 Ask a question",
    placeholder="Why was my insurance claim rejected?"
)

run = st.button("Run Modular RAG")

# ------------------ Run Pipeline ------------------
if run and query:

    result = rag.run(query)

    # =========================
    # TOP: Retrieval + Geometry
    # =========================
    st.subheader("🔵 Retrieval & Vector Geometry")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("#### 🔎 Vector DB Scores")

        vec_labels = [f"C{i+1}" for i in range(len(result["retrieved"]))]
        vec_scores = [score for _, score in result["retrieved"]]

        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.barh(vec_labels, vec_scores)
        ax.invert_yaxis()
        ax.set_xlabel("Similarity")
        ax.set_title("Vector DB Retrieval")
        st.pyplot(fig)

    with col2:
        st.markdown("#### 🧠 Vector Space")

        query_emb = Embedder().embed([query])[0]
        retrieved_indices = [
            rag.vector_store.texts.index(doc)
            for doc, _ in result["retrieved"]
        ]

        plot_vector_space_true(
            rag._all_embeddings,
            query_emb,
            retrieved_indices
        )

    # =========================
    # RETRIEVED TEXTS
    # =========================
    with st.expander("📄 Retrieved Chunks (Noisy but Fast)", expanded=False):
        for doc, score in result["retrieved"]:
            st.info(f"**Score:** {score:.4f}\n\n{doc}")

    st.divider()

    # =========================
    # RE-RANKING SECTION
    # =========================
    st.subheader("🟢 Cross-Encoder Re-ranking")

    col3, col4 = st.columns([1.1, 1])

    with col3:
        st.markdown("#### 🎯 Re-ranked Scores")

        rerank_labels = [f"R{i+1}" for i in range(len(result["reranked"]))]
        rerank_scores = [score for _, score in result["reranked"]]

        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.barh(rerank_labels, rerank_scores)
        ax.invert_yaxis()
        ax.set_xlabel("Relevance")
        ax.set_title("Cross-Encoder Ranking")
        st.pyplot(fig)

    with col4:
        with st.expander("📄 Re-ranked Chunks (High Precision)", expanded=True):
            for doc, score in result["reranked"]:
                st.success(f"**Relevance:** {score:.4f}\n\n{doc}")

    st.divider()

    # =========================
    # CONTEXT + ANSWER
    # =========================
    st.subheader("🧠 Reasoning → Answer")

    col5, col6 = st.columns([1, 1])

    with col5:
        st.markdown("#### 🟣 Final Context")
        st.code(result["context"], language="markdown")

    with col6:
        st.markdown("#### 🤖 Gemini Answer")
        st.markdown(result["answer"])

    st.divider()

    # =========================
    # NOISE REDUCTION FUNNEL
    # =========================
    st.subheader("🔽 Noise Reduction Funnel")

    col7, col8 = st.columns([1, 2])

    with col7:
        funnel_df = pd.DataFrame({
            "Stage": ["Retrieved", "Re-ranked", "Final"],
            "Chunks": [
                len(result["retrieved"]),
                len(result["reranked"]),
                1
            ]
        })

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(funnel_df["Stage"], funnel_df["Chunks"], marker="o")
        ax.set_ylabel("Chunks")
        ax.set_title("Noise Reduction")
        st.pyplot(fig)

    with col8:
        st.info(
            "🔍 **Insight**\n\n"
            "Vector DB retrieves broadly.\n"
            "Cross-encoder filters aggressively.\n"
            "LLM sees only what matters."
        )

else:
    st.info("Enter a query and click **Run Modular RAG**")
