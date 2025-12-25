# 🧩 Modular RAG Visualizer  
### Understanding why Vector Databases alone are not enough

🔗 **Live Demo:** [[Add Demo Link Here](https://drive.google.com/drive/folders/1AeOBonHmU9iy4q4-5UdeerngSCi0Nx3j?usp=drive_link)]  


---

## 📌 Overview

Large Language Models (LLMs) often **hallucinate** — not because they lack intelligence, but because they are forced to answer questions **without sufficient or well-structured context**.

This project is a **fully visual, modular Retrieval-Augmented Generation (RAG) demo** that shows — step by step — **where hallucinations originate and how modular RAG reduces them**.

Instead of treating RAG as a black box, this demo focuses on:
- Explainability
- Visualization
- Component-level reasoning

---

## 🧠 What This Demo Demonstrates

### 🔵 1. Vector Database Retrieval
- Fast semantic similarity search
- High recall but noisy results
- Visualized using similarity score charts

### 🧠 2. Vector Space Visualization (PCA)
- PCA projection of embeddings
- Shows:
  - All document chunks
  - Retrieved chunks
  - Query embedding
- Explains why **similarity ≠ relevance**

### 🟢 3. Cross-Encoder Re-ranking
- High-precision semantic scoring
- Improves relevance
- Exposes limitations for **multi-part questions**

### 🟣 4. Context Construction
- Carefully selected and ordered chunks
- Demonstrates that **context quality is the real bottleneck**

### 🤖 5. Answer Generation
- Final grounded response using Gemini
- Reduced hallucinations due to better context

### 🔽 Noise Reduction Funnel
- Visual funnel showing chunk reduction across RAG stages

---

## ✨ Key Learnings

- Vector databases are necessary, but **not sufficient**
- Cross-encoders improve relevance, but don’t fully understand intent
- Hallucinations often result from:
  - Missing context
  - Poor chunk selection
  - Weak context construction
- **RAG is a system-design problem, not a tooling problem**

---

## 🏗️ Modular Architecture

