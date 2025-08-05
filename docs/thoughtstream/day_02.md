# 🧠 ThoughtStream — Day 02

_Date: 2025-08-06_

## ✅ Focus Areas Today:
- Embedding models & tokenization
- Vector indexing & retrieval algorithms
- FAISS & ChromaDB deep dive
- Dual Vector DB design (semantic vs precise)
- Chunking refinements
- Repo integration & smart ingestion

---

## 🔍 Key Learnings & Insights

### 🧬 Embeddings & Tokenization

- Understood that **text is first tokenized** using a tokenizer and then **converted into dense vector embeddings**.
- Explored how **embedding models** like `all-MiniLM-L6-v2` (from Hugging Face) are ideal for compact, semantic representations of text (suitable for retrieval).
- Realized the distinction between:
  - **Encoders**: Convert text → vector (used in RAG)
  - **Decoders**: Generate text from prompts (used in LLMs)
- Had a core revelation: **RAG = Encoder (for context) + Decoder (for response)**.

---

## 🗂️ Vector Databases — Architecting Dual Strategy

### 1️⃣ ChromaDB (DB_SEMANTIC)
- Uses **HNSW (Hierarchical Navigable Small World Graph)** with **Cosine Similarity**.
- Optimal for **fuzzy, semantically rich queries**.
- Chroma auto-handles persistence (manual `.persist()` deprecated).
- Limited indexing control, but fast to use and deploy.

### 2️⃣ FAISS IVF (DB_PRECISE)
- Designed for **fine-grained, high-fidelity matches**.
- Customizable via:
  - `nlist`: number of centroids (clusters)
  - `nprobe`: number of clusters probed during search
- Index built with:
  - `IndexIVFFlat`
  - `DistanceStrategy.EUCLIDEAN_DISTANCE` (L2)
- FAISS GPU installation attempted but failed (requires CUDA wheels). Sticking to CPU for now.

---

## 🧪 Chunking Optimization

- Implemented **token-based** and **character-based** splitters.
- For code: prefer token split (preserves syntax integrity).
- For dense academic documents: character splitter with recursive fallback.
- Introduced per-extension chunk configuration via `chunking_config.yaml`.
- Observed edge cases:
  - Some chunk sizes > 2000 tokens due to fallback logic not breaking large paragraphs.
- Ensured chunking remains **context-aware** yet avoids over-splitting.

---

## ⚙️ Engineering Improvements

- Created `IngestEngine` class with:
  - Dynamic loader selection based on extension
  - Smart splitter logic (TokenTextSplitter vs RecursiveCharacterSplitter)
  - Ignoring `.env`, vector db, and `__pycache__` directories
- Refined FAISS IVF Index build:
  - Vector conversion
  - Index training + probing
  - Metadata assignment via docstore
- Maintained dual vector store build via `run_full_ingestion()` method
- Extracted config into:
  - `chunking_config.yaml`
  - `ivf_config.yaml`

---

## 🧠 Thoughtful Design Decisions

- Decided to use **HuggingFace embeddings** for ingestion to avoid burning OpenAI API credits.
- Plan to use **OpenAI LLM (via API)** only for final query generation/inference.
- Designed the system to be **modular**, allowing later integration with:
  - MCP server (for auto-refresh + ingestion)
  - LoRA-finetuned local LLM via vLLM (for cost-efficient hosting)
- Introduced chunk splitting heuristics to ensure:
  - Smaller chunks for dense data
  - Larger coherent chunks for code files
- Realized the **importance of managing context window budgets** and how chunk overlap plays a role.

---

## 🗃️ Tomorrow's Goals

- Hook up retriever interface for querying
- Finalize LLM integration pipeline (OpenAI API first)
- Design logic to dynamically switch between DB_SEMANTIC and DB_PRECISE
- Consider hybrid retrieval (Chroma first → fallback to FAISS for edge cases)
- Start testing first prompts & refine QA quality

---

_🧠 “When your thoughts have structure, your code will follow.”_

