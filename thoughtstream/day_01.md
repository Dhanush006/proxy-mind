# 🧠 Day 01: Foundations of Proximind — RAG, MCP, and Local LLM Ecosystem

## 📌 What I’m Building

I'm building **Proximind**, a RAG-powered AI agent that reflects my thinking, work, and learning. The goal is to create an intelligent assistant that can:
- Answer questions about my projects (as if it were me)
- Reflect my thought process and problem-solving journey
- Grow dynamically with my learning and GitHub updates

---

## 📚 Today’s Topics

### 1. 🔁 Retrieval-Augmented Generation (RAG)

- RAG combines **retrievers** and **LLMs**.
- My documents (project code, blogs, notes) will be split into chunks, embedded using models like `MiniLM`, and stored in **ChromaDB**.
- During a query, the top-K relevant chunks are retrieved and sent to the LLM to generate a context-aware answer.

### 2. 🌐 What is MCP (Model Context Protocol)?

- MCP is a **conceptual interface for tool-using agents**.
- It defines how an LLM can:
  - Discover tools it has access to
  - Query or select between multiple resources (docs, APIs, retrievers)
  - Respond dynamically based on what it finds

In Proximind:
- I plan to implement a lightweight **MCP-like system** using:
  - A `tool_registry.json` or `.yaml` config
  - GitHub auto-sync and ingestion monitoring
  - A FastAPI server to serve metadata to the agent if needed

### 3. 🧠 Local LLM Stack

I learned about the differences between cloud-hosted and local LLMs:

| Stack | Tools |
|-------|-------|
| Cloud | OpenAI API (GPT-3.5/4) |
| Local | `transformers`, `ollama`, `vllm`, `llama.cpp` |

- `vLLM`: For **fast local inference** and serving OpenAI-style `/chat/completions` endpoints
- `ollama`: Simplifies running quantized models (e.g., Mistral, LLaMA2)
- `huggingface/transformers`: Lets me download and use hundreds of open-source LLMs

### 4. 🔧 LoRA + QLoRA

I explored **parameter-efficient fine-tuning**:

- **LoRA (Low-Rank Adaptation):** Adds small adapter matrices to the base model instead of updating all weights
- **QLoRA:** Enables fine-tuning quantized models (INT4) with LoRA — massively reduces resource requirements

> With LoRA, I can fine-tune a 7B model on Colab without needing A100s.

### 5. ⚖️ Quantization

Quantization shrinks model size by reducing weight precision (e.g., from FP32 to INT4).

- Reduces memory & speeds up inference
- Tools: `bitsandbytes`, `AutoGPTQ`, `GGUF` (for llama.cpp)

---

## 🧭 Next Steps

- Create `proxymind-learnings` repo to hold this journal
- Begin ingestion from this folder into the RAG system (`docs/thoughtstream`)
- Add tool discovery config (`tool_registry.yaml`)
- Scaffold a basic FastAPI MCP server (optional but helpful)

---

## ✍️ Reflection

It’s empowering to know I can control and grow my own AI agent — not just a chatbot, but a mirror of my technical identity. Day 1 gave me confidence that the pieces are all within reach: RAG, embeddings, vector stores, LLMs, and the glue that binds them — MCP.

> *Tomorrow: Start implementing the ingestion logic, followed by a CLI query interface.*

