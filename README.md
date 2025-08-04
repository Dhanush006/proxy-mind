
# ProxyMind: Your Personal AI Interview Proxy

> **Tagline:** *"Your knowledge. Your voice. Delivered by your digital clone."*

ProxyMind is a Retrieval-Augmented Generation (RAG)-powered personal AI assistant trained on your own work — projects, code, blogs, and research. It acts as your AI **interview proxy**, ready to answer technical or project-related questions just as you would, enabling:

* 🎯 Interview preparation  
* 💼 Portfolio exploration  
* 🧠 Skill showcasing  
* 🧪 Self-reflection and learning  

---

## 🚀 Core Features

* **RAG-Powered Search:** Indexes and retrieves knowledge from your real work (GitHub repos, papers, docs).  
* **OpenAI Integration (v1):** Uses GPT-4 or GPT-3.5 to generate answers in your own tone and knowledge.  
* **Dynamic Self-Updating via MCP (v2):**
  * Live GitHub and project folder syncing  
  * Auto-discovery of new tools, code, and docs  
  * Serves the LLM with context-aware resources via a lightweight MCP interface  
* **Future Plans:**
  * Switchable LLM backend (e.g., via vLLM or HuggingFace Transformers)  
  * Integration with Ollama and local model support  
  * GitHub auto-sync  
  * Custom UI for querying  

---

## 🧠 About MCP Integration

> **MCP (Model Context Protocol)** is an evolving framework that allows AI agents to discover, manage, and utilize contextual resources dynamically.

In ProxyMind, a lightweight version of MCP is being developed:

* 🧭 Self-discovery of tools, repos, and learning logs (e.g., `thoughtstream/`)  
* 🔁 Dynamic vector DB updates when GitHub or notes change  
* 🔧 JSON/YAML-based tool and document registry  
* 📡 Planned: local FastAPI-based MCP server that can serve metadata to the RAG system  

**Why this matters:**  
As you grow, your assistant grows with you. New code? New blog? It finds it. It answers with it.

---

## 🛠️ Tech Stack

| Component      | Tool                                          |
| -------------- | --------------------------------------------- |
| LLM            | OpenAI (GPT-4) or local (future: vLLM)        |
| Embeddings     | `sentence-transformers` (e.g., MiniLM, mpnet) |
| Vector DB      | `ChromaDB`                                    |
| Pipeline       | `LangChain` + `LCEL`                          |
| Interface      | CLI (initial) + FastAPI (planned)             |
| Auto-Discovery | Custom MCP-like interface (planned)           |

---

## 📦 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/proxymind.git
cd proxymind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your `.env` file

```
OPENAI_API_KEY=your-openai-key
```

### 4. Add documents to `docs/`

Put your project READMEs, code snippets, blog exports, research papers in this folder. These will be indexed.

### 5. Run the RAG Agent

```bash
python run_proxymind.py
```

---

## 🔍 How It Works

```
Your Docs & Code
     ⬇
Text Splitter (LangChain)
     ⬇
Embeddings (MiniLM or OpenAI)
     ⬇
Stored in Vector DB (Chroma)
     ⬇
RAG Chain (Retriever + LLM)
     ⬇
Answer like You
```

---

## 🧬 Why the Name "ProxyMind"?

> It's your *mind by proxy*. A digital duplicate that reflects your understanding, your tone, and your narrative.

Perfect for job interviews, demo days, or showcasing your learning journey.

---

## 📚 Credits & Acknowledgements

* Powered by [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [OpenAI](https://openai.com/), and [Sentence Transformers](https://www.sbert.net/).  
* Inspired by research on Retrieval-Augmented Generation, LoRA fine-tuning, vector databases, and local model hosting.

---

## 💡 Future Ideas

* Deploy on Hugging Face Spaces or Render  
* Web UI with chat and file upload  
* Self-updating knowledge from GitHub activity (via MCP)  
* Multi-agent planning with dynamic tool routing  

---

## 🧠 Built by Dhanush D Shekar

Student @ Texas A&M University | AI Researcher | Firmware Engineer

> *Follow along as I build ProxyMind from scratch to production-ready AI agent.*
