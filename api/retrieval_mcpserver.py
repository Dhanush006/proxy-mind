"""
retrieval_mcpserver.py
-----------------------
MCP server exposing a single tool: retrieve_context
- Chooses between Chroma (semantic) or FAISS (precise) retrieval
- Returns top-k document snippets as plain text
"""

import os
import sys
import time
import signal
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Graceful shutdown
def signal_handler(sig, frame):
    print("🛑 Shutting down MCP retrieval server gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- Configuration ---
load_dotenv()
CHROMA_PATH = "/Users/dhanushdshekar/Documents/Academia/Projects/proxy-mind/vector_dbs/db_semantic"
FAISS_PATH = "/Users/dhanushdshekar/Documents/Academia/Projects/proxy-mind/vector_dbs/db_precise"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Initialize embeddings ---
print("🔧 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --- Load vector DBs ---
print("🧠 Loading vector databases...")
try:
    chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print("✅ Chroma (semantic) loaded.")
except Exception as e:
    chroma_db = None
    print(f"⚠️ Could not load Chroma DB: {e}")

try:
    faiss_db = FAISS.load_local(FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS (precise) loaded.")
except Exception as e:
    faiss_db = None
    print(f"⚠️ Could not load FAISS DB: {e}")

# --- Initialize MCP server ---
mcp = FastMCP(
    name="retrieval-service",
    host="127.0.0.1",
    port=8081,
)

# --- Tool definition ---
@mcp.tool(name="retrieve_context", description="Retrieve relevant context passages from vector stores based on the query.")
def retrieve_context(query: str, retrieval_mode: str = "auto", top_k: int = 5) -> str:
    """
    Retrieve the most relevant context passages from the dual vector stores
    (ChromaDB and FAISS) based on the user's query.
    
    This tool forms part of the Model-Computer Protocol (MCP) architecture within
    the ProxyMind project. It allows an LLM or orchestration layer to dynamically
    retrieve project-specific knowledge for accurate and context-grounded responses.

    The underlying data includes material from:
      • Personal research papers and code repositories (e.g., solar irradiance forecasting, RAG systems, MCP tools)
      • Experimentation notes and documentation
      • Architecture and implementation discussions from local projects

    This tool is designed as part of the MCP (Model-Computer Protocol) service layer
    for Retrieval-Augmented Generation (RAG) applications. It acts as an intelligent
    retriever that interfaces with two complementary databases:
    
        1. **ChromaDB** — optimized for *semantic similarity* using cosine or dot-product
           distance metrics. Ideal for retrieving conceptually similar documents even when
           phrasing differs.
        
        2. **FAISS (IVF/L2)** — optimized for *precise vector distance* search and efficient
           large-scale retrieval. It ensures numerical and lexical proximity when exact or
           near-exact context is required.
    
    ### Parameters
    ----------
    query : str
        The natural-language query or question for which contextual information
        is to be retrieved.
    
    top_k : int, optional (default=4)
        The number of top documents/chunks to return from each database.
    
    mode : str, optional (default="auto")
        Determines how retrieval should be performed:
        
        - `"semantic"` : Only ChromaDB (semantic retriever) is used.
        - `"precise"`  : Only FAISS (precise retriever) is used.
        - `"auto"`     : The system dynamically decides based on query type.
          For example, factual/numeric queries trigger FAISS, whereas open-ended
          or conceptual queries trigger ChromaDB. In uncertain cases, results
          from both databases are merged and ranked.

    ### Returns
    -------
    str
        A formatted string containing the top retrieved passages, concatenated
        with source metadata (e.g., document name, chunk index, and similarity score),
        suitable for direct injection into an LLM prompt as context.

    ### Behavior
    ------------
    When invoked, this tool performs the following:
    
    1. Parses the query and determines retrieval strategy (based on `mode`).
    2. Loads the pre-built FAISS and/or ChromaDB vector stores from their configured paths.
    3. Executes nearest-neighbor searches in the selected stores.
    4. Merges, deduplicates, and sorts the retrieved results by relevance.
    5. Returns a clean, human-readable string of relevant context passages.

    ### Example
    --------
    ```python
    context = retrieve_context(
        query="Explain the effect of temperature on solar irradiance forecasting models",
        top_k=5,
        mode="auto"
    )
    print(context)
    ```
    
    Example Output:
    ```
    [Semantic Match: 'solar_forecasting_report.txt#3' | score=0.84]
    Temperature influences irradiance prediction by altering atmospheric scattering...

    [Precise Match: 'ensemble_model_notes.md#1' | score=0.81]
    The LSTM-based temperature predictor feeds its output into the irradiance model...
    ```

    ### Notes
    -----
    - Designed for use with Groq-based LLM orchestration through MCP.
    - Allows contextual decision-making by the LLM, which may choose whether
      to invoke this tool or answer directly from its own knowledge.
    - Ensures modular microservice-style integration for scalable RAG pipelines.
    """
    global chroma_db, faiss_db

    # Auto-mode heuristic
    if retrieval_mode == "auto":
        if len(query.split()) > 5 or any(w in query.lower() for w in ["explain", "architecture", "overview", "ProxyMind"]):
            retrieval_mode = "semantic"
        else:
            retrieval_mode = "precise"

    db = chroma_db if retrieval_mode == "semantic" else faiss_db
    if db is None:
        return f"❌ {retrieval_mode.title()} DB unavailable."

    try:
        docs = db.similarity_search(query, k=top_k)
        contents = [d.page_content for d in docs]
        response = {
            "retrieval_mode": retrieval_mode,
            "results": contents
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"⚠️ Retrieval failed: {e}"

# --- Run server ---
if __name__ == "__main__":
    print("🚀 Starting MCP retrieval server using stdio transport")
    try:
        mcp.run()  
    except Exception as e:
        print(f"Error starting server: {e}")
        time.sleep(3)