# api/main.py
import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ingest.ingest_engine import IngestEngine

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize MCP Server
logger.debug("Initializing MCP server")
mcp = FastMCP("ProxyMind MCP Server")

# Define embedding
hf_model = "sentence-transformers/all-MiniLM-L6-v2"
logger.debug("Loading HuggingFace embeddings: %s", hf_model)
emb = HuggingFaceEmbeddings(model_name=hf_model)

CHROMA_PATH = r"D:\Git_Projects\ProxyMind\proxy-mind\vector_dbs\db_semantic"
FAISS_PATH = r"D:\Git_Projects\ProxyMind\proxy-mind\vector_dbs\db_precise"

logger.debug("Loading Chroma database from %s", CHROMA_PATH)
chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)
logger.debug("Loading FAISS database from %s", FAISS_PATH)
faiss = FAISS.load_local(FAISS_PATH, embeddings=emb, allow_dangerous_deserialization=True)

# -- Retrieval Tool --
@mcp.tool()
def retrieve_relevant_personal_projects_and_data_tool(query: str, db_type: str = "semantic", top_k: int = 5) -> list[str]:
    logger.debug("Executing retrieval tool with query: %s, db_type: %s, top_k: %d", query, db_type, top_k)
    retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": top_k}) \
        if db_type == "semantic" else faiss.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    logger.info("Retrieved %d documents for query: %s", len(docs), query)
    return [d.page_content for d in docs]

# -- Updater Tool --
@mcp.tool()
def updater_relevant_personal_projects_and_data_tool() -> str:
    logger.debug("Executing updater tool")
    ingest = IngestEngine()
    ingest.run_full_ingestion()
    logger.info("Full ingestion completed successfully")
    return "✅ Full ingestion completed successfully"

if __name__ == "__main__":
    logger.info("Starting MCP server")
    mcp.run(transport="stdio")