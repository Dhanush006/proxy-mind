# client/mcp_client.py
import asyncio
import os
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from rag.base_llm import BaseLLM
from rag.huggingface_llm import HuggingFaceLLM
from rag.ollama_llm import OllamaLLM

async def call_tool(session, name, **kwargs):
    result = await session.call_tool(name, kwargs)
    return result

async def run(query: str, llm_name="huggingface", db_type="semantic"):
    params = StdioServerParameters(command="uv", args=["run", "api/main.py"])
    async with stdio_client(server_params=params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Retrieve relevant documents from MCP server
            docs = await call_tool(session, "retrieval_tool", query=query, db_type=db_type, top_k=5)

    # Use retrieved docs directly with LLM
    # (Assuming you have a pre-defined LLM integration for HuggingFace or Ollama)
    llm: BaseLLM = OllamaLLM() if llm_name == "ollama" else HuggingFaceLLM()
    prompt = f"Context:\n\n{''.join(docs)}\n\nQuestion: {query}"
    answer = llm.invoke(prompt)
    return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--llm", choices=["ollama", "huggingface"], default="huggingface")
    parser.add_argument("--db", choices=["semantic", "precise"], default="semantic")
    args = parser.parse_args()

    result = asyncio.run(run(args.query, args.llm, args.db))
    print("\n📌 FINAL RESPONSE:\n", result)
